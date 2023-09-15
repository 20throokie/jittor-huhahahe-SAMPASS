# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path.append("/home/zixuanqian/tmp/origin/src/")
import math
import os
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import shutil
import time
from logging import getLogger
import argparse
import copy

import numpy as np
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1

import jittor.transform as transforms
import src.pseudo_transforms as custom_transforms
import src.resnet as resnet_models
from options import getOption
from src.singlecropdataset import PseudoLabelDataset
from src.utils import (AverageMeter, accuracy, fix_random_seeds, initialize_exp, restart_from_checkpoint)

logger = getLogger()
#parser = getOption()


def main():
    global args
    parser = argparse.ArgumentParser(description="Finetune")
    parser.add_argument("--dump_path", type=str, default='./modify_pixel_finetune6', help="The path to save results.")
    parser.add_argument("--dump_checkpoints", type=str, default='./modify_pixel_finetune6')
    parser.add_argument("--data_path", type=str, default="/home/zixuanqian/tmp/ImageNetS50/ImageNetS50/train", help="The path to ImagenetS dataset.")
    parser.add_argument("--pretrained", type=str, default="/home/zixuanqian/tmp/origin/vit_pass4/pixel_finetune/checkpoint.pth.tar", help="The model checkpoint file.")
    parser.add_argument("-a", "--arch", metavar="ARCH", default='resnet18', help="The model architecture.")
    parser.add_argument("-c", "--num-classes", default=50, type=int, help="The number of classes.")
    parser.add_argument("--checkpoint_key", type=str, default='state_dict', help="key of model in checkpoint")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument('--base_lr', default=1)
    parser.add_argument('--final_lr', default=0.01)
    parser.add_argument('--start_warmup', default=0.01)
    parser.add_argument('--wd', default=0.000001)
    parser.add_argument('--warmup_epochs', default=0)
    parser.add_argument('--workers', default=0)
    parser.add_argument('--seed', default=31)
    parser.add_argument('--checkpoint_freq', default=2)
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--pseudo_path', default="/home/zixuanqian/tmp/origin/vit_pass4/pixel_attention/train/")
    parser.add_argument('--finetune_scale_factor', default=0.5)

    args = parser.parse_args()
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, 'epoch', 'loss')

    # build data
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = PseudoLabelDataset(
        args.data_path,
        custom_transforms.Compose([
            custom_transforms.RandomResizedCropSemantic(224),
            custom_transforms.RandomHorizontalFlipSemantic(),
            custom_transforms.ToTensorSemantic(),
            normalize,
        ]),
        pseudo_path=args.pseudo_path,
    )
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True
    )
    logger.info('Building data done with {} images loaded.'.format(
        len(train_dataset)))

    # build model
    if args.arch in resnet_models.__dict__.keys():
        model = resnet_models.__dict__[args.arch](
            hidden_mlp=0,
            output_dim=0,
            nmb_prototypes=0,
            num_classes=args.num_classes,
            train_mode='finetune')
    else:
        raise NotImplementedError()

    if jt.in_mpi:
        for n, p in model.named_parameters():
            p.assign(p.mpi_broadcast())

    for name, param in model.named_parameters():
        # logger.info(f'**********************************************')
        # logger.info(f'{name}')
        # logger.info(f'**********************************************')
        if "backbone" in name or "last_layer" in name:
            param.requires_grad = False

    # loading pretrained weights
    checkpoint = jt.load(args.pretrained)[args.checkpoint_key]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    logger.info("Loaded pretrained weights '{}'".format(args.pretrained))

    # copy model to GPU
    if jt.rank == 0:
        logger.info(model)
    logger.info('Building model done.')

    # build optimizer
    if args.optim == 'sgd':
        optimizer = jt.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
    else:
        raise NotImplementedError()

    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr,
                                     len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = \
        np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 +
            math.cos(
                math.pi * t /
                (len(train_loader) * (args.epochs - args.warmup_epochs))
            )
        )
                for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info('Building optimizer done.')

    # optionally resume from a checkpoint
    to_restore = {'epoch': 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, 'checkpoint.pth.tar'),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer
    )
    start_epoch = to_restore['epoch']

    # loss function
    criterion = nn.CrossEntropyLoss()

    init_weight = {}
    for k in list(checkpoint.keys()):
        if k in "last_layer":
            init_weight[k] = checkpoint[k]

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info('============ Starting epoch %i ... ============' % epoch)

        # train the network
        scores = train(train_loader, model, optimizer, criterion, epoch,
                       lr_schedule, init_weight)
        training_stats.update(scores)

        # save checkpoints
        if jt.rank == 0:
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            jt.save(
                save_dict,
                os.path.join(args.dump_path, 'checkpoint.pth.tar'),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, 'checkpoint.pth.tar'),
                    os.path.join(args.dump_checkpoints,
                                 'ckp-' + str(epoch) + '.pth.tar'),
                )
        jt.sync_all()


def train(train_loader, model, optimizer, criterion, epoch, lr_schedule, init_weight):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    peer_losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    end = time.time()

    for it, (inputs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr_schedule[iteration] * param_group['lr_scale']
            else:
                param_group['lr'] = lr_schedule[iteration]

        # ============ forward step ... ============
        labels = labels[:, 1, :, :] * 256 + labels[:, 0, :, :]
        labels = labels.long()
        labels = nn.interpolate(labels.float().unsqueeze(1),
                               scale_factor=args.finetune_scale_factor,
                               mode='nearest').long().squeeze(1)
        embedding = model.execute_backbone(inputs)
        # pseudo_labels = model.last_layer(embedding).detach()
        # pseudo_labels = jt.argmax(pseudo_labels, dim=1, keepdims=False)[0]
        output = model.up1(embedding)
        output = jt.nn.interpolate(output, scale_factor=4, mode="bilinear", align_corners=False)
        output = model.up2(output)
        output = nn.interpolate(output,
                               size=(labels.shape[1], labels.shape[2]),
                               mode='bilinear')
        shuffled_labels = labels[jt.randperm(labels.shape[0]),:,:]

        peer_loss = bootstrapingloss(output, shuffled_labels, labels, epoch)
        loss = peer_loss

        (acc1, ) = accuracy(
            output.permute(0, 2, 3, 1).contiguous().view(-1, output.shape[1]),
            labels.view(-1))

        # ============ backward and optim step ... ============
        optimizer.step(loss)

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        peer_losses.update(peer_loss.item(), inputs[0].size(0))
        acc.update(acc1.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if jt.rank == 0 and it % 50 == 0:
            logger.info('Epoch: [{0}][{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'PeerLoss {peer_loss.val:.4f} ({peer_loss.avg:.4f})\t'
                        'Acc@1 {acc1.val:.2f} ({acc1.avg:.2f})\t'
                        'Lr: {lr:.8f}'.format(
                            epoch,
                            it,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            peer_loss=peer_losses,
                            acc1=acc,
                            lr=optimizer.param_groups[-1]['lr'],
                        ))
    return (epoch, losses.avg)


def bootstrapingloss(output, shuffled_labels, authentic_labels, epoch):

    # peer_loss = jt.nn.cross_entropy_loss(output, authentic_labels) - \
    #             0.1 * jt.nn.cross_entropy_loss(output, shuffled_labels)
    peer_loss = jt.nn.cross_entropy_loss(output, authentic_labels)
    # output_ = jt.nn.softmax(output, dim=1)
    # output_topk = jt.topk(output_, dim=1, k=2)[0]
    # uncertainty_loss = 1 - (output_topk[:, 0, :, :] - output_topk[:, 1, :, :]).mean()
    return peer_loss


if __name__ == '__main__':
    main()
