#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import cv2
import csv
import PIL
import PIL.Image as Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json

import src.resnet as resnet_model
import jittor as jt
import jittor
import jittor.transform as transforms


from cutler.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint="/home/zixuanqian/tmp/cutler/sam_vit_b_01ec64.pth")
#sam.to('cuda')

import cutler.dino as dino
# modfied by Xudong Wang based on third_party/TokenCut
from third_party.TokenCut.unsupervised_saliency_detection import utils, metric
from third_party.TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
# bilateral_solver codes are modfied based on https://github.com/poolio/bilateral_solver/blob/master/notebooks/bilateral_solver.ipynb
# from third_party.TokenCut.unsupervised_saliency_detection.bilateral_solver import BilateralSolver, BilateralGrid
# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from cutler.crf import densecrf

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.ImageNormalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = jt.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix

    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def maskcut_forward(img, feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    """
    Implementation of MaskCut.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      N: number of pseudo-masks per image.
    """

    # centroids = np.load("/home/zixuanqian/tmp/cutler/maskcut/centroids.npy")
    # centroids = torch.Tensor(np.array(centroids)).to('cpu')
    # centroids = torch.nn.functional.normalize(centroids, dim=1, p=2)
    #
    # model = resnet_model.__dict__['resnet18'](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')
    # checkpoint = jt.load("/home/zixuanqian/tmp/cutler/maskcut/checkpoint.pth.tar")['state_dict']
    # for k in list(checkpoint.keys()):
    #     if k.startswith('module.'):
    #         checkpoint[k[len('module.'):]] = checkpoint[k]
    #         del checkpoint[k]
    #         k = k[len('module.'):]
    #     if k not in model.state_dict().keys():
    #         del checkpoint[k]
    # model.load_state_dict(checkpoint)
    # model.eval()
    #
    # transform = jittor.transform.Compose([
    #     jittor.transform.Resize(256),
    #     jittor.transform.ToTensor(),
    #     jittor.transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ]
    # )
    # data = transform(img)
    # data = jittor.array(jittor.unsqueeze(data, dim=0))
    # h = img.height
    # w = img.width
    # out, mask = model(data, mode='inference_pixel_attention')
    # out = torch.Tensor(np.array(out)).to('cpu')
    # mask = torch.Tensor(np.array(mask)).to('cpu')
    # mask = torch.nn.functional.interpolate(mask, size=(h,w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    # out = torch.nn.functional.normalize(out, dim=1, p=2)
    # B, C, H, W = out.shape
    # out = out.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)  # (32*32,512)
    #
    # cosine = torch.matmul(out, centroids.t())
    # cosine = cosine.view(1, H, W, 50).permute(0, 3, 1, 2)
    #
    # logit = mask
    # prediction = torch.max(cosine, dim=1, keepdim=True)[0]
    # prediction = torch.nn.functional.interpolate(prediction.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)
    # prediction = prediction.numpy()

    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = jt.array(np.zeros(dims))
            #if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        # temp1 = np.argmax(second_smallest_vec)
        # temp2 = bipartition.reshape(dims).astype(float)
        # _, _, _, cc = detect_box(temp2, temp1, dims, scales=scales, initial_im_size=init_image_size)
        # pseudo_mask = np.zeros(dims)
        # pseudo_mask[cc[0],cc[1]] = 1
        # pseudo_mask = torch.from_numpy(pseudo_mask)
        # temp3 = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=(h,w),
        #                             mode='nearest').squeeze().numpy()

        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        # if np.sum(np.where(temp3>0, prediction, 0)) / np.sum(temp3) < (np.sum(prediction) - np.sum(np.where(temp3>0, prediction, 0))) / (h*w - np.sum(temp3)):
        #     reverse = True
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = jt.array(pseudo_mask)
        #if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = jt.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            # temp4 = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=(h, w),
            #                       mode='nearest').squeeze().to('cpu')
            # threshold = torch.sum(torch.where(temp4>0, logit, 0)) / torch.sum(temp4)
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = jt.array(pseudo_mask)
                #if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = jt.nn.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartition_masked = bipartition.numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = jt.array(eigvec)
        #if not cpu: eigvec = eigvec.to('cuda')
        eigvec = jt.nn.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.numpy())

    return seed, bipartitions, eigvecs

def maskcut(img_path, backbone,patch_size, tau, N=1, fixed_size=480, cpu=False) :
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize)
    tensor = jt.unsqueeze(tensor, dim=0)
    #if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]

    _, bipartition, eigvec = maskcut_forward(I.copy(), feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N, cpu=cpu)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, I_new

def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask,
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided,
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None
    else:
        if bounding_box is None:
            bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}

def show_mask(mask_, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask_.shape[-2:]
    mask_image = mask_.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def visualize(img, mask_, input_point, input_label, args, img_folder, img_name):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask_, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.savefig(os.path.join(args.vis_dir, img_folder, img_name))

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser('MaskCut script')
    # default arguments
    parser.add_argument('--out-dir', type=str, default="../result8/", help='output directory')
    parser.add_argument('--vit-arch', type=str, default='tiny', choices=['base', 'small', 'tiny'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--nb-vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

    # additional arguments
    parser.add_argument('--dataset-path', type=str, default="/home/zixuanqian/tmp/ImageNetS50/ImageNetS50/validation/", help='path to the dataset')
    parser.add_argument('--tau', type=float, default=0.6, help='threshold used for producing binary graph')
    parser.add_argument('--num-folder-per-job', type=int, default=50, help='the number of folders each job processes')
    parser.add_argument('--job-index', type=int, default=0, help='the index of the job (for imagenet: in the range of 0 to 1000/args.num_folder_per_job-1)')
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default="/home/zixuanqian/tmp/cutler/maskcut/dino_tiny.pth", help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--vis-dir', type=str, default="../vis8")
    parser.add_argument('--dump-path', type=str, default=None)

    args = parser.parse_args()

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384
    elif args.vit_arch == 'tiny' and args.patch_size ==16:
        feat_dim = 192
    elif args.vit_arch == 'small' and args.patch_size ==16:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print (msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    img_folders = os.listdir(args.dataset_path)

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    cate_query = {}
    with open("/home/zixuanqian/tmp/cutler/datasets/dataset/ImageNetS50/labels.txt", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            class_id, class_name = row
            cate_query[str(class_id)] = int(class_name) + 1

    # with open("/home/zixuanqian/tmp/cutler/result.json", "r", encoding="utf-8") as f:
    #     content = json.load(f)
    #     content = content[6]
    #     content['predictions'] = [pred+1 for pred in content['predictions']]

    # clustercentroids = np.load("/home/zixuanqian/tmp/cutler/maskcut/centroids.npy")
    # clustercentroids = torch.Tensor(np.array(clustercentroids)).to('cpu')
    # clustercentroids = torch.nn.functional.normalize(clustercentroids, dim=1, p=2)
    # model = resnet_model.__dict__['resnet18'](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')
    # checkpoint = jt.load("/home/zixuanqian/tmp/cutler/maskcut/checkpoint.pth.tar")['state_dict']
    # for k in list(checkpoint.keys()):
    #     if k.startswith('module.'):
    #         checkpoint[k[len('module.'):]] = checkpoint[k]
    #         del checkpoint[k]
    #         k = k[len('module.'):]
    #     if k not in model.state_dict().keys():
    #         del checkpoint[k]
    # model.load_state_dict(checkpoint)
    # model.eval()
    model = resnet_model.__dict__["resnet18"](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=50)
    #checkpoint = jt.load("/home/zixuanqian/tmp/origin/vit_pass4/pixel_finetune/checkpoint.pth.tar")["state_dict"]
    checkpoint = jt.load("/home/zixuanqian/tmp/origin/modify_pixel_finetune4/checkpoint.pth.tar")["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    model.eval()

    transform = jittor.transform.Compose([
        jittor.transform.Resize(256),
        jittor.transform.ToTensor(),
        jittor.transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    )

    start_idx = max(args.job_index*args.num_folder_per_job, 0)
    end_idx = min((args.job_index+1)*args.num_folder_per_job, len(img_folders))

    image_id, segmentation_id = 1, 1
    image_names = []
    for img_folder in img_folders[start_idx:end_idx]:
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        img_list = sorted(os.listdir(args.img_dir))

        for img_name in tqdm(img_list) :
            # get image path
            img_path = os.path.join(args.img_dir, img_name)
            # get pseudo-masks for each image using MaskCut
            try:
                bipartitions, _, I_new = maskcut(img_path, backbone, args.patch_size, \
                    args.tau, N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)
            except:
                print(f'Skipping {img_name}')
                continue

            I = Image.open(img_path).convert('RGB')
            width, height = I.size
            box_prompt = []
            res = []
            for idx, bipartition in enumerate(bipartitions):
                # post-process pesudo-masks with CRF
                pseudo_mask = densecrf(np.array(I_new), bipartition)
                pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

                # filter out the mask that have a very different pseudo-mask after the CRF
                mask1 = jt.array(bipartition)
                mask2 = jt.array(pseudo_mask)
                # if not args.cpu:
                #     mask1 = mask1.cuda()
                #     mask2 = mask2.cuda()
                if metric.IoU(mask1, mask2) < 0.5:
                    pseudo_mask = pseudo_mask * -1

                # construct binary pseudo-masks
                pseudo_mask[pseudo_mask < 0] = 0
                pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
                pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

                res.append(pseudo_mask)

                # create coco-style image info
                if img_name not in image_names:
                    # image_info = create_image_info(
                    #     image_id, "{}/{}".format(img_folder, img_name), (height, width, 3))
                    # output["images"].append(image_info)
                    image_names.append(img_name)

                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                # if annotation_info is not None:
                #     if annotation_info['bbox'] is not None:
                #         box_prompt.append(annotation_info['bbox'])
                if annotation_info is not None:
                    #output["annotations"].append(annotation_info)
                    segmentation_id += 1

            # if not os.path.exists(os.path.join(args.dump_path, img_folder)):
            #     os.makedirs(os.path.join(args.dump_path, img_folder))
            if not os.path.exists(os.path.join(args.vis_dir, img_folder)):
                os.makedirs(os.path.join(args.vis_dir, img_folder))
            if not os.path.exists(os.path.join(args.out_dir, img_folder)):
                os.makedirs(os.path.join(args.out_dir, img_folder))


            data = transform(I)
            data = jittor.array(jittor.unsqueeze(data, dim=0))
            h = I.height
            w = I.width

            output = model(data)
            output = jt.nn.interpolate(output, (h, w), mode="bilinear", align_corners=False)
            prediction = jt.argmax(output, dim=1, keepdims=True)[0]
            prediction = prediction.squeeze()
            prediction = prediction.numpy()
            res = np.array(res)
            res = np.sum(res, axis=0).astype(np.uint8)
            res[res>=1] = 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(res.copy(), connectivity=8)
            dropsmall = []
            for i in range(stats.shape[0]):
                if stats[i][4] < I.height * I.width * 0.01:
                    dropsmall.append(i)
            if len(dropsmall) > 0:
                centroids = np.delete(centroids, np.array(dropsmall), axis=0)
            predictor = SamPredictor(sam)
            predictor.set_image(np.array(I))
            input_point = np.array(centroids)
            input_label = np.array([1 for _ in range(input_point.shape[0])])
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=None,
                multimask_output=True,
            )
            bestmask = masks[0]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bestmask.copy().astype(np.uint8),connectivity=8)
            for i in range(num_labels):
                if stats[i][4] > I.height * I.width * 0.01:
                    obj = np.where(labels==i)
                    obj = np.array(obj)
                    prediction_ = prediction[obj[0,:], obj[1,:]].flatten()
                    unique_elements, counts = np.unique(prediction_, return_counts=True)
                    index = np.argmax(counts)
                    vote_sementics = unique_elements[index]
                    prediction = np.where(labels == i, vote_sementics, prediction)

            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bestmask.copy().astype(np.uint8), connectivity=8)
            # p = output.numpy().squeeze()
            # for i in range(num_labels):
            #     obj = np.where(labels==i)
            #     obj = np.array(obj)
            #     p_ = p[:, obj[0,:], obj[1,:]]
            #     vote_sementics_index = np.argmax(p_)
            #     vote_sementics_index = np.unravel_index(vote_sementics_index, p_.shape)
            #     vote_sementics = np.array(vote_sementics_index)[0]
            #     prediction = np.where(labels==i, vote_sementics, prediction)

            #visualize(np.array(I), bestmask.copy(), input_point, input_label, args, img_folder, img_name)
            # mask_ = np.where(bestmask>0)
            # mask_ = np.array(mask_)
            # for i in range(len(mask_[0])):
            #     if prediction[mask_[0,i], mask_[1,i]] > 0:
            #         continue
            #     else:
            #         prediction[mask_[0,i], mask_[1,i]] = torch.argsort(output[:,:,mask_[0,i], mask_[1,i]], dim=1, descending=True).squeeze()[1]
            trimmed_prediction = np.where(bestmask>0, prediction, 0)
            submitmask = np.zeros((I.height, I.width, 3))
            submitmask[:, :, 0] = trimmed_prediction % 256
            submitmask[:, :, 1] = trimmed_prediction // 256
            suffix = img_name.replace('JPEG', 'png')
            submitmask = Image.fromarray(submitmask.astype(np.uint8))
            submitmask.save(os.path.join(args.out_dir, img_folder, suffix))



            # out, rawmask = model(data, mode='inference_pixel_attention')
            # out = torch.Tensor(np.array(out)).to('cpu')
            # rawmask = torch.Tensor(np.array(rawmask)).to('cpu')
            # rawmask = torch.nn.functional.interpolate(rawmask, size=(h, w), mode="bilinear",align_corners=False).squeeze()
            #
            # out = torch.nn.functional.normalize(out, dim=1, p=2)
            # B, C, H, W = out.shape
            # out = out.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C) # (h*w,512)
            #
            # cosine = torch.matmul(out, clustercentroids.t())
            # cosine = cosine.view(1, H, W, 50).permute(0, 3, 1, 2)
            #
            # logit = rawmask.numpy()
            # prediction = torch.max(cosine, dim=1, keepdim=True)[0]
            # temp = prediction.numpy()
            # sementics = torch.max(cosine, dim=1, keepdim=True)[1] + 1
            # sementics = sementics.numpy()
            # sementics_id = sementics[np.unravel_index(np.argmax(temp), temp.shape)]
            # prediction = torch.nn.functional.interpolate(prediction.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)
            # prediction = prediction.numpy()
            # # sementics = torch.nn.functional.interpolate(sementics.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)
            # # sementics = sementics.numpy().astype(np.int8)

            # res = np.array(res)
            # res = np.sum(res, axis=0).astype(np.uint8)
            # res[res>1] = 1
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(res.copy(), connectivity=8)
            # dropsmall = []
            # centroids1 = centroids
            # for i in range(stats.shape[0]):
            #     if stats[i][4] < I.height * I.width * 0.005:
            #         dropsmall.append(i)
            # if len(dropsmall) > 0:
            #     centroids1 = np.delete(centroids1, np.array(dropsmall), axis=0)
            # dropconfuse = []
            # for i in range(centroids1.shape[0]):
            #     point = centroids1[i,:].astype(np.int8)
            #     if prediction[point[1], point[0]] < 0.6:
            #         dropconfuse.append(i)
            # if len(dropconfuse) > 0:
            #     centroids1 = np.delete(centroids1, np.array(dropconfuse), axis=0)
            # attention = np.where(prediction>0.7, 1, 0).astype(np.int8)
            # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(attention.copy(), connectivity=8)
            # dropsmall = []
            # for i in range(stats.shape[0]):
            #     if stats[i][4] < I.height * I.width * 0.005:
            #         dropsmall.append(i)
            # centroids2 = centroids
            # if len(dropsmall) > 0:
            #     centroids2 = np.delete(centroids, np.array(dropsmall), axis=0)
            # # centroids2 = []
            # # for i in range(1, num_labels+1):
            # #     temp = np.unravel_index(np.argmax(np.where(labels==i, prediction, 0)), prediction.shape)
            # #     centroids2.append(temp)
            # centroids3 = np.concatenate([centroids1, np.array(centroids2)], axis=0)
            # predictor = SamPredictor(sam)
            # predictor.set_image(np.array(I))
            # input_point = np.array(centroids3)
            # input_label = np.array([1 for _ in range(input_point.shape[0])])
            # masks, scores, logits = predictor.predict(
            #     point_coords=input_point,
            #     point_labels=input_label,
            #     box=None,
            #     multimask_output=True,
            # )
            # masks = masks[0]
            # visualize(np.array(I), masks, input_point, input_label, args, img_folder, img_name)
            # submitmask = np.zeros((I.height, I.width, 3))
            # #mask_applysementics = np.where(masks>0, sementics, 0)
            # #mask_applysementics = masks * sementics_id
            # #mask_applysementics = masks * cate_query[str(img_folder)]
            # lookupid = img_name.replace('.JPEG', '')
            # lookupid = content['lookup'].index(lookupid)
            # sementics_id = content['predictions'][lookupid]
            # mask_applysementics = masks * sementics_id
            # submitmask[:, :, 0] = mask_applysementics % 256
            # submitmask[:, :, 1] = mask_applysementics // 256
            # suffix = img_name.replace('JPEG', 'png')
            # suffix_temp = img_name.replace('JPEG', 'npy')
            # submitmask = Image.fromarray(submitmask.astype(np.uint8))
            # submitmask.save(os.path.join(args.dump_path, img_folder, suffix))
            # np.save(os.path.join(args.dump_path, img_folder, suffix_temp), logit)



            # box_prompt = torch.Tensor(np.array(box_prompt)).to('cuda')
            # if box_prompt.shape[0] > 0:
            #     input_box = predictor.transform.apply_boxes_torch(boxes=box_prompt,original_size=(I.height, I.width))
            #     masks, _, _ = predictor.predict_torch(
            #         point_coords=None,
            #         point_labels=None,
            #         boxes=input_box,
            #         multimask_output=True,
            #     )
            #     masks= masks[:,0,:,:].squeeze(axis=1).to('cpu').numpy()
            #     masks = np.sum(masks, axis=0)
            #     masks[masks>1] = 1
            #     visualize(masks.copy() * 255, box_prompt.cpu(), args, img_folder, img_name)
            #     submitmask = np.zeros((I.height, I.width, 3))
            #     masks = masks * cate_query[str(img_folder)]
            #     submitmask[:, :, 0] = masks % 256
            #     submitmask[:, :, 1] = masks // 256
            # else:
            #     submitmask = np.zeros((I.height, I.width, 3))
            # suffix = img_name.replace('JPEG','png')
            # cv2.imwrite(os.path.join(args.dump_path, img_folder, suffix), submitmask)


            # res = np.array(res)
            # res = np.sum(res, axis=0)
            # res[res>=1] = 1
            # res = res * 255
            # res = Image.fromarray(res.astype(np.uint8))
            # res.save(os.path.join(args.out_dir, img_folder, img_name))
            # cv2.imwrite(os.path.join(args.vis_dir, img_folder, img_name), res.copy()*255)
            # suffix = img_name.replace('JPEG', 'png')
            # submitmask = np.zeros((I.height, I.width, 3))
            # res = res * cate_query[str(img_folder)]
            # submitmask[:, :, 0] = res % 256
            # submitmask[:, :, 1] = res // 256
            # submitmask = Image.fromarray(submitmask.astype(np.uint8))
            # submitmask.save(os.path.join(args.out_dir, img_folder, suffix))

            image_id += 1

    # save annotations
    # if len(img_folders) == args.num_folder_per_job and args.job_index == 0:
    #     json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N)
    # else:
    #     json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, start_idx, end_idx)
    # with open(json_name, 'w') as output_json_file:
    #     json.dump(output, output_json_file)
    # print(f'dumping {json_name}')
    # print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
