import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

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
from pycocotools import mask
import pycocotools.mask as mask_util

import src.resnet as resnet_model
import jittor as jt
jt.flags.use_cuda = 1
import jittor
import jittor.transform as transforms

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

def visualize(img, mask_, input_point, input_label, img_folder, img_name, isbackground):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask_, plt.gca())
    show_points(input_point, input_label, plt.gca())
    if isbackground:
        if not os.path.exists(os.path.join("./visbackground", img_folder)):
            os.makedirs(os.path.join("./visbackground", img_folder))
        plt.savefig(os.path.join("./visbackground", img_folder, img_name))
    else:
        if not os.path.exists(os.path.join("./visforeground", img_folder)):
            os.makedirs(os.path.join("./visforeground", img_folder))
        plt.savefig(os.path.join("./visforeground", img_folder, img_name))
    return

def visualize2(img, mask_, img_folder, img_name):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask_, plt.gca())
    if not os.path.exists(os.path.join("./vis_example", img_folder)):
        os.makedirs(os.path.join("./vis_example", img_folder))
    plt.savefig(os.path.join("./vis_example", img_folder, img_name))

    return

from cutler.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
sam = sam_model_registry["vit_b"](checkpoint="/home/zixuanqian/tmp/cutler/sam_vit_b_01ec64.pth")

model = resnet_model.__dict__["resnet18"](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune',
                                          num_classes=50)
# checkpoint = jt.load("/home/zixuanqian/tmp/origin/vit_pass4/pixel_finetune/checkpoint.pth.tar")["state_dict"]
checkpoint = jt.load("/home/zixuanqian/tmp/origin/modify_pixel_finetune5/checkpoints/ckp-14.pth.tar")["state_dict"]
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

dataset_path = "/home/zixuanqian/tmp/ImageNetS50/ImageNetS50/validation/"
dump_path = "./val_result"
img_folders = os.listdir(dataset_path)
for img_folder in img_folders:
    img_dir = os.path.join(dataset_path, img_folder)
    img_list = sorted(os.listdir(img_dir))
    for img_name in tqdm(img_list):
        img_path = os.path.join(img_dir, img_name)
        I = Image.open(img_path).convert('RGB')
        data = transform(I)
        data = jittor.array(jittor.unsqueeze(data, dim=0))
        h = I.height
        w = I.width
        # predictor = SamPredictor(sam)
        # predictor.set_image(np.array(I))
        mask_generator = SamAutomaticMaskGenerator(sam)

        output = model.execute_backbone(data)
        output = model.up1(output)
        output = jt.nn.interpolate(output, scale_factor=4, mode="bilinear", align_corners=False)
        output = model.up2(output)
        output = jt.nn.interpolate(output, (h, w), mode="bilinear", align_corners=False)
        prediction = jt.argmax(output, dim=1, keepdims=True)[0]
        prediction = prediction.squeeze()
        prediction = prediction.numpy()

        if not os.path.exists(os.path.join(dump_path, img_folder)):
            os.makedirs(os.path.join(dump_path, img_folder))

        # sementics, area = np.unique(prediction, return_counts=True)
        #
        # submitmask = np.zeros((I.height, I.width, 3))

        # for sementics_ in sementics:
        #     if sementics_ != 0:
        #         mask = np.where(prediction==sementics_, 1, 0)
        #         mask = mask.astype(np.uint8)
        #         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.copy(), connectivity=8)
        #         dropsmall = []
        #         for i in range(stats.shape[0]):
        #             if stats[i][4] < I.height * I.width * 0.01:
        #                 dropsmall.append(i)
        #         if len(dropsmall) > 0:
        #             centroids = np.delete(centroids, np.array(dropsmall), axis=0)
        #         input_point = np.array(centroids)
        #         input_label = np.array([1 for _ in range(input_point.shape[0])])
        #         masks, scores, logits = predictor.predict(
        #             point_coords=input_point,
        #             point_labels=input_label,
        #             box=None,
        #             mask_input=None,
        #             multimask_output=True,
        #         )
        #         bestmask = masks[0]
        #         submitmask[:, :, 0] = np.where(bestmask==1, sementics_, submitmask[:, :, 0])
        #
        # visualize(np.array(I), bestmask.copy(), input_point, input_label, img_folder, img_name, 0)
        # suffix = img_name.replace('JPEG', 'png')
        # submitmask = Image.fromarray(submitmask.astype(np.uint8))
        # submitmask.save(os.path.join(dump_path, img_folder, suffix))


        # mask = np.where(prediction>0, 1, 0)
        # mask = mask.astype(np.uint8)
        # # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.copy(), connectivity=8)
        # # dropsmall = []
        # # for i in range(stats.shape[0]):
        # #     if stats[i][4] < I.height * I.width * 0.01:
        # #         dropsmall.append(i)
        # # if len(dropsmall) > 0:
        # #     centroids = np.delete(centroids, np.array(dropsmall), axis=0)
        # # input_point = np.array(centroids)
        # # input_label = np.array([1 for _ in range(input_point.shape[0])])
        # mask = cv2.resize(mask, (256,256))
        # mask = np.array(mask)
        # mask = np.expand_dims(mask, axis=0)
        #
        # masks, scores, logits = predictor.predict(
        #     # point_coords=input_point,
        #     # point_labels=input_label,
        #     box=None,
        #     mask_input=mask,
        #     multimask_output=True,
        # )
        # bestmask = masks[0]
        # #visualize(np.array(I), bestmask.copy(), input_point, input_label, img_folder, img_name, 0)
        # visualize2(np.array(I), bestmask.copy(), img_folder, img_name)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bestmask.copy().astype(np.uint8),
        #                                                                         connectivity=8)
        # for i in range(num_labels):
        #     if stats[i][4] > I.height * I.width * 0.01:
        #         obj = np.where(labels == i)
        #         obj = np.array(obj)
        #         prediction_ = prediction[obj[0, :], obj[1, :]].flatten()
        #         unique_elements, counts = np.unique(prediction_, return_counts=True)
        #         index = np.argmax(counts)
        #         vote_sementics = unique_elements[index]
        #         prediction = np.where(labels == i, vote_sementics, prediction)
        #
        # trimmed_prediction = np.where(bestmask > 0, prediction, 0)
        # submitmask = np.zeros((I.height, I.width, 3))
        # submitmask[:, :, 0] = trimmed_prediction % 256
        # submitmask[:, :, 1] = trimmed_prediction // 256
        # suffix = img_name.replace('JPEG', 'png')
        # submitmask = Image.fromarray(submitmask.astype(np.uint8))
        # submitmask.save(os.path.join(dump_path, img_folder, suffix))


        masks = mask_generator.generate(np.array(I))
        for mask in masks:
            mask = mask['segmentation']
            obj = np.where(mask>0)
            obj = np.array(obj)
            prediction_ = prediction[obj[0, :], obj[1, :]].flatten()
            unique_elements, counts = np.unique(prediction_, return_counts=True)
            index = np.argmax(counts)
            vote_sementics = unique_elements[index]
            prediction = np.where(mask>0, vote_sementics, prediction)

        submitmask = np.zeros((I.height, I.width, 3))
        submitmask[:, :, 0] = prediction % 256
        submitmask[:, :, 1] = prediction // 256
        suffix = img_name.replace('JPEG', 'png')
        submitmask = Image.fromarray(submitmask.astype(np.uint8))
        submitmask.save(os.path.join(dump_path, img_folder, suffix))
        newmask = prediction > 0
        visualize2(img=np.array(I), mask_=newmask, img_folder=img_folder, img_name=suffix)





