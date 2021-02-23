#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : inference.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/4 下午7:47
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import torch
import cv2
import time
from PIL import Image

from tqdm import tqdm
import glob
import os
from scipy.io import loadmat


import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import colorEncode
import torch.nn as nn
from torch.cuda.amp import autocast

from model import EfficientUNetPlusPlus

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Image.MAX_IMAGE_PIXELS = 1000000000000000

def visualize_result(img_dir, pred):
    #
    img=cv2.imread(img_dir)
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            1: "耕地",
            2: "林地",
            3: "草地",
            4: "道路",
            5: "城镇建设用地",
            6: "农村建设用地",
            7: "工业用地",
            8: "构筑物",
            9: "水域",
            10: "裸地"
        }
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    #
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    #print(pred_color.shape)
    #pred_color=cv2.resize(pred_color,(256,256))
    im_vis = np.concatenate((img, pred_color), axis=1)

    #
    #img_name=image_demo_dir.split('/')[-1]
    save_dir,name=os.path.split(img_dir)
    Image.fromarray(im_vis).save('demo/256x256_deeplab_44.png')

def get_infer_transform():
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform


def inference(model, img_dir):
    transform=get_infer_transform()
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
    #print(img.shape)
    with torch.no_grad():
        img=img.cuda()
        output = model(img)
    #
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred+1



def main():

    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'
    test_imgs = os.path.join(dataset_path, 'test_jpg')

    model_name = 'efficientnet-b6'  # efficientnet-b4
    n_class = 10
    model = EfficientUNetPlusPlus(model_name, n_class).cuda()
    model = torch.nn.DataParallel(model)

    # load checkpoint
    checkpoints = torch.load('outputs/efficientnet-b6/ckpt/checkpoint-epoch44.pth')
    model.load_state_dict(checkpoints['state_dict'])

    model.eval()
    use_demo = False
    assert_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if use_demo:
        img_dir = 'demo/000097.jpg'
        pred = inference(img_dir)
        infer_start_time = time.time()
        visualize_result(img_dir, pred)
        #
    else:
        out_dir = 'result/results/'
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        test_paths = glob.glob(os.path.join(test_imgs, '*.jpg'))
        for per_path in tqdm(test_paths):
            result = inference(model, per_path)
            img = Image.fromarray(np.uint8(result))
            img = img.convert('L')
            # print(out_path)
            out_path = os.path.join(out_dir, per_path.split('/')[-1][:-4] + '.png')
            img.save(out_path)

if __name__=="__main__":
    main()



