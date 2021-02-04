#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : visual.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/4 下午5:04
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import os.path as osp
import glob

import numpy as np
import PIL.Image as Image

import matplotlib.pyplot as plt


def get_palette():
    """

    :return:
    """
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))

    color_array = np.array([[0, 0, 0],  # other
                            [177, 191, 122],  # farm_land
                            [0, 128, 0],  # forest
                            [128, 168, 93],  # grass
                            [62, 51, 0],  # road
                            [128, 128, 0],  # urban_area
                            [128, 128, 128],  # countryside
                            [192, 128, 0],  # industrial_land
                            [0, 128, 128],  # construction
                            [132, 200, 173],  # water
                            [128, 64, 0]],  # bareland
                           dtype='uint8')

    palette[:3*11]=color_array.flatten()

    return palette


def show_palette(palette, cat_label:dict):
    """

    :param palette:
    :param cat_label:
    :return:
    """
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 4),
                            subplot_kw={'xticks': [], 'yticks': []})

    for cat, label in cat_label.items():

        cat_palette = np.ones((4, 6, 3), dtype=np.int32) * np.array([palette[label*3: (label+1)*3]])

        axs.flat[label-1].imshow(cat_palette)
        axs.flat[label-1].set_title(cat, size=14)

    plt.show()


def convert_img_mode(img_path):
    """

    :param img_path:
    :return:
    """
    raw_img = Image.open(img_path)
    rgb_img = raw_img.convert(mode='RGB')

    return rgb_img


def visual_mask(mask_img=None, mask_path=None):
    """

    :param mask_path:
    :param palette:
    :return:
    """
    palette = get_palette()
    if mask_img is None:
        mask_img = Image.open(mask_path)
    else:
        mask_img = Image.fromarray(np.array(mask_img, dtype=np.uint8))
    mask_img.putpalette(palette)

    return mask_img


def tensor_to_numpy(img_tensor, mean_std=False):

    img_array = img_tensor.numpy().transpose((1, 2, 0))
    mean, std = [0.625, 0.448, 0.688], [0.131, 0.177, 0.101]
    if mean_std:
        img_array = img_array * np.array(std) + np.array(mean)

    img_array = np.clip(img_array, 0, 1)
    image = np.array(img_array * 255, dtype=np.uint8)

    return image


