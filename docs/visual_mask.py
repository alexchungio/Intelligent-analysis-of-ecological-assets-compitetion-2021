#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : visual_mask.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/3 下午5:21
# @ Software   : PyCharm
#-------------------------------------------------------

# reference https://raw.githubusercontent.com/xmojiao/deeplab_v2/master/voc2012/create_labels_21.py

import os
import os.path as osp
import glob
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image
import cv2 as cv
import matplotlib.pyplot as plt
from IPython import display

# os.chdir('/home/dl/deeplab_v2/voc2012/features/deeplab_largeFOV/val/fc8/')
# path=os.getcwd()
# files=os.listdir(path)
# labels_path=os.path.join(path,'labels')


def get_palette():
    """

    :return:
    """
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))

    color_array = np.array([[0, 0, 0],
                            [128, 128, 0],
                            [0, 128, 0],
                            [128, 168, 93],
                            [62, 51, 0],
                            [128, 128, 0],
                            [128, 128, 128],
                            [192, 128, 0],
                            [0, 128, 128],
                            [132, 200, 173],
                            [128, 64, 0]], dtype='uint8')

    palette[:3*11]=color_array.flatten()

    return palette


# for afile in files:
#     file_path=os.path.join(path,afile)
#     if os.path.isfile(file_path):
#         if os.path.getsize(file_path)==0:
#             continue
#         mat_idx=afile[:afile.find('.mat')]
#         mat_file=sio(file_path)
#         mat_file=mat_file['data']
#         labels=np.argmax(mat_file,axis=2).astype(np.uint8)
#         label_img=Image.fromarray(labels.reshape(labels.shape[0],labels.shape[1]))
#         label_img.putpalette(palette)
#         label_img=label_img.transpose(Image.FLIP_LEFT_RIGHT)
#         label_img = label_img.rotate(90)
#         dst_path=os.path.join(labels_path,mat_idx+'.png')
#         label_img.save(dst_path)


def visual_image(img_path):
    """

    :param image:
    :return:
    """


    # fig, axs = plt.subplots(1, 2)
    raw_img = Image.open(img_path)
    raw_img_array = np.array(raw_img)
    print('raw format {} shape {}'.format(raw_img.mode, raw_img_array.shape))
    raw_img.show(title='rgb-nir image')

    rgb_img = raw_img.convert(mode='RGB')
    rgb_img_array = np.array(rgb_img)
    print('rgb format {} shape {}'.format(rgb_img.mode, rgb_img_array.shape))
    rgb_img.show(title='rgb image')

    assert (raw_img_array[:, :, :3] == rgb_img_array).all()
    # axs[0].imshow(raw_img)
    # axs[0].set_title('rgb-nir image')
    # axs[1].imshow(rgb_img)
    # axs[1].set_title('rgb image')
    # plt.show()

def show_palette(palette, cat_label:dict):
    """

    :param palette:
    :param cat_label:
    :return:
    """
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(16, 10),
                            subplot_kw={'xticks': [], 'yticks': []})


    for cat, label in cat_label.items():

        cat_palette = np.ones((6, 8, 3), dtype=np.int32) * np.array([palette[label*3: (label+1)*3]])

        axs.flat[label-1].imshow(cat_palette)
        axs.flat[label-1].set_title(cat, size=20)

    plt.show()




def visual_mask(mask_path, palette):

    mask_img = Image.open(mask_path)
    mask_img.putpalette(palette)

    mask_img.show()

def main():
    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'
    train_dataset_path = os.path.join(dataset_path, 'suichang_round1_train_210120')
    test_dataset_path = os.path.join(dataset_path, 'suichang_round1_test_partA_210120')
    palette = get_palette()

    train_image = glob.glob(osp.join(train_dataset_path, '*.tif'))
    train_mask = glob.glob(osp.join(train_dataset_path, '*.png'))

    cat_label = {'farm_land': 1,
                 'forest': 2,
                 'grass': 3,
                 'road': 4,
                 'urban_area': 5,
                 'countryside': 6,
                 'industrial_land': 7,
                 'construction': 8,
                 'water': 9,
                 'bareland': 10}

    assert len(train_image) == len(train_mask)

    # visual image
    for img_path in train_image:
        visual_image(img_path)
        break

    # visual palette
    show_palette(palette, cat_label)


    for mask_path in train_mask:
        visual_mask(mask_path, palette)

        break

    print('Done')
if __name__ == "__main__":
    main()




