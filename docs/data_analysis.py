#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : data_analysis.py
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

import numpy as np
import PIL.Image as Image

import matplotlib.pyplot as plt
from collections import Counter

from tqdm import tqdm
import pandas as pd


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


def check_visual_image(img_path):
    """

    :param image:
    :return:
    """
    # fig, axs = plt.subplots(1, 2)
    raw_img = Image.open(img_path)
    raw_img_array = np.array(raw_img)
    print('raw format {} shape {}'.format(raw_img.mode, raw_img_array.shape))
    raw_img.show()

    rgb_img = raw_img.convert(mode='RGB')
    rgb_img_array = np.array(rgb_img)
    print('rgb format {} shape {}'.format(rgb_img.mode, rgb_img_array.shape))
    rgb_img.show()

    assert (raw_img_array[:, :, :3] == rgb_img_array).all()
    # axs[0].imshow(raw_img)
    # axs[0].set_title('rgb-nir image')
    # axs[1].imshow(rgb_img)
    # axs[1].set_title('rgb image')
    # plt.show()
    return rgb_img

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


def visual_mask(mask_path, palette):
    """

    :param mask_path:
    :param palette:
    :return:
    """
    mask_img = Image.open(mask_path)
    mask_img.putpalette(palette)

    return mask_img


def visual_dataset(image_dataset, mask_dataset, palette, num_row=2, num_col=6):
    """

    :param image_dataset:
    :param mask_dataset:
    :param platte:
    :param num_row:
    :param num_col:
    :return:
    """
    img_data = image_dataset[: num_col]
    mask_data = mask_dataset[: num_col]
    fig, axs = plt.subplots(nrows=num_row, ncols=num_col, figsize=(16, 6),
                            subplot_kw={'xticks': [], 'yticks': []})

    for index, (img_path, mask_path) in enumerate(zip(img_data, mask_data)):

        rgb_img = convert_img_mode(img_path)
        mask_img = visual_mask(mask_path, palette)

        img_name = os.path.basename(img_path).split('.')[0]
        axs.flat[index].imshow(rgb_img)
        axs.flat[index].set_title(img_name, size=15)
        axs.flat[index + num_col].imshow(mask_img)

    plt.show()



def get_label_number(mask_dataset):

    label_count = Counter()

    for mask_path in tqdm(mask_dataset):
        mask_img = Image.open(mask_path)
        label_count.update(np.array(mask_img).flatten())
        # for label, num in count.items():
        #
        #     if label in label_count.keys():
        #         label_count[label] += num
        #     else:
        #         label_count[label] = num

    return label_count


def show_count(label_count, label_cat):
    """

    :param label_count:
    :param label_cat:
    :return:
    """
    cat_count = {label_cat[label]: value for label, value in label_count.items()}
    plt.style.use({'figure.figsize': (16, 16)})
    indices = cat_count.keys()
    values = cat_count.values()
    count_df = pd.DataFrame(list(values), index=indices)
    print(count_df)
    count_df.iloc[:, 0].plot.pie()
    plt.legend()
    plt.axis('off')
    plt.show()


def main():
    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'
    train_dataset_path = os.path.join(dataset_path, 'suichang_round1_train_210120')
    test_dataset_path = os.path.join(dataset_path, 'suichang_round1_test_partA_210120')
    palette = get_palette()

    train_image = glob.glob(osp.join(train_dataset_path, '*.tif'))
    train_mask = glob.glob(osp.join(train_dataset_path, '*.png'))

    test_image = glob.glob(osp.join(test_dataset_path, '*.tif'))

    print('number of train sample: {}'.format(len(train_image)))
    print('number of test sample: {}'.format(len(test_image)))

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
    # for img_path in train_image:
    #     check_visual_image(img_path)
    #     break

    # visual palette
    show_palette(palette, cat_label)


    for mask_path in train_mask:
        visual_mask(mask_path, palette)

    visual_dataset(train_image, train_mask, palette)

    label_count = get_label_number(train_mask)
    print(label_count)


    print('Done')
if __name__ == "__main__":
    main()

    # c = Counter()
    #
    # a = np.random.randint(1, 10, 100)
    # b = np.random.randint(1, 10, 100)
    #
    # c.update(a)
    # print(c)
    # c.update(b)
    #
    # print(c)









