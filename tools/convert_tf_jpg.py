#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : convert_tf_jpg.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/7 下午3:13
# @ Software   : PyCharm
#-------------------------------------------------------


import cv2.cv2 as cv
import cv2
import os
import shutil
from tqdm import tqdm


dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'

train_image = os.path.join(dataset_path, 'suichang_round1_train_210120')

test_image= os.path.join(dataset_path, 'suichang_round1_test_partA_210120')


save_imgs = os.path.join(dataset_path, 'images')
save_masks = os.path.join(dataset_path, 'masks')
save_test_imgs = os.path.join(dataset_path, 'test_jpg')


def main():

    # split image and mask
    os.makedirs(save_imgs, exist_ok=True)
    os.makedirs(save_masks, exist_ok=True)
    tif_list = [x for x in os.listdir(train_image)]  # 获取目录中所有tif格式图像列表
    for num, name in tqdm(enumerate(tif_list)):  # 遍历列表
        if name.endswith(".tif"):
            img = cv.imread(os.path.join(train_image, name), -1)  # 读取列表中的tif图像
            cv.imwrite(os.path.join(save_imgs, name.split('.')[0] + ".jpg"), img)  # tif 格式转 jpg
        else:
            img = cv.imread(os.path.join(train_image, name), cv2.IMREAD_GRAYSCALE)
            img = img - 1
            cv2.imwrite(os.path.join(save_masks, name), img)
            # shutil.copy(os.path.join(images_dir, name),os.path.join(save_masks,name))


    os.makedirs(save_test_imgs, exist_ok=True)
    for name in os.listdir(test_image):
        img = cv.imread(os.path.join(test_image, name), -1)  #
        cv.imwrite(os.path.join(save_test_imgs, name.split('.')[0] + ".jpg"), img)  # tif 格式转 jpg

if __name__ == "__main__":
    main()