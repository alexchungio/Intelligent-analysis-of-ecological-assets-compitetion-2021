#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : RSCDataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/5 下午8:30
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2
import logging
import numpy as np
from glob import glob
import os.path as osp
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data as data


from dataset.transforms import train_transform, val_transform



class RSCDataset(data.Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            # img_trans = img_nd.transpose(2, 0, 1)
            img_trans = img_nd.transpose((2, 0, 1))
        except:
            print(img_nd.shape)

        # normalize
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(os.path.join(self.imgs_dir,  f'{idx}.*'))
        mask_file = glob(os.path.join(self.masks_dir,  f'{idx}.*'))

        image = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        #if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        return {
            'image': image,
            'label': mask.long()
        }

def main():
    pass


if __name__ == "__main__":
    pass