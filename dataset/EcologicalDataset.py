#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : EcologicalDataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/3 下午4:33
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import numpy as np


import os
import cv2
import logging

from glob import glob
import os.path as osp
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data as data

from dataset.transforms import get_transforms, get_albu_transform
from misc.visual import visual_mask, tensor_to_numpy



from dataset.transforms import train_transform, val_transform

class EcologicalDataset(data.Dataset):

    def __init__(self, image_path, transforms=None, album_aug=None, mode='test'):
        super(EcologicalDataset, self).__init__()

        self.image_path = image_path
        self.transforms = transforms
        self.album_aug = album_aug
        self.mode = mode

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):

        try:
            image = self.pil_loader(self.image_path[index])

            if self.mode == 'train':
                mask = Image.open(self.image_path[index].replace('.tif', '.png'))
                image = np.array(image)
                mask = np.array(mask) - 1 # label: [1, ..., 10] => [0, ..., 9]
                if self.album_aug is not None:
                    augments = self.album_aug(image=image, mask=mask)

                    image = self.transforms(augments['image'])
                    mask = augments['mask'].astype(np.int64)
                else:
                    image = self.transforms(image)
                    mask = mask.astype(np.int64)
            else:
                image = self.transforms(image)
                mask = None

            return image, mask
        except:
            print('Corrupted due to cannot read {}'.format(self.image_path[index]))

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def main():
    plt.ion()  # interactive mode

    # data_provider = DataProvider()
    #
    # train_loader = data_provider(args.train_data, args.batch_size, backbone=None, phase='train', num_worker=4)
    # eval_loader = data_provider(args.val_data, args.batch_size, backbone=None, phase='val', num_worker=4)
    #
    # images, labels = next(iter(eval_loader))
    #
    # plot_image_class(images[:36], labels[:36], index_class=index_class)

    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'

    train_image_path = glob.glob(osp.join(dataset_path, 'suichang_round1_train_210120', '*.tif'))
    # train_mask_path = glob.glob(osp.join(dataset_path, 'suichang_round1_train_210120', '*.png'))

    test_image_path = glob.glob(osp.join(dataset_path, 'suichang_round1_test_partA_210120', '*.tif'))

    train_dataset = EcologicalDataset(image_path=train_image_path,
                                      transforms=get_transforms(size=256, mode='train'),
                                      album_aug=get_albu_transform(image_size=256),
                                      mode='train')

    test_dataset = EcologicalDataset(image_path=train_image_path,
                                     transforms=get_transforms(size=256, mode='test'),
                                     mode='test')

    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

    # for image, mask in train_dataset:
    image, mask = train_dataset[150]

    image = tensor_to_numpy(img_tensor=image, mean_std=True)
    mask = visual_mask(mask_img=mask + 1)
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(mask, cmap='gray')
    plt.subplot(122)
    plt.imshow(image)
    plt.show()

    print('Done')



if __name__ == "__main__":
    main()


