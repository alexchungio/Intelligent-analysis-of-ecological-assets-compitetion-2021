#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : transforms.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/4 下午2:18
# @ Software   : PyCharm
#-------------------------------------------------------

from torchvision import transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = ['get_transforms', 'get_albu_transform', 'train_transform', 'val_transform']


train_transform = A.Compose([
    # reszie
    A.Resize(256, 256),
    #
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5)
    ]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def get_albu_transform(image_size):
    """

    :param image_size:
    :return:
    """
    albu_transform = A.Compose([
                                A.Resize(image_size, image_size),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.RandomRotate90(),
                            ])

    return albu_transform


def get_train_transform(mean, std, size):
    """
    Data augmentation and normalization for training
    :param mean:
    :param std:
    :param size:
    :return:
    """
    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_transforms


def get_test_transform(mean, std, size):
    """
    Just normalization for validation
    :param mean:
    :param std:
    :param size:
    :return:
    """
    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean,
                    std=std),
    ])

    return test_transforms


def get_transforms(size, mode='test'):
    """

    :param size:
    :param mode:
    :return:
    """
    assert mode in ['train', 'val', 'test']
    mean, std = [0.625, 0.448, 0.688], [0.131, 0.177, 0.101]
    if mode in ['train']:
        transformations =get_train_transform(mean, std, size)
    else:
        transformations = get_test_transform(mean, std, size)

    return transformations


def main():
    pass

if __name__ == "__main__":
    main()






