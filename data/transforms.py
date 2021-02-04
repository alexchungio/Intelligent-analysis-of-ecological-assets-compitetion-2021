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

import torchvision




#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : transforms.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/11/4 上午11:56
# @ Software   : PyCharm
#-------------------------------------------------------

import random
from PIL import Image, ImageFilter
from torchvision import transforms
import albumentations as albu
from torchvision import transforms as T

__all__ = ['get_transforms', 'get_albu_transform']



def get_albu_transform(image_size):
    """

    :param image_size:
    :return:
    """
    albu_transform = albu.Compose([
                                albu.Resize(image_size, image_size),
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.RandomRotate90(),
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






