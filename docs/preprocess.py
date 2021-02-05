#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : preprocess.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/4 下午9:45
# @ Software   : PyCharm
#-------------------------------------------------------


import numpy as np


def class_reweight(class_count):
    """
    re-sample | re-weight
    use training long tail dataset
    :param class_count:
    :return:
    """
    assert isinstance(class_count, dict)
    # sort
    order_count = sorted(class_count.items(), key=lambda x:x[0], reverse=False)

    # get label weight
    order_class_weight = [np.log(np.sum(order_count) / count) for _, count in order_count]

    # minimize weight equal to 1
    order_class_weight /= max(order_class_weight)

    # prevent too small
    order_class_weight = [weight + 0.1 for weight in order_class_weight ]

    return order_class_weight


def main():

    class_count = {2: 783364798,
                    1: 173067508,
                    6: 19350932,
                    4: 17751239,
                    9: 35714475,
                    3: 6086055,
                    7: 4600435,
                    5: 5458338,
                    8: 3419876,
                    10: 876456}

    class_weight = class_reweight(class_count)

    print(class_weight)


if __name__ == "__main__":
    main()

    # class_weight = [0.25430845,  0.14128766, 0.72660323, 0.57558217, 0.74196072, 0.56340895,
    #                 0.76608468, 0.80792181, 0.47695224, 1.]