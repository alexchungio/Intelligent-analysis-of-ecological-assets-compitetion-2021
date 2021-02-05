#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : efficient_unet.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/5 下午8:15
# @ Software   : PyCharm
#-------------------------------------------------------

import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast


class EfficientUNet(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    # automatic mixed precision
    @autocast()
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x
#