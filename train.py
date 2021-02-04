#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/3 下午4:25
# @ Software   : PyCharm
#-------------------------------------------------------
import random
import os.path as osp
import numpy as np
import pathlib, sys, os, random, time
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import shutil

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import optimizer
from torch.optim import lr_scheduler
# import torch.nn.parallel as parallel
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from data.dataset import EcologicalDataset
from data.transforms import get_albu_transform, get_transforms
from misc.metric import get_iou, AverageMeter
torch.backends.cudnn.enabled = True

import segmentation_models_pytorch as smp

# set random sees

random.seed(2021)
torch.manual_seed(2021)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir='./output/summary')

show_iter = 200

def train(model, data_loader, optimizer, criterion, device):
    """

    :param model:
    :param criterion:
    :param device:
    :return:
    """
    model.train()

    losses = AverageMeter()

    pbar = tqdm(data_loader)
    for image, target in pbar:
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), image.size(0))

        pbar.set_description('\ttrain => loss {:.4f}'.format(losses.avg), refresh=True)

    return losses.avg


def eval(model, data_loader, criterion, device):
    """

    :param model:
    :param data_loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()

    losses = AverageMeter()
    val_iou = []
    pbar = tqdm(data_loader)
    with torch.no_grad():
        for image, target in pbar:
            image, target = image.to(device), target.to(device)
            output = model(image)

            # calculate loss
            loss = criterion(output, target)
            # calculate iou
            pred = output.argmax(1)
            iou = get_iou(pred, target)
            val_iou.append(iou)

            losses.update(loss.item(), image.size(0))

            pbar.set_description('eval loss {0}'.format(loss.item()), refresh=True)

    pbar.write('\teval => loss {:.4f}'.format(losses.avg))

    return losses.avg, val_iou



def main():
    ### global variable
    NUM_FOLD = 7
    EPOCHES = 24
    BATCH_SIZE = 8
    IMAGE_SIZE = 256
    loss_weights = None

    checkpoint_path = osp.join('./outputs', 'checkpoints')

    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'

    epoch_info = 'Epoch {:6d}: ' + 'val_iou \u2502{:7.3f}|' * 2 + 'time \u2502{:6.2f}'

    class_name = ['farm_land', 'forest', 'grass', 'road', 'urban_area','countryside', 'industrial_land', 'construction',
                  'water', 'bareland']
    print('  '.join(class_name))

    train_image_path = glob.glob(osp.join(dataset_path, 'suichang_round1_train_210120', '*.tif'))
    # train_mask_path = glob.glob(osp.join(dataset_path, 'suichang_round1_train_210120', '*.png'))

    dataset = EcologicalDataset(image_path=train_image_path,
                                      transforms=get_transforms(size=IMAGE_SIZE, mode='train'),
                                      album_aug=get_albu_transform(image_size=IMAGE_SIZE),
                                      mode='train')

    # train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    train_idx, val_idx = [], []
    for fold_idx in range(NUM_FOLD):
        for i in range(len(dataset)):
            if i % NUM_FOLD == fold_idx:
                val_idx.append(i)
            else:
                train_idx.append(i)

        # split dataset
        train_dataset = data.Subset(dataset, train_idx)
        val_dataset = data.Subset(dataset, val_idx)

        # define training and validation data loaders
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[16, 20], gamma=0.1)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)


        best_iou = 0
        for epoch in range(1, EPOCHES + 1):

            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_iou = eval(model, val_loader, criterion, device)

            print('\t'.join(np.stack(val_iou).mean(0).round(3).astype(str)))
            print(epoch_info.format(epoch, np.mean(val_iou), (time.time() - start_time) / 60 ** 1))


            # calculate miou
            miou = np.stack(val_iou).mean(0).mean()

            # save logs
            writer.add_scalars(main_tag='epoch/loss', tag_scalar_dict={'train': train_loss, 'val': val_loss},
                               global_step=epoch)
            writer.add_scalar(tag='epoch/miou', scalar_value=miou, global_step=epoch)

            # add learning_rate to logs
            writer.add_scalar(tag='lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
            scheduler.step()

            # save checkpoint
            state = {
                'epoch': epoch,
                'iou': best_iou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if best_iou < np.stack(val_iou).mean(0).mean():
                best_iou = np.stack(val_iou).mean(0).mean()

                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(state, osp.join(checkpoint_path, 'model_{0}.pth'.format(fold_idx)))

        torch.save(state, osp.join(checkpoint_path, '20210204_latest.pth'))

        break


if __name__ == "__main__":
    main()