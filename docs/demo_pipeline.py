#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : demo_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/3 上午10:40
# @ Software   : PyCharm
#-------------------------------------------------------

import os.path as osp
import numpy as np
import pathlib, sys, os, random, time
import numba, cv2, gc
from tqdm import tqdm_notebook
import glob

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
torch.backends.cudnn.enabled = True

import torchvision
from torchvision import transforms as T
from PIL import Image

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


### global variable
NUM_FOLD = 7
EPOCHES = 30
BATCH_SIZE = 8
IMAGE_SIZE = 256


trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
])


class_weights = torch.tensor([0.25430845,  0.24128766, 0.72660323, 0.57558217, 0.74196072, 0.56340895, 0.76608468, 0.80792181, 0.47695224, 1.],
                             dtype=torch.float32)


class TianChiDataset(D.Dataset):
    def __init__(self, paths, transform, test_mode=False):
        self.paths = paths
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize([0.625, 0.448, 0.688],
                        [0.131, 0.177, 0.101]),
        ])

    # get dataset operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.test_mode:
            mask = cv2.imread(self.paths[index].replace('.tif', '.png')) - 1
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments['image']), augments['mask'][:, :, 0].astype(np.int64)
        else:
            return self.as_tensor(img), ''

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def validation(model, loader, loss_fn):

    model.eval()
    with torch.no_grad():
        val_iou = []
        for image, target in loader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)
            output = output.argmax(1)
            iou = get_iou(output, target)
            val_iou.append(iou)

    return val_iou


def get_iou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)

        uion = p.sum() + t.sum()
        overlap = (p * t).sum()

        # print(idx, uion, overlap)
        iou = 2 * overlap / (uion + 0.001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)


def train(dataset):
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)
    class_name = ['farm_land', 'forest', 'grass', 'road', 'urban_area',
                  'countryside', 'industrial_land', 'construction',
                  'water', 'bareland']
    print('  '.join(class_name))

    valid_idx, train_idx = [], []

    for fold_idx in range(NUM_FOLD):
        for i in range(len(dataset)):
            if i % NUM_FOLD == fold_idx:
                valid_idx.append(i)
            else:
                train_idx.append(i)

        train_ds = D.Subset(dataset, train_idx)
        valid_ds = D.Subset(dataset, valid_idx)

        # define training and validation dataset loaders
        loader = D.DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        vloader = D.DataLoader(
            valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
            in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=10,  # model output channels (number of classes in your dataset)
        )
        model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=0.00025, weight_decay=1e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(DEVICE)

        best_iou = 0
        for epoch in range(1, EPOCHES + 1):
            losses = []
            start_time = time.time()
            model.train()
            model.to(DEVICE)
            for image, target in tqdm_notebook(loader):
                image, target = image.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(image)

                # break
                loss = loss_fn(output, target)
                loss.backward()

                optimizer.step()
                losses.append(loss.item())

            viou = validation(model, vloader, loss_fn)

            lr_schedule.step()

            print('\t'.join(np.stack(viou).mean(0).round(3).astype(str)))

            print(raw_line.format(epoch, np.array(losses).mean(), np.mean(viou),
                                  (time.time() - start_time) / 60 ** 1))
            if best_iou < np.stack(viou).mean(0).mean():
                best_iou = np.stack(viou).mean(0).mean()
                torch.save(model.state_dict(), 'model_{0}.pth'.format(fold_idx))

        torch.save(model.state_dict(), '20210204_latest.pth'.format(fold_idx))

        break


def test(dataset):
    """

    :return:
    """
    trfm = T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.625, 0.448, 0.688],
                    [0.131, 0.177, 0.101]),
    ])

    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
        in_channels=3,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=10,  # model output channels (number of classes in your dataset)
    )
    model.eval()
    model.to(DEVICE)
    model.load_state_dict(torch.load("./20210204_latest.pth"))


    for idx, name in enumerate(tqdm(dataset)):
        img = cv2.imread(name)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = trfm(image)
        image_batch = torch.unsqueeze(image, dim=0)
        with torch.no_grad():
            image = image_batch.to(DEVICE)
            score1 = model(image).cpu().numpy()

            b = torch.flip(image, [0, 3])
            score2 = model(torch.flip(image, [0, 3]))
            #         score2 = score2.cpu().numpy()
            score2 = torch.flip(score2, [3, 0]).cpu().numpy()

            score3 = model(torch.flip(image, [0, 2]))
            #         score3 = score3.cpu().numpy()
            score3 = torch.flip(score3, [2, 0]).cpu().numpy()

            score = (score1 + score2 + score3) / 3.0

            score_sigmoid = score[0].argmax(0) + 1

            print(score_sigmoid.min(), score_sigmoid.max())

            os.makedirs('results', exist_ok=True)
            cv2.imwrite('results/' + name.split('/')[-1].replace('.tif', '.png'), score_sigmoid)
        # break

def main():

    dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'

    train_dataset = TianChiDataset(
        glob.glob(osp.join(dataset_path, 'suichang_round1_train_210120', '*.tif')),
        trfm, False
    )

    test_dataset = glob.glob(osp.join(dataset_path, 'suichang_round1_test_partA_210120', '*.tif')[:])
    # visualize mask
    # image, mask = dataset[150]
    # plt.figure(figsize=(16, 8))
    # plt.subplot(121)
    # plt.imshow(mask, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(image)
    # plt.show()
    # train(train_dataset)

    test(test_dataset)

if __name__ == "__main__":
    main()