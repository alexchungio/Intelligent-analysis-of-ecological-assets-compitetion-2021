#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/5 下午7:53
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import os.path as osp
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import  autocast, GradScaler  # need pytorch>1.6
from pytorch_toolbelt import losses as L

from utils.metric import IOUMetric
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss
from torchcontrib.optim import SWA
from utils import NoamLR

from model import EfficientUNetPlusPlus
from dataset import RSCDataset
from dataset import train_transform, val_transform
from utils import AverageMeter, second2time, initial_logger, smooth

Image.MAX_IMAGE_PIXELS = 1000000000000000
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# set random seed
use_cuda = True
seed = 2020
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


#  dataset
dataset_path = osp.join('/media/alex/80CA308ECA308288/alex_dataset/ecological-assets', 'split_dataset')
train_anns_dir = osp.join(dataset_path, 'ann_dir', 'train')
val_anns_dir = osp.join(dataset_path, 'ann_dir', 'val')
train_val_anns_dir = osp.join(dataset_path, 'ann_dir', 'train_val')

train_imgs_dir = osp.join(dataset_path, 'img_dir', 'train')
val_imgs_dir = osp.join(dataset_path, 'img_dir', 'val')
train_val_imgs_dir = osp.join(dataset_path, 'img_dir', 'train_val')


def eval(model, valid_loader, criterion, epoch, logger):
    """

    :config model:
    :config valid_loader:
    :config criterion:
    :config epoch:
    :config logger:
    :return:
    """
    model.eval()
    valid_epoch_loss = AverageMeter()
    valid_iter_loss = AverageMeter()
    iou = IOUMetric(10)
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(valid_loader)):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            pred = model(data)
            loss = criterion(pred.float(), target)
            pred = pred.cpu().data.numpy()
            pred = np.argmax(pred, axis=1)
            iou.add_batch(pred, target.cpu().data.numpy())
            #
            image_loss = loss.item()
            valid_epoch_loss.update(image_loss)
            valid_iter_loss.update(image_loss)
            # if batch_idx % iter_inter == 0:
            #     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
            #         epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, valid_iter_loss.avg))
        val_loss = valid_iter_loss.avg
        acc, acc_cls, iu, mean_iu, fwavacc = iou.evaluate()
        logger.info('[val] epoch:{} miou:{:.2f}'.format(epoch, mean_iu))

        return valid_epoch_loss, mean_iu


def train(config, model, train_data, valid_data, plot=False, device='cuda'):
    """

    :config config:
    :config model:
    :config train_data:
    :config valid_data:
    :config plot:
    :config device:
    :return:
    """
    # 初始化参数
    model_name = config['model_name']
    epochs = config['epochs']
    batch_size = config['batch_size']

    class_weights = config['class_weights']
    disp_inter = config['disp_inter']
    save_inter = config['save_inter']
    min_inter = config['min_inter']
    iter_inter = config['iter_inter']

    save_log_dir = config['save_log_dir']
    save_ckpt_dir = config['save_ckpt_dir']
    load_ckpt_dir = config['load_ckpt_dir']
    accumulation_steps = config['accumulation_steps']
    # automatic mixed precision
    scaler = GradScaler()
    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=1)
    #
    if config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    # SWA
    if config['swa']:
        swa_opt = SWA(optimizer, swa_start=config['swa_start'], swa_freq=config['swa_freq'], swa_lr=config['swa_lr'])

    # warm_up_with_multistep_lr = lambda \
    #     epoch: epoch / warmup_epochs if epoch <= warmup_epochs else gamma ** len(
    #     [m for m in milestones if m <= epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                     last_epoch=-1)
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'NoamLR':
        scheduler = NoamLR(optimizer, warmup_steps=config['warmup_steps'])
    else:
        raise NotImplementedError

    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    DiceLoss_fn = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    CrossEntropyLoss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    Lovasz_fn = LovaszLoss(mode='multiclass')
    criterion = L.JointLoss(first=DiceLoss_fn, second=CrossEntropyLoss_fn,
                            first_weight=0.5, second_weight=0.5).cuda()

    logger = initial_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) + '_' + model_name + '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = len(train_loader)
    valid_loader_size = len(valid_loader)
    best_iou = 0
    best_epoch = 0
    best_mode = copy.deepcopy(model)
    start_epoch = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info(
        'Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size,
                                                                                       valid_data_size))
    # execute train
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            with autocast():  # need pytorch>1.6
                pred = model(data)
                loss = criterion(pred, target)
                # 2.1 loss regularization
                regular_loss = loss / accumulation_steps
                # 2.2 back propagation
                scaler.scale(regular_loss).backward()
                # 2.3 update parameters of net
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step(epoch + batch_idx / train_loader_size)
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            # scheduler.step(epoch + batch_idx / train_loader_size)
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - start_time
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'], train_iter_loss.avg,
                    spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # validation
        valid_epoch_loss, mean_iou = eval(model, valid_loader, criterion, epoch, logger)
        # save loss and lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # save checkpoint
        if (epoch + 1) % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载

        # save best model
        if mean_iou > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mean_iou
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        # scheduler.step()

    # show loss curve
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    return best_mode, model


def main():
    model_name = 'efficientnet-b6'  # xception
    n_class = 10

    class_weights = torch.tensor(
        [0.25430845, 0.24128766, 0.72660323, 0.57558217, 0.74196072, 0.56340895, 0.76608468, 0.80792181, 0.47695224,
         1.],dtype=torch.float32)

    # ---------------------------------------config------------------------------------------
    config = {}
    config['model_name'] = model_name  # 模型名称
    config['epochs'] = 92  # 训练轮数
    config['batch_size'] = 8  # 批大小
    # config['lr'] = 3e-4  # AdamW
    config['optimizer'] = 'adamw'
    config['lr'] = 3e-4
    config['gamma'] = 0.1  # 学习率衰减系数
    config['momentum'] = 0.9  # 动量
    config['weight_decay'] = 5e-4  # 权重衰减

    config['swa'] = False

    config['scheduler'] = 'CosineAnnealingWarmRestarts'
    config['accumulation_steps'] = 2  # gradient accumulate
    config['warmup_steps'] = 2
    config['milestones'] = [40, 50]
    config['class_weights'] = class_weights


    # log config
    config['disp_inter'] = 1  # 显示间隔(epoch)
    config['save_inter'] = 4  # 保存间隔(epoch)
    config['iter_inter'] = 50  # 显示迭代间隔(batch)

    # save path
    config['save_log_dir'] = None  # log path
    config['min_inter'] = 10  # minimize epoch to save checkpoint
    config['save_ckpt_dir'] = None  # checkpoint path

    config['load_ckpt_dir'] = None  # 加载权重路径（继续训练）

    #-----------------------------model---------------------------------------
    model = EfficientUNetPlusPlus(model_name, n_class).cuda()
    model = torch.nn.DataParallel(model)

    # -----------------------------model save path-----------------------------
    save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
    save_log_dir = os.path.join('./outputs/', model_name)
    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(save_log_dir, exist_ok=True)

    config['save_log_dir'] = save_log_dir  # log path
    config['save_ckpt_dir'] = save_ckpt_dir  # checkpoint path

    #-------------------------------- load dataset--------------------------------------------
    train_data = RSCDataset(train_val_imgs_dir, train_val_anns_dir, transform=train_transform)
    valid_data = RSCDataset(val_imgs_dir, val_anns_dir, transform=val_transform)

    # training
    best_model, model = train(config, model, train_data, valid_data)

    print('Done')

if __name__ == "__main__":
    main()


