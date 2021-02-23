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
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6
from pytorch_toolbelt import losses as L

from utils.metric import IOUMetric
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss

from utils import AverageMeter, second2time, initial_logger, smooth

from model import EfficientUNetPlusPlus
from dataset import RSCDataset
from dataset import train_transform, val_transform
from torch.cuda.amp import autocast
#
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS = 1000000000000000

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 准备数据集
dataset_path = osp.join('/media/alex/80CA308ECA308288/alex_dataset/ecological-assets', 'split_dataset')
train_labels_dir= osp.join(dataset_path, 'ann_dir', 'train')
val_labels_dir=osp.join(dataset_path, 'ann_dir', 'val')
ann_dir_train_val=osp.join(dataset_path, 'ann_dir', 'train_val')
train_imgs_dir=osp.join(dataset_path, 'img_dir', 'train')
val_imgs_dir=osp.join(dataset_path, 'img_dir', 'val')


def eval(model, valid_loader, criterion, epoch, loager):

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
        loager.info('[val] epoch:{} miou:{:.2f}'.format(epoch, mean_iu))

        return valid_epoch_loss, mean_iu


def train(param, model, train_data, valid_data, class_weights=None, plot=False, device='cuda'):
    """

    :param param:
    :param model:
    :param train_data:
    :param valid_data:
    :param plot:
    :param device:
    :return:
    """
    # 初始化参数
    model_name = param['model_name']
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']

    warmup_epochs = param['warmup_epochs']
    milestones = param['milestones']

    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    min_inter = param['min_inter']
    iter_inter = param['iter_inter']

    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    accumulation_steps = param['accumulation_steps']
    # automatic mixed precision
    scaler = GradScaler()
    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                     last_epoch=-1)
    # warm_up_with_multistep_lr = lambda \
    #     epoch: epoch / warmup_epochs if epoch <= warmup_epochs else gamma ** len(
    #     [m for m in milestones if m <= epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    DiceLoss_fn = DiceLoss(mode='multiclass')
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    CrossEntropyLoss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    Lovasz_fn = LovaszLoss(mode='multiclass')
    criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                            first_weight=0.5, second_weight=0.5).cuda()
    logger = initial_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d %H:%M:%S", time.localtime()) + '_' + model_name + '.log'))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_epoch = 0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info(
        'Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size,
                                                                                       valid_data_size))
    # execute train
    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
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
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # validation
        valid_epoch_loss, mean_iou = eval(model, valid_loader, criterion, epoch, logger)
        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # save checkpoint
        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载

        # 保存最优模型
        if mean_iou > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mean_iou
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        # scheduler.step()
        # 显示loss
    # 训练loss曲线
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

    class_weights = torch.tensor(
        [0.25430845, 0.24128766, 0.72660323, 0.57558217, 0.74196072, 0.56340895, 0.76608468, 0.80792181, 0.47695224,
         1.],dtype=torch.half)
    model_name = 'efficientnet-b6'  # xception
    n_class = 10
    model = EfficientUNetPlusPlus(model_name, n_class).cuda()
    model = torch.nn.DataParallel(model)
    # checkpoints=torch.load('outputs/efficientnet-b6-3729/ckpt/checkpoint-epoch20.pth')
    # model.load_state_dict(checkpoints['state_dict'])
    # 模型保存路径
    save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
    save_log_dir = os.path.join('./outputs/', model_name)

    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(save_ckpt_dir, exist_ok=True)
    train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
    valid_data = RSCDataset(val_imgs_dir, val_labels_dir, transform=val_transform)

    # 参数设置
    param = {}

    param['epochs'] = 50  # 训练轮数
    param['batch_size'] = 6  # 批大小
    param['lr'] = 1e-2  # 学习率
    param['gamma'] = 0.2  # 学习率衰减系数
    param['step_size'] = 5  # 学习率衰减间隔
    param['momentum'] = 0.9  # 动量
    param['weight_decay'] = 5e-4  # 权重衰减
    param['disp_inter'] = 1  # 显示间隔(epoch)
    param['save_inter'] = 4  # 保存间隔(epoch)
    param['iter_inter'] = 50  # 显示迭代间隔(batch)
    param['min_inter'] = 10
    param['warmup_epochs'] = 2
    param['milestones'] = [40, 50]

    ## gradient accumalate
    param['accumulation_steps'] = 3
    param['model_name'] = model_name  # 模型名称
    param['save_log_dir'] = save_log_dir  # 日志保存路径
    param['save_ckpt_dir'] = save_ckpt_dir  # 权重保存路径

    # 加载权重路径（继续训练）
    param['load_ckpt_dir'] = None

    # 训练
    best_model, model = train(param, model, train_data, valid_data)

    print('Done')

if __name__ == "__main__":
    main()

