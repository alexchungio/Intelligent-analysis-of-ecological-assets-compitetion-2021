#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/2/5 下午7:51
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import argparse


# Root Path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT_PATH = sys.path(__file__)
print(ROOT_PATH)
print (20*"++--")

# Parse arguments
parser = argparse.ArgumentParser(description= 'PyTorch ImageNet Training')

# Dataset
parser.add_argument('--dataset', default='/media/alex/80CA308ECA308288/alex_dataset/planet', type=str)
parser.add_argument('--train-label', default=os.path.join('/media/alex/80CA308ECA308288/alex_dataset/planet', 'train_classes.csv'), type=str)
parser.add_argument('--train-dataset', default=os.path.join('/media/alex/80CA308ECA308288/alex_dataset/planet', 'train-jpg'), type=str)
parser.add_argument('--test-dataset', default=os.path.join('/media/alex/80CA308ECA308288/alex_dataset/planet', 'test-jpg'), type=str)

parser.add_argument('--tags-count', default=os.path.join(ROOT_PATH, 'dataset', 'counts', 'tags_count.csv'), type=str,
                    help='path to count the number of class') #new_shu_label
parser.add_argument('--corr', default=os.path.join(ROOT_PATH, 'dataset', 'counts', 'corr.csv'), type=str,
                    help='path to the correlation matrix of class')
parser.add_argument('--labels', default=os.path.join(ROOT_PATH, 'dataset', 'counts', 'labels.csv'), type=str,
                    help='path to train dataset and labels')
parser.add_argument('--classes', default=os.path.join(ROOT_PATH, 'dataset', 'classes', 'planet.names'), type=str,
                    help='path to class name')
parser.add_argument('--class-weights', default=os.path.join(ROOT_PATH, 'dataset', 'classes', 'class_weights'), type=str,
                    help='path to save class weight')
# parser.add_argument('-train', '--train_data', default=os.path.join(ROOT_PATH, 'dataset', 'labels', 'train.txt'), type=str) #new_shu_label
# parser.add_argument('-val', '--val_data', default=os.path.join(ROOT_PATH, 'dataset', 'labels', 'val.txt'), type=str)


parser.add_argument('--fold', type=int, default=0, metavar='N', help='Train/valid fold #. (default: 0')
parser.add_argument('--multi-label', action='store_true', default=True, help='Multi-label target')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default=os.path.join(ROOT_PATH, 'outputs', 'weights'), type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--best-checkpoint', default=os.path.join(ROOT_PATH, 'outputs', 'weights', 'best_ckpt.pth.tar'),
                    type=str, metavar='PATH',  help='path to save best checkpoint (default: checkpoint)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='resume the train log info')
# Logs
parser.add_argument('-s', '--summary-dir', default=os.path.join(ROOT_PATH, 'outputs', 'summary'), type=str, metavar='PATH',
                    help='path to save logs (default: logs)')
parser.add_argument('--summary-iter', default=100, type=int, help='number of iterator to save logs (default: 1)')

# inference
parser.add_argument('--inference-path', default=os.path.join(ROOT_PATH, 'outputs', 'inference'), type=str, metavar='PATH',
                    help='path to save inference result (default: logs)')

# Train
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful at restart)')

parser.add_argument('--num_classes', default=17, type=int, metavar='N',
                    help='number of classification of image')
parser.add_argument('--image-type', default='.jpg', type=str, metavar='N',
                    help='train and val image type')
parser.add_argument('--image-size', default=256, type=int, metavar='N',
                    help='train and val image size')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batch size (default: 256)')
parser.add_argument('--num-works', default=4, type=int, metavar='N',
                    help='number subprocesses to use for dataset loading (default: 4)')

# LR
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate，1e-2， 1e-4, 0.001')

parser.add_argument('--lr-times', '--lr_accelerate_times', default=5, type=int,
                    metavar='LR', help='custom layer lr accelerate times')

parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--decay-epoch', type=int, default=15, metavar='N', help='epoch interval to decay LR')
parser.add_argument('--gamma', '--factor', type=float, default=0.1, help='LR is multiplied by gamma/factor on schedule.')


# Loss
parser.add_argument('--loss', default='mlsm', type=str, metavar='LOSS',
                    help='Loss function (default: "nll"')

parser.add_argument('--reweight', action='store_true', default=False,
                    help='Use class weights for specified labels as loss penalty')


# Optimizer
parser.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam', 'AdaBound', 'radam'], metavar='N',
                         help='optimizer (default=sgd)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--use-nesterov', default=False, dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum for SGD')

parser.add_argument('--drop', '--dropout', default=0.1, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--alpha', default=0.99, type=float, metavar='M',
                         help='alpha for RMSprop (default: 0.99)')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--mixup', default=True, type=bool,help='use mixup training strategy')
parser.add_argument('--mixup-alpha', default=0.2, type=float,help='mixup parameter setting')


# metric
parser.add_argument('--threshold', default=0.3, type=float, help='threshold of classify')
parser.add_argument('--score_metric', default='loss', type=str, choices=['loss', 'f2'], help='Type of score metric')


# Architecture
parser.add_argument('--model-name', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--global-pool', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
# Misc
parser.add_argument('--manual-seed', default=2020, type=int, help='manual seed')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                    help='Start with pretrained version of specified network')


# Device setting
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()


if __name__ == "__main__":
    print(args.multi_label)