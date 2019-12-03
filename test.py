#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
test for the PALNet
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
University of Adelaide
18/11/2018
"""

import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import datetime
from tqdm import tqdm

import models
import SscDataLoader
import sscMetrics

print(torch.__version__)


parser = argparse.ArgumentParser(description='PyTorch version PALNet for SSC')
parser.add_argument('--data_test', default='./test', metavar='DIR', help='path to test dataset')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--encoding', default='TSDF', type=str, metavar='Encoding', help='encoding of input voxels')


global args
args = parser.parse_args()

def main():
    # ---- Check CUDA
    if torch.cuda.is_available():
        print("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
    else:
        print("Sorry, You DO NOT have a CUDA device!")
        return

    # ---- Evaluation    
    time_start = datetime.datetime.now()
    if args.resume:
        if not os.path.isfile(args.resume):
            raise Exception("=> No checkpoint found at '{}'".format(args.resume))
    else:
        raise Exception("=> NO checkpoint")
    print('Test mode. Load checkpoint {}'.format(args.resume))

    net = models.PALNet().cuda()

    load_checkpoint = torch.load(args.resume)
    net.load_state_dict(load_checkpoint['state_dict_G'])

    test_loader = torch.utils.data.DataLoader(
        dataset=SscDataLoader.NYUv2Dataset(args.data_test, 'TEST_TSDF', encoding=args.encoding, downsample=4),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    t_p, t_r, t_iou, t_acc, t_ssc_iou, t_m_iou = validate_on_dataset(net, test_loader)
    print('Results of completion] p {:.4f}, r {:.4f}, IoU {:.4f}'.format(t_p, t_r, t_iou))
    print('Results semantic scene completion] mIoU {:.4f}, SSC IoU:{}'.format(t_m_iou, t_ssc_iou))


def validate_on_dataset(model, date_loader, save_ply=False):
    """
    Evaluate on validation set.
        model: network with parameters loaded
        date_loader: TEST mode
    """
    ply_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H.%M.%S")
    model.eval()  # switch to evaluate mode.
    val_acc, val_p, val_r, val_iou = 0.0, 0.0, 0.0, 0.0
    _C = 12
    val_cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
    val_iou_ssc = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
    count = 0

    for step, (depth, ftsdf, y_true, nonempty, position, filename) in tqdm(enumerate(date_loader), desc='Validating', unit='frame'):
        var_2d_depth = Variable(depth.float()).cuda()
        var_3d_ftsdf = Variable(ftsdf.float()).cuda()
        position = position.long().cuda()
        y_pred = model(x_tsdf=var_3d_ftsdf, x_depth=var_2d_depth, p=position)  # y_pred.size(): (bs, C, W, H, D)
        y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy
        y_true = y_true.numpy()  # torch tensor to numpy
        nonempty = nonempty.numpy()
        p, r, iou, acc, iou_sum, cnt_class = validate_on_batch(y_pred, y_true, nonempty)
        count += 1
        val_acc += acc
        val_p += p
        val_r += r
        val_iou += iou
        val_iou_ssc = np.add(val_iou_ssc, iou_sum)
        val_cnt_class = np.add(val_cnt_class, cnt_class)
        # print('acc_w, acc, p, r, iou', acc_w, acc, p, r, iou)
        # y_pred = y_pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy

    val_acc = val_acc / count
    val_p = val_p / count
    val_r = val_r / count
    val_iou = val_iou / count
    # val_iou_ssc = np.divide(val_iou_ssc, val_cnt_class)  # what if cnt_class[i]==0
    # val_iou_ssc_mean = np.mean(val_iou_ssc)  # what if cnt_class[i]==0
    val_iou_ssc, val_iou_ssc_mean = sscMetrics.get_iou(val_iou_ssc, val_cnt_class)
    return val_p, val_r, val_iou, val_acc, val_iou_ssc, val_iou_ssc_mean


def validate_on_batch(predict, target, nonempty=None):  # CPU
    """
        predict: (bs, channels, D, H, W)
        target:  (bs, channels, D, H, W)
    """
    y_pred = predict
    y_true = target
    p, r, iou = sscMetrics.get_score_completion(y_pred, y_true, nonempty)
    acc, iou_sum, cnt_class, tp_sum, fp_sum, fn_sum = sscMetrics.get_score_semantic_and_completion(y_pred, y_true, nonempty)
    return p, r, iou, acc, iou_sum, cnt_class


if __name__ == '__main__':
    main()
