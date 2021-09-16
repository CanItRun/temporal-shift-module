# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
from lumo import Params
import torch

best_prec1 = 0

from lumo import Params


class MyParams(Params):

    def __init__(self):
        super().__init__()
        self.dataset = self.choice('UCF101')
        self.arch = 'BNInception'
        self.num_class = 101
        self.num_segments = 1
        self.consensus_type = 'avg'
        self.k = 3
        self.dropout = 0.5
        self.img_feature_dim = 256
        self.suffix = None
        self.temporal_pool = True
        self.modality = 'RGB'
        self.shift = False
        self.shift_div = 8
        self.shift_place = 'blockres'
        self.temporal_pool = False
        self.non_local = False
        self.dense_sample = False
        self.pretrain = 'imagenet'

    def iparams(self):
        super().iparams()
        full_arch_name = self.arch
        if self.shift:
            full_arch_name += '_shift{}_{}'.format(self.shift_div, self.shift_place)
        if self.temporal_pool:
            full_arch_name += '_tpool'
        self.store_name = '_'.join(
            ['TSM', self.dataset, self.modality, full_arch_name, self.consensus_type, 'segment%d' % self.num_segments,
             'e{}'.format(self.epochs)])
        if self.pretrain != 'imagenet':
            self.store_name += '_{}'.format(self.pretrain)
        if self.lr_type != 'step':
            self.store_name += '_{}'.format(self.lr_type)
        if self.dense_sample:
            self.store_name += '_dense'
        if self.non_local > 0:
            self.store_name += '_nl'
        if self.suffix is not None:
            self.store_name += '_{}'.format(self.suffix)


if __name__ == '__main__':
    args = MyParams()
    args.from_args()

    model = TSN(args.num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=False,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=False,
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    x = torch.rand(2, 3, 8, 224, 224)
