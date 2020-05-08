from __future__ import absolute_import, division

import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps
import  matplotlib.pyplot as plt
from got10k.utils.viz import show_frame
import  matplotlib.pyplot as plt
class RandomStretch(object):                                 #随机拉伸

    def __init__(self, max_stretch=0.05, interpolation='bilinear'):
        assert interpolation in ['bilinear', 'bicubic']
        self.max_stretch = max_stretch
        self.interpolation = interpolation

    def __call__(self, img):
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        size = np.round(np.array(img.size, float) * scale).astype(int)#转换为int类型带缩放
        if self.interpolation == 'bilinear':
            method = Image.BILINEAR                                #双线性插值
        elif self.interpolation == 'bicubic':
            method = Image.BICUBIC
        return img.resize(tuple(size), method)   #报错就换tuple（size）为size


class Pairwise(Dataset):

    def __init__(self, seq_dataset, **kargs):
        super(Pairwise, self).__init__()
        self.cfg = self.parse_args(**kargs)

        self.seq_dataset = seq_dataset
        self.indices = np.random.permutation(len(seq_dataset))
        # augmentation for exemplar and instance images
        self.transform_z = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            CenterCrop(self.cfg.exemplar_sz),
            ToTensor()])
        self.transform_x = Compose([
            RandomStretch(max_stretch=0.05),
            CenterCrop(self.cfg.instance_sz - 8),
            RandomCrop(self.cfg.instance_sz - 2 * 8),
            ToTensor()])

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            'pairs_per_seq': 10,
            'max_dist': 100,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5}

        for key, val in kargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('GenericDict', cfg.keys())(**cfg)
        ##anno代表x,y最小值+w+h

