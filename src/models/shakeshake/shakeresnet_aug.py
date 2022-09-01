# -*- coding: utf-8 -*-

import math

import torch.nn as nn
import torch.nn.functional as F
from .shakeresnet import ShakeResNet
import torchvision.transforms as transforms


class ShakeResNetAug(ShakeResNet):

    def __init__(self, depth, w_base, num_classes, my_transforms=None):
        super(ShakeResNetAug, self).__init__(
            depth=depth, 
            w_base=w_base, 
            num_classes=num_classes)
        if my_transforms is not None:
            if isinstance(my_transforms, dict):
                self.transforms = my_transforms
            elif isinstance(my_transforms, transforms):
                self.transforms = {
                    'post_conv': my_transforms,
                    'post_layer1': my_transforms,
                    'post_layer2': my_transforms,
                    'post_layer3': my_transforms,
                }
            self.transforms = nn.ModuleDict(self.transforms)

    def forward(self, x):
        h = self.c_in(x)
        if self.training and 'post_conv' in self.transforms:
            out = self.transforms['post_conv'](out)

        h = self.layer1(h)
        if self.training and 'post_layer1' in self.transforms:
            out = self.transforms['post_layer1'](out)

        h = self.layer2(h)
        if self.training and 'post_layer2' in self.transforms:
            out = self.transforms['post_layer2'](out)

        h = self.layer3(h)
        if self.training and 'post_layer3' in self.transforms:
            out = self.transforms['post_layer3'](out)

        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.fc_out(h)
        return h
