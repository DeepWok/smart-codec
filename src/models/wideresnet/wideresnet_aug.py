import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms 
import numpy as np
from .wideresnet import ExampleWiseBatchNorm2d, VirtualBatchNorm2d, WideBasic, conv3x3, _bn_momentum


class WideResNetAug(nn.Module):
    def __init__(
            self, 
            depth, widen_factor, dropout_rate, num_classes, 
            adaptive_dropouter_creator, 
            adaptive_conv_dropouter_creator, 
            groupnorm, examplewise_bn, virtual_bn,
            my_transforms=None):
        super(WideResNetAug, self).__init__()
        self.in_planes = 16
        self.adaptive_conv_dropouter_creator = adaptive_conv_dropouter_creator

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        assert sum([groupnorm,examplewise_bn,virtual_bn]) <= 1
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.adaptive_dropouters = [] #nn.ModuleList()

        if groupnorm:
            print('Uses group norm.')
            self.norm_creator = lambda c: nn.GroupNorm(max(c//CpG, 1), c)
        elif examplewise_bn:
            print("Uses Example Wise BN")
            self.norm_creator = lambda c: ExampleWiseBatchNorm2d(c, momentum=_bn_momentum)
        elif virtual_bn:
            print("Uses Virtual BN")
            self.norm_creator = lambda c: VirtualBatchNorm2d(c, momentum=_bn_momentum)
        else:
            self.norm_creator = lambda c: nn.BatchNorm2d(c, momentum=_bn_momentum)

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = self.norm_creator(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)
        if adaptive_dropouter_creator is not None:
            last_dropout = adaptive_dropouter_creator(nStages[3])
        else:
            last_dropout = lambda x: x
        self.adaptive_dropouters.append(last_dropout)

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

    def to(self, *args, **kwargs):
        super().to(*args,**kwargs)
        print(*args)
        for ad in self.adaptive_dropouters:
            if hasattr(ad,'to'):
                ad.to(*args,**kwargs)
        return self

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for i,stride in enumerate(strides):
            ada_conv_drop_c = self.adaptive_conv_dropouter_creator if i == 0 else None
            new_block = block(self.in_planes, planes, dropout_rate, self.norm_creator, stride, adaptive_dropouter_creator=ada_conv_drop_c)
            layers.append(new_block)
            if ada_conv_drop_c is not None:
                self.adaptive_dropouters.append(new_block.dropout)

            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.training and 'post_conv' in self.transforms:
            out = self.transforms['post_conv'](out)

        out = self.layer1(out)
        if self.training and 'post_layer1' in self.transforms:
            out = self.transforms['post_layer1'](out)

        out = self.layer2(out)
        if self.training and 'post_layer2' in self.transforms:
            out = self.transforms['post_layer2'](out)

        out = self.layer3(out)
        if self.training and 'post_layer3' in self.transforms:
            out = self.transforms['post_layer3'](out)

        out = F.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.adaptive_dropouters[-1](out)
        out = self.linear(out)

        return out
