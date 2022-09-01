'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from numpy import isin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 
from .resnet import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, my_transforms=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if my_transforms is not None:
            if isinstance(my_transforms, dict):
                self.transforms = my_transforms
            elif isinstance(my_transforms, transforms):
                self.transforms = {
                    'post_conv': my_transforms,
                    'post_layer1': my_transforms,
                    'post_layer2': my_transforms,
                    'post_layer3': my_transforms,
                    'post_layer4': my_transforms,
                } 
            self.transforms = nn.ModuleDict(self.transforms)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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

        out = self.layer4(out)
        if self.training and 'post_layer4' in self.transforms:
            out = self.transforms['post_layer4'](out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, my_transforms=None):
        super(ResNetImageNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        if my_transforms is not None:
            if isinstance(my_transforms, dict):
                self.transforms = my_transforms
            elif isinstance(my_transforms, transforms):
                self.transforms = {
                    'post_conv': my_transforms,
                    'post_layer1': my_transforms,
                    'post_layer2': my_transforms,
                    'post_layer3': my_transforms,
                    'post_layer4': my_transforms,
                }
            self.transforms = nn.ModuleDict(self.transforms)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out = x
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
        out = self.layer4(out)
        if self.training and 'post_layer4' in self.transforms:
            out = self.transforms['post_layer4'](out)

        out = self.avgpool(out)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(my_transforms=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], my_transforms=my_transforms)


def ResNet50(my_transforms=None, num_class=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], my_transforms=my_transforms, num_classes=num_class)


def ResNet18ImageNet(num_classes):
    return ResNetImageNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet50ImageNet(num_classes):
    return ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

