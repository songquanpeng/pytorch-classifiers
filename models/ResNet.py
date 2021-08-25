from typing import Type, Union, List

import torch.nn.functional as F
from torch import nn


class BasicResBlock(nn.Module):
    """ A building block for ResNet34 """
    expansion = 1

    def __init__(self, in_channels, planes, stride=1):
        """
        when stride=2, this block is used for downsampling
        """
        super().__init__()
        out_channels = planes * BasicResBlock.expansion
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.main = nn.Sequential(*layers)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = x
        out = self.main(x)
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = F.relu(out)
        return out


class BottleneckResBlock(nn.Module):
    """ A bottleneck building block for ResNet-50/101/152. """
    expansion = 4

    def __init__(self, in_channels, planes, stride=1):
        """
        when stride=2, this block is used for downsampling
        """
        super().__init__()
        out_channels = planes * BottleneckResBlock.expansion
        layers = [
            nn.Conv2d(in_channels, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.main = nn.Sequential(*layers)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = x
        out = self.main(x)
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicResBlock, BottleneckResBlock]], num_layers: List[int], num_classes: int,
                 in_channels: int = 3):
        super().__init__()
        # input shape: 3x224x224
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_planes, 7, 2, 3, bias=False),  # 64x112x112
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 64x56x56
        )
        self.conv2 = self.make_layer(block, num_layers[0], 64, downsample=False)  # 64x56x56
        self.conv3 = self.make_layer(block, num_layers[1], 128, downsample=True)  # 128x28x28
        self.conv4 = self.make_layer(block, num_layers[2], 256, downsample=True)  # 256x14x14
        self.conv5 = self.make_layer(block, num_layers[3], 512, downsample=True)  # 512x7x7
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 512x1x1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, num, planes, downsample=True):
        layers = [block(self.in_planes, planes, stride=2 if downsample else 1)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out


def ResNet18(args):
    return ResNet(BasicResBlock, [2, 2, 2, 2], args.num_classes, args.img_dim)


def ResNet34(args):
    return ResNet(BasicResBlock, [3, 4, 6, 3], args.num_classes, args.img_dim)


def ResNet50(args):
    return ResNet(BottleneckResBlock, [3, 4, 6, 3], args.num_classes, args.img_dim)


def ResNet101(args):
    return ResNet(BottleneckResBlock, [3, 4, 23, 3], args.num_classes, args.img_dim)


def ResNet152(args):
    return ResNet(BottleneckResBlock, [3, 8, 36, 3], args.num_classes, args.img_dim)
