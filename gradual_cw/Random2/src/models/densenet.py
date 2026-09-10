"""DenseNet model definitions for CIFAR-style inputs.

Adapted from:
https://github.com/weiaicunzai/pytorch-cifar100
(original file path: models/densenet.py)
"""

import torch
import torch.nn as nn
import math


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate
        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.residual_function(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.down_sample(x)


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module(f"dense_block_{i + 1}", self._make_dense_layer(block, inner_channels, nblocks[i]))
            inner_channels += nblocks[i] * self.growth_rate
            out_channels = int(reduction * inner_channels)
            self.features.add_module(f"transition_layer_{i + 1}", Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module(
            f"dense_block_{len(nblocks)}",
            self._make_dense_layer(block, inner_channels, nblocks[len(nblocks) - 1]),
        )
        inner_channels += nblocks[len(nblocks) - 1] * self.growth_rate

        self.features.add_module("bn", nn.BatchNorm2d(inner_channels))
        self.features.add_module("relu", nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inner_channels, num_classes)

        self._initialize_weights()

    def _make_dense_layer(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for i in range(nblocks):
            dense_block.add_module(f"bottleneck_layer_{i + 1}", block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def densenet121(num_classes=10):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes)


def densenet169(num_classes=10):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes)


def densenet201(num_classes=10):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes)


def densenet161(num_classes=10):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_classes=num_classes)
