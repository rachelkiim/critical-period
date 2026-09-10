"""MLP model composed of custom linear layers."""

import torch
import torch.nn as nn

from src.models.linear import Linear

class LinearNetwork(nn.Module):
    def __init__(self, in_features, num_layers, num_hidden_list, mode, w_seed=-1, b_seed=-1):
        super(LinearNetwork, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list
        self.mode = mode

        self.linear = [Linear(self.in_features, self.num_hidden_list[0], True, self.mode, 0, w_seed, b_seed)]
        self.batchnorm = [nn.BatchNorm1d(self.num_hidden_list[0])]
        self.relu = [nn.ReLU()]

        for idx in range(self.num_layers - 1):
            self.linear.append(Linear(self.num_hidden_list[idx], self.num_hidden_list[idx + 1], True, mode, idx+1, w_seed, b_seed))
            self.batchnorm.append(nn.BatchNorm1d(self.num_hidden_list[idx + 1]))
            self.relu.append(nn.ReLU())
        self.linear = nn.ModuleList(self.linear)
        self.relu = nn.ModuleList(self.relu)
        self.batchnorm = nn.ModuleList(self.batchnorm)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        for idx in range(self.num_layers - 1):
            x = self.linear[idx](x)
            x = self.relu[idx](x)
            x = self.batchnorm[idx](x)
        x = self.linear[-1](x)
        return x
