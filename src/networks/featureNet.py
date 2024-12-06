import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np  # 用于计算参数量
from base.base_net import BaseNet  # 确保路径正确

class FeatureNet(BaseNet):
    """A fully connected network with three hidden layers for feature vectors."""

    def __init__(self, input_dim=116, hidden_dim1=128, hidden_dim2=64, hidden_dim3=32, output_dim=64):
        super(FeatureNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)   # 第一隐藏层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) # 第二隐藏层
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3) # 第三隐藏层
        self.fc4 = nn.Linear(hidden_dim3, output_dim)  # 输出层
        self.rep_dim = output_dim  # 表示最后一层的输出维度

    def forward(self, x):
        """Forward pass logic."""
        x = F.relu(self.fc1(x))  # 第一隐藏层激活
        x = F.relu(self.fc2(x))  # 第二隐藏层激活
        x = F.relu(self.fc3(x))  # 第三隐藏层激活
        x = self.fc4(x)          # 输出层，不需要激活
        return x

