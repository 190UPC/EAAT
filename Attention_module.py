from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os
from moudle.A import *


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

"""se_eff"""
class attention(nn.Module):
    def __init__(self):
        super(attention,self).__init__()
        self.se = SE_Block(64)
        self.eff=EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)
    def forward(self, x):
        out = self.se(x)  # 通道注意力调整
        out = self.eff(out)

        return out
