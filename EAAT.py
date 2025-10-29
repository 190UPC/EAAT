import torch.nn as nn
import ops as ops
import torch
import torch.nn.functional as F
import numpy as np
import os
from moudle.A import *
from Attention_module import *
# from Mymoudle.trans_moudle import *
from moudle.LTB import *


class DBRAB(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(DBRAB, self).__init__()

        self.rb1 = ops.ResidualBlock(wn, 64, 64)
        # self.rb1 = ops.ResidualBlock_res(wn, 64, 64)

        # self.rb2 = ops.ResidualBlock(wn, 64, 64)
        # self.rb3 = ops.ResidualBlock(wn, 64, 64)

        # self.reduction_1 = ops.BasicConv2d(wn, 64*4, 64, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1, A_1 = self.rb1(o0)

        b2, A_2 = self.rb1(b1)

        b3, A_3 = self.rb1(b2)

        # Feature_bank = self.reduction_1(torch.cat([c0, b1, b2, b3],1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))

        # out = Feature_bank + x + Attention_bank
        out = x + Attention_bank
        return out, Attention_bank


class Net(nn.Module):

    def __init__(self, **kwargs):
        super(Net, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        self.b1 = DBRAB(64, 64, wn=wn)
        self.b2 = DBRAB(64, 64, wn=wn)
        self.b3 = DBRAB(64, 64, wn=wn)
        self.t = LightTransformerBlock(64)
        # self.reduction_1 = ops.BasicConv2d(wn, 64*4, 64, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)

        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))
        # self.casatt=CASAtt(dim=64)
        # self.efficent_attention = EfficientAttention(in_channels=64, key_channels=128, head_count=4, value_channels=128)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        res = x

        x = self.entry_1(x)
        c0 = o0 = x

        b1, A_1 = self.b1(o0)

        b2, A_2 = self.b2(b1)

        b3, A_3 = self.b3(b2)
        # A_1 = self.t(A_1)
        # A_2 = self.t(A_2)
        A_3 = self.t(A_3)
        # Feature_bank = self.reduction_1(torch.cat([c0, b1, b2, b3],1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))
        """---------------------------------------------------------------"""
        # print(Attention_bank.shape)
        # """修改"""
        # Feature_bank=self.casatt(Feature_bank)
        # Attention_bank=self.efficent_attention(Attention_bank)
        """-----------------------------------------------------------------"""
        out = x + Attention_bank

        out = self.upsample(out, scale=scale)

        out = self.exit1(out)

        skip = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        out = skip + out

        out = self.add_mean(out)

        return out
