"""
Copyright VIP Group
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/rwightman/pytorch-image-models
Original copyright of Ross Wightman below, modifications by VIP Group

Hacked together by / copyright Ross Wightman
"""
import jittor as jt
from jittor import nn
import jittor.nn as F

from .conv_bn_act import ConvBnAct
from .create_act import create_act_layer, get_act_layer
from .helpers import make_divisible


class ChannelAttn(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """

    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(ChannelAttn, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = create_act_layer(gate_layer)

    def execute(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.amax((2, 3), keepdim=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttn):
    """An experimental 'lightweight' that sums avg + max pool first
    """

    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightChannelAttn, self).__init__(
            channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias)

    def execute(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.max((2, 3), keepdim=True)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * x_attn.sigmoid()


class SpatialAttn(nn.Module):
    """ Original CBAM spatial attention module
    """

    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(SpatialAttn, self).__init__()
        self.conv = ConvBnAct(2, 1, kernel_size, act_layer=None)
        self.gate = create_act_layer(gate_layer)

    def execute(self, x):
        x_attn = jt.concat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """

    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvBnAct(1, 1, kernel_size, act_layer=None)
        self.gate = create_act_layer(gate_layer)

    def execute(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.max(dim=1, keepdim=True)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class CbamModule(nn.Module):
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=1,
            spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(CbamModule, self).__init__()
        self.channel = ChannelAttn(
            channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = SpatialAttn(spatial_kernel_size, gate_layer=gate_layer)

    def execute(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class LightCbamModule(nn.Module):
    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=1,
            spatial_kernel_size=7, act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightCbamModule, self).__init__()
        self.channel = LightChannelAttn(
            channels, rd_ratio=rd_ratio, rd_channels=rd_channels,
            rd_divisor=rd_divisor, act_layer=act_layer, gate_layer=gate_layer, mlp_bias=mlp_bias)
        self.spatial = LightSpatialAttn(spatial_kernel_size)

    def execute(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x
