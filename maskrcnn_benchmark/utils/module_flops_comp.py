import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from maskrcnn_benchmark.utils.registry import Registry

sys.setrecursionlimit(10000)


MODULE_FLOPS_COMP = Registry()


@MODULE_FLOPS_COMP.register("Conv2d")
def build_conv2d_flops(module: Module, oup: Tensor):
    kh, kw = module.kernel_size
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups
    h, w = oup.shape[-2:]
    return in_channels * out_channels * w * h * kh * kw // groups


@MODULE_FLOPS_COMP.register("Linear")
def build_linear_flops(module: Module, oup: Tensor):
    in_features = module.in_features
    out_features = module.out_features
    return in_features * out_features