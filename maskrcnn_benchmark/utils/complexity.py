import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from maskrcnn_benchmark.utils.module_flops_comp import MODULE_FLOPS_COMP


class Complexity(object):
    __all_modules__ = (nn.Conv2d,)
    """docstring for Complexity"""
    def __init__(self, mode='flops', module_flops_comp=None):
        super(Complexity, self).__init__()
        self.mode = mode
        self.flops = 0
        self.weights = 0
        if module_flops_comp is None:
            self.module_flops_comp = MODULE_FLOPS_COMP
        else:
            self.module_flops_comp = module_flops_comp

    def hook(self, module: Module, inp: (Tensor,), oup: Tensor):
        module_type = str(type(module)).split('\'')[1].split('.')[-1]
        self.flops += self.module_flops_comp[module_type](module, oup)

    def __call__(self, model: Module, x: Tensor, rngs: tuple) -> int:
        if self.mode == 'flops':
            self.flops = 0
            handles = []
            for idx, m in enumerate(model.modules()):
                if isinstance(m, self.__all_modules__):
                    handle = m.register_forward_hook(self.hook)
                    handles.append(handle)
            model.eval()
            with torch.no_grad():
                _ = model(x, rngs)
            for h in handles: h.remove()
            return self.flops
        elif self.mode == 'weights':
            self.weights = 0
            import numpy as np
            for name, param in model.named_parameters():
                if 'weight' in name and not 'bn' in name:
                    self.weights += np.prod(list(param.size()))
                pass
            return self.weights
        else:
            raise NotImplementedError


def main():
    complexity = Complexity()
    inp = torch.randn(1, 3, 224, 224)
    from resnet50 import resnet50
    model = resnet50()
    print(complexity(model, inp))
    print(complexity(model, inp))
    print(complexity(model, inp))


if __name__ == '__main__':
    main()