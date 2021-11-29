import torch
from torch import nn


def get_padding_value(kernel_size):
    kernel_sizes = (kernel_size,)
    from functools import reduce
    from operator import __add__
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
    return conv_padding
