import torch
import numpy as np
from torch import nn
from leaf_pytorch.filters import GaborFilter


class GaborInit:
    def __init__(self, default_window_len=401, **kwargs):
        super(GaborInit, self).__init__()
        self.def_win_len = default_window_len
        self._kwargs = kwargs

    def __call__(self, shape, dtype=None):
        n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
        window_len = self.def_win_len if len(shape) == 2 else shape[0]
        gabor_filters = GaborFilter(n_filters=n_filters, window_len=window_len, **self._kwargs)
        if len(shape) == 2:
            return gabor_filters.gabor_params_from_mels()
        else:
            # only needed in case of > 2-dim weights
            # even_indices = torch.arange(start=0, end=shape[2], step=2)
            # odd_indices = torch.arange(start=1, end=shape[2], step=2)
            # filters = gabor_filters.gabor_filters()
            raise NotImplementedError("implementation incomplete. Use even valued filter dimensions")
