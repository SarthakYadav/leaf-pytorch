import torch
import math
from typing import Tuple, Callable
from torch import nn
from leaf_pytorch.initializers import GaborInit
from leaf_pytorch.impulse_responses import gabor_filters
from leaf_pytorch.utils import get_padding_value


class GaborConstraint(nn.Module):
    def __init__(self, kernel_size):
        super(GaborConstraint, self).__init__()
        self._kernel_size = kernel_size

    def forward(self, kernel_data):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * torch.sqrt(2 * torch.log(torch.tensor(2))) / math.pi
        sigma_upper = self._kernel_size * torch.sqrt(2 * torch.log(torch.tensor(2))) / math.pi
        clipped_mu = torch.clamp(kernel_data[:, 0], mu_lower, mu_upper).unsqueeze(1)
        clipped_sigma = torch.clamp(kernel_data[:, 1], sigma_lower, sigma_upper).unsqueeze(1)
        return torch.cat([clipped_mu, clipped_sigma], dim=-1)


class GaborConv1d(nn.Module):
    def __init__(self, filters, kernel_size,
                 strides, padding,
                 initializer,
                 use_bias=False,
                 sort_filters=False):
        super(GaborConv1d, self).__init__()
        self._filters = filters // 2
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._sort_filters = sort_filters

        assert isinstance(initializer, Callable)
        #     initializer = override_initializer
        # else:

        # initializer = GaborInit(self._filters, default_window_len=self._kernel_size,
        #                             sample_rate=16000, min_freq=60.0, max_freq=7800.0)
        init_weights = initializer((self._filters, 2))
        self.constraint = GaborConstraint(self._kernel_size)
        self._kernel = nn.Parameter(init_weights)
        if self._padding.lower() == "same":
            self._pad_value = get_padding_value(self._kernel_size)
        else:
            self._pad_value = self._padding
        if self._use_bias:
            self._bias = torch.nn.Parameter(torch.ones(self._filters*2,))
        else:
            self._bias = None

    def forward(self, x):
        # apply Gabor constraint
        kernel = self.constraint(self._kernel)
        if self._sort_filters:
            raise NotImplementedError("sort filter functionality not yet implemented")
        filters = gabor_filters(kernel, self._kernel_size)
        real_filters = filters.real
        img_filters = filters.imag
        stacked_filters = torch.cat([real_filters.unsqueeze(1), img_filters.unsqueeze(1)], dim=1)
        stacked_filters = torch.reshape(stacked_filters, (2 * self._filters, self._kernel_size))
        stacked_filters = stacked_filters.unsqueeze(1)

        if self._padding.lower() == "same":
            x = nn.functional.pad(x, self._pad_value, mode='constant', value=0)
            pad_val = 0
        else:
            pad_val = self._pad_value

        output = nn.functional.conv1d(x, stacked_filters,
                                      bias=self._bias, stride=self._strides, padding=pad_val)
        return output
