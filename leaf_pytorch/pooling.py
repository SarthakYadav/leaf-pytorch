import torch
from torch import nn
from torch.nn import functional as F
from leaf_pytorch import impulse_responses
from leaf_pytorch.utils import get_padding_value


class GaussianLowPass(nn.Module):
    def __init__(self, in_channels, kernel_size, strides=1,
                 padding="same", use_bias=True):
        super(GaussianLowPass, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.in_channels = in_channels

        w = torch.ones((1, 1, in_channels, 1)) * 0.4
        # const init of 0.4 makes it approximate a Hanning window
        self.weights = nn.Parameter(w)
        if self.use_bias:
            self._bias = torch.nn.Parameter(torch.ones(in_channels,))
        else:
            self._bias = None

        if self.padding.lower() == "same":
            self.pad_value = get_padding_value(kernel_size)
        else:
            self.pad_value = self.padding

    def forward(self, x):
        kernel = impulse_responses.gaussian_lowpass(self.weights, self.kernel_size)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.permute(2, 0, 1)

        if self.padding.lower() == "same":
            x = nn.functional.pad(x, self.pad_value, mode='constant', value=0)
            pad_val = 0
        else:
            pad_val = self.pad_value
        outputs = F.conv1d(x, kernel, bias=self._bias, stride=self.strides, padding=pad_val, groups=self.in_channels)
        return outputs
