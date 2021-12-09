import torch
from torch import nn
from leaf_pytorch import convolution
from leaf_pytorch import initializers
from leaf_pytorch import pooling
from leaf_pytorch import postprocessing
from leaf_pytorch import utils


class SquaredModulus(nn.Module):
    def __init__(self):
        super(SquaredModulus, self).__init__()
        self._pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.transpose(1, 2)
        output = 2 * self._pool(x ** 2.)
        output = output.transpose(1, 2)
        return output


class Leaf(nn.Module):
    def __init__(
            self,
            n_filters: int = 40,
            sample_rate: int = 16000,
            window_len: float = 25.,
            window_stride: float = 10.,
            preemp: bool = False,
            init_min_freq = 60.0,
            init_max_freq = 7800.0,
            mean_var_norm: bool = False,
            pcen_compression: bool = True,
            use_legacy_complex=False,
            initializer="default"
    ):
        super(Leaf, self).__init__()
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)
        if preemp:
            raise NotImplementedError("Pre-emp functionality not implemented yet..")
        else:
            self._preemp = None
        if initializer == "default":
            initializer = initializers.GaborInit(
                default_window_len=window_size, sample_rate=sample_rate,
                min_freq=init_min_freq, max_freq=init_max_freq
            )
        self._complex_conv = convolution.GaborConv1d(
            filters=2 * n_filters,
            kernel_size=window_size,
            strides=1,
            padding="same",
            use_bias=False,
            initializer=initializer,
            use_legacy_complex=use_legacy_complex
        )
        self._activation = SquaredModulus()
        self._pooling = pooling.GaussianLowPass(n_filters, kernel_size=window_size,
                                                strides=window_stride, padding="same")
        self._instance_norm = None
        if mean_var_norm:
            raise NotImplementedError("Instance Norm functionality not added yet..")
        if pcen_compression:
            self._compression = postprocessing.PCENLayer(
                n_filters,
                alpha=0.96,
                smooth_coef=0.04,
                delta=2.0,
                floor=1e-12,
                trainable=True,
                learn_smooth_coef=True,
                per_channel_smooth_coef=True)
        else:
            self._compression = None
        self._maximum_val = torch.tensor(1e-5)

    def forward(self, x):
        if self._preemp:
            x = self._preemp(x)
        outputs = self._complex_conv(x)
        outputs = self._activation(outputs)
        outputs = self._pooling(outputs)
        outputs = torch.maximum(outputs, torch.tensor(1e-5, device=outputs.device))
        if self._compression:
            outputs = self._compression(outputs)
        if self._instance_norm is not None:
            outputs = self._instance_norm(outputs)
        return outputs
