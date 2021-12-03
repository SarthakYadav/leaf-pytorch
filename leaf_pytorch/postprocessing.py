import torch
from torch import nn


class ExponentialMovingAverage(nn.Module):
    def __init__(self, in_channels, coeff_init, per_channel: bool = False):
        super(ExponentialMovingAverage, self).__init__()
        self._coeff_init = coeff_init
        self._per_channel = per_channel
        weights = torch.ones(in_channels,) if self._per_channel else torch.ones(1,)
        self._weights = nn.Parameter(weights * self._coeff_init)

    def forward(self, x):
        w = torch.clamp(self._weights, min=0., max=1.)
        initial_state = x[:, :, 0]

        def scan(init_state, x, w):
            x = x.permute(2, 0, 1)
            acc = init_state
            results = []
            for ix in range(len(x)):
                acc = (w * x[ix]) + ((1.0 - w) * acc)
                results.append(acc.unsqueeze(0))
            results = torch.cat(results, dim=0)
            results = results.permute(1, 2, 0)
            return results

        return scan(initial_state, x, w)


class PCENLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 alpha: float = 0.96,
                 smooth_coef: float = 0.04,
                 delta: float = 2.0,
                 root: float = 2.0,
                 floor: float = 1e-6,
                 trainable: bool = False,
                 learn_smooth_coef: bool = False,
                 per_channel_smooth_coef: bool = False):
        super(PCENLayer, self).__init__()
        self._alpha_init = alpha
        self._delta_init = delta
        self._root_init = root
        self._smooth_coef = smooth_coef
        self._floor = floor
        self._trainable = trainable
        self._learn_smooth_coef = learn_smooth_coef
        self._per_channel_smooth_coef = per_channel_smooth_coef

        self.alpha = nn.Parameter(torch.ones(in_channels) * self._alpha_init)
        self.delta = nn.Parameter(torch.ones(in_channels) * self._delta_init)
        self.root = nn.Parameter(torch.ones(in_channels) * self._root_init)

        if self._learn_smooth_coef:
            self.ema = ExponentialMovingAverage(in_channels, coeff_init=self._smooth_coef,
                                                per_channel=self._per_channel_smooth_coef)
        else:
            raise ValueError("SimpleRNN based ema not implemented.")

    def forward(self, x):
        alpha = torch.min(self.alpha, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        root = torch.max(self.root, torch.tensor(1.0, dtype=x.dtype, device=x.device))
        ema_smoother = self.ema(x)
        one_over_root = 1. / root
        output = ((x / (self._floor + ema_smoother) ** alpha.view(1, -1, 1) + self.delta.view(1, -1, 1))
                  ** one_over_root.view(1, -1, 1) - self.delta.view(1, -1, 1) ** one_over_root.view(1, -1, 1))
        return output
