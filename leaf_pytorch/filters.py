import math

import torch
import torchaudio
import numpy as np
from leaf_pytorch import impulse_responses
from torch import nn


class GaborFilter():
    def __init__(self,
                 n_filters: int = 40,
                 min_freq: float = 0.,
                 max_freq: float = 8000.,
                 sample_rate: int = 16000,
                 window_len: int = 401,
                 n_fft: int = 512,
                 normalize_energy: bool = False):
        super(GaborFilter, self).__init__()
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy

    def gabor_params_from_mels(self):
        coeff = torch.sqrt(2. * torch.log(torch.tensor(2.))) * self.n_fft
        sqrt_filters = torch.sqrt(self.mel_filters())
        center_frequencies = torch.argmax(sqrt_filters, dim=1)
        peaks, _ = torch.max(sqrt_filters, dim=1, keepdim=True)
        half_magnitudes = peaks / 2.
        fwhms = torch.sum((sqrt_filters >= half_magnitudes).float(), dim=1)
        output = torch.cat([
            (center_frequencies * 2 * np.pi / self.n_fft).unsqueeze(1),
            (coeff / (np.pi * fwhms)).unsqueeze(1)
        ], dim=-1)
        print(output.shape)
        return output

    def _mel_filters_areas(self, filters):
        peaks, _ = torch.max(filters, dim=1, keepdim=True)
        return peaks * (torch.sum((filters > 0).float(), dim=1, keepdim=True) + 2) * np.pi / self.n_fft


    def mel_filters(self):
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.n_filters,
            sample_rate=self.sample_rate
        )
        mel_filters = mel_filters.transpose(1, 0)
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
        return mel_filters

    def gabor_filters(self):
        gabor_filters = impulse_responses.gabor_filters(self.gabor_params_from_mels, size=self.window_len)
        output = gabor_filters * torch.sqrt(
            self._mel_filters_areas(self.mel_filters) * 2 * math.sqrt(math.pi) * self.gabor_params_from_mels[:, 1:2]
        ).type(torch.complex64)
        return output
