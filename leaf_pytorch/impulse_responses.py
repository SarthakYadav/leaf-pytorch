import math
import torch


def gabor_impulse_response(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (-t ** 2).unsqueeze(0), dims=1))
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        torch.complex(torch.tensor(0.), torch.tensor(1.))
        * torch.tensordot(center_frequency_complex.unsqueeze(1), t_complex.unsqueeze(0), dims=1)
    )
    denominator = denominator.type(torch.complex64).unsqueeze(1)
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401):
    t = torch.arange(-(size // 2), (size + 1) // 2, dtype=torch.float, device=kernel.device)
    return gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma, filter_size: int):
    sigma = torch.clamp(sigma, min=(2. / filter_size), max=0.5)
    t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
    t = torch.reshape(t, (1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator) ** 2)
