import math
import torch


def gabor_impulse_response(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm**2), -t**2, dims=0))
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        torch.complex(torch.tensor(0.), torch.tensor(1.)) * torch.tensordot(center_frequency_complex, t_complex, dims=0)
    )
    denominator = denominator.type(torch.complex64).unsqueeze(1)
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401):
    t = torch.arange(-(size // 2), (size + 1) // 2, dtype=torch.float)
    return gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])
