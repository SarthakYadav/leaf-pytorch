import math
import torch


def gabor_impulse_response(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (-t ** 2.).unsqueeze(0), dims=1))
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        torch.complex(torch.tensor(0.), torch.tensor(1.))
        * torch.tensordot(center_frequency_complex.unsqueeze(1), t_complex.unsqueeze(0), dims=1)
    )
    denominator = denominator.type(torch.complex64).unsqueeze(1)
    gaussian = gaussian.type(torch.complex64)
    return denominator * sinusoid * gaussian


def gabor_impulse_response_legacy_complex(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)
    gaussian = torch.exp(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (-t ** 2.).unsqueeze(0), dims=1))
    temp = torch.tensordot(center.unsqueeze(1), t.unsqueeze(0), dims=1)
    temp2 = torch.zeros(*temp.shape + (2,), device=temp.device)

    # since output of torch.tensordot(..) is multiplied by 0+j
    # output can simply be written as flipping real component of torch.tensordot(..) to the imag component

    temp2[:, :, 0] *= -1 * temp2[:, :, 0]
    temp2[:, :, 1] = temp[:, :]

    # exponent of complex number c is
    # o.real = exp(c.real) * cos(c.imag)
    # o.imag = exp(c.real) * sin(c.imag)

    sinusoid = torch.zeros_like(temp2, device=temp.device)
    sinusoid[:, :, 0] = torch.exp(temp2[:, :, 0]) * torch.cos(temp2[:, :, 1])
    sinusoid[:, :, 1] = torch.exp(temp2[:, :, 0]) * torch.sin(temp2[:, :, 1])

    # multiplication of two complex numbers c1 and c2 -> out:
    # out.real = c1.real * c2.real - c1.imag * c2.imag
    # out.imag = c1.real * c2.imag + c1.imag * c2.real

    denominator_sinusoid = torch.zeros(*temp.shape + (2,), device=temp.device)
    denominator_sinusoid[:, :, 0] = (
            (denominator.view(-1, 1) * sinusoid[:, :, 0])
            - (torch.zeros_like(denominator).view(-1, 1) * sinusoid[:, :, 1])
    )
    denominator_sinusoid[:, :, 1] = (
            (denominator.view(-1, 1) * sinusoid[:, :, 1])
            + (torch.zeros_like(denominator).view(-1, 1) * sinusoid[:, :, 0])
    )

    output = torch.zeros(*temp.shape + (2,), device=temp.device)

    output[:, :, 0] = (
        (denominator_sinusoid[:, :, 0] * gaussian)
        - (denominator_sinusoid[:, :, 1] * torch.zeros_like(gaussian))
    )
    output[:, :, 1] = (
            (denominator_sinusoid[:, :, 0] * torch.zeros_like(gaussian))
            + (denominator_sinusoid[:, :, 1] * gaussian)
    )
    return output


def gabor_filters(kernel, size: int = 401, legacy_complex=False):
    t = torch.arange(-(size // 2), (size + 1) // 2, dtype=kernel.dtype, device=kernel.device)
    if not legacy_complex:
        return gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])
    else:
        return gabor_impulse_response_legacy_complex(t, center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma, filter_size: int):
    sigma = torch.clamp(sigma, min=(2. / filter_size), max=0.5)
    t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
    t = torch.reshape(t, (1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator) ** 2)
