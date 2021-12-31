import os
import time
import glob
import augment
import torch_audiomentations
import torch
import random
import numpy as np
from utilities.data.utils import load_audio
from utilities.data.raw_waveform_parser import RawAudioParser


def image_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def gauss_noise(image, sigma_sq):
    h, w = image.shape
    gauss = np.random.normal(0, sigma_sq, (h, w))
    gauss = gauss.reshape(h, w)
    image = image + gauss
    return image


# Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    # spec = spec.copy()
    # print('[spec_augment] spec type', type(spec))
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec


class SpecAugment:
    def __init__(self,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
                            self.num_mask,
                            self.freq_masking,
                            self.time_masking,
                            image.min())


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, trg=None):
        if trg is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, trg = t(image, trg)
            return image, trg


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class ToTensor:
    def __call__(self, array):
        return torch.from_numpy(array).float()


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        start = random.randint(0, signal.shape[1] - self.size)
        return signal[:, start: start + self.size]


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):

        if signal.shape[1] > self.size:
            start = (signal.shape[1] - self.size) // 2
            return signal[:, start: start + self.size]
        else:
            return signal


class PadToSize_NP:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = np.pad(signal, pad_width,
                                'constant', constant_values=signal.min())
            else:
                signal = np.pad(signal, pad_width, 'wrap')
        return signal


class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        if signal.shape[1] < self.size:
            padding = self.size - signal.shape[1]
            offset = padding // 2
            pad_width = ((0, 0), (offset, padding - offset))
            if self.mode == 'constant':
                signal = torch.nn.functional.pad(signal, pad_width[1], "constant", value=signal.min())
                # signal = np.pad(signal, pad_width,
                #                 'constant', constant_values=signal.min())
            else:
                # signal = np.pad(signal, pad_width, 'wrap')
                try:
                    signal = torch.nn.functional.pad(signal, pad_width[1], "replicate")
                except NotImplementedError as ex:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! signal.shape", signal.shape, self.size)
        return signal


class TimeMasking:
    def __init__(self, time_perc=0.2, num_masks=2):
        self.time_perc = time_perc
        self.num_masks = num_masks

    def __call__(self, x):
        time0 = time.time()
        num_masks = random.randint(1, self.num_masks)
        for i in range(num_masks):
            _, timesteps = x.shape
            time_percentage = random.uniform(0.0, self.time_perc)
            num_frames_to_mask = int(time_percentage * timesteps)
            t0 = int(np.random.uniform(low=0.0, high=timesteps - num_frames_to_mask))
            # x[:, t0:t0 + num_frames_to_mask] = 0
            x[:, t0:t0 + num_frames_to_mask].zero_()

        time1 = time.time()
        return x


class ClipValue:
    def __init__(self, max_clip_val=0.1):
        self.clamp_factor = max_clip_val

    def __call__(self, x):
        factor = random.uniform(0.0, self.clamp_factor)
        x_min, x_max = x.min(), x.max()
        x.clamp_(min=x_min * factor, max=x_max * factor)
        return x


class RandomReverb:
    def __init__(self, reverb_range=(10, 50),
                 damping_range=(10, 50),
                 room_scale_range=(0, 100),
                 sampling_rate=16000):
        assert len(reverb_range) == 2
        assert len(damping_range) == 2
        assert len(room_scale_range) == 2
        self.reverb_min, self.reverb_max = reverb_range
        self.damping_min, self.damping_max = damping_range
        self.room_scale_min, self.room_scale_max = room_scale_range
        self.src_info = {"rate": sampling_rate}

    def __call__(self, x):
        t0 = time.time()
        reverberance = np.random.randint(self.reverb_min, self.reverb_max + 1)
        damping = np.random.randint(self.damping_min, self.damping_max + 1)
        room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)
        x = augment.EffectChain().reverb(reverberance, damping, room_scale).channels(1).apply(x, src_info=self.src_info)
        t1 = time.time()
        return x


class BackgroundNoiseGenerator:
    def __init__(self, noise_path,
                 in_memory=False, sr=16000,
                 min_duration=2, waveform_parser=None,
                 num_samples=16000):
        assert os.path.exists(noise_path)
        self.files = glob.glob(os.path.join(noise_path, "*.flac"))
        if len(self.files) == 0:
            self.files = glob.glob(os.path.join(noise_path, "*", "*.flac"))
        self.in_memory = in_memory
        self.sr = sr
        self.min_duration = min_duration
        self.waveform_parser = waveform_parser
        self.num_samples = num_samples
        self.tfs = Compose([
            PadToSize(num_samples, "wrap"),
            RandomCrop(num_samples)
        ])
        if self.in_memory:
            print("loading noise audio in memory")
            self.audios = []
            for f in self.files:
                audio = load_audio(f, sr, min_duration)
                if self.waveform_parser:
                    audio, _ = self.waveform_parser(audio)
                self.audios.append(audio)
            print("{} noises loaded in memory..".format(len(self.audios)))

    def __call__(self):
        idx = random.randint(0, len(self.files)-1)
        if self.in_memory:
            audio = self.audios[idx]
        else:
            audio = load_audio(self.files[idx], self.sr, self.min_duration)
            if self.waveform_parser:
                audio, _ = self.waveform_parser(audio)
        audio = self.tfs(audio)
        return audio


class AddRandomNoise:
    def __init__(self, noise_generator, snr_range=(10, 25)):
        self.noise_generator = noise_generator
        self.snr_range = snr_range

    def __call__(self, x):
        # snr = np.random.randint(self.snr_range[0], self.snr_range[1]+1)
        snr = np.random.uniform(self.snr_range[0], self.snr_range[1]+1)
        r = np.exp(snr * np.log(10) / 10)
        coeff = r / (1.0 + r)

        noise_instance = self.noise_generator()
        assert noise_instance.numel() == x.numel(
        ), 'Noise and signal shapes are incompatible'
        noised = coeff * x + (1.0 - coeff) * noise_instance.view_as(x)
        # print("[random noise] min: {} | max: {}".format(noised.min(), noised.max()))
        return noised


class RandomGain:
    def __init__(self, min_gain_in_db=-18.0,
                 max_gain_in_db=6.0, prob=0.5, sr=16000):
        self.gain = torch_audiomentations.Gain(min_gain_in_db=min_gain_in_db,
                                               max_gain_in_db=max_gain_in_db,
                                               p=prob, sample_rate=sr)

    def __call__(self, x):
        t0 = time.time()
        spec = x.detach().clone().unsqueeze(0)
        spec = self.gain(spec)
        t1 = time.time()
        return spec[0]


class AddGaussianNoise:
    """
    Adds Random Gaussian Noise
    this changes the amplitude of the audio and can yield values outside [-1., 1.]
    so normalize after this
    """
    def __init__(self, min_amplitude=0.001, max_amplitude=0.015):
        super().__init__()
        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
    
    def __call__(self, x):
        noise = torch.randn(x.shape).float()
        random_amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        return x + random_amplitude * noise


class PeakNormalization:
    def __init__(self, sr=16000):
        self.peak_norm = torch_audiomentations.PeakNormalization(apply_to="only_too_loud_sounds",
                                                                 p=1., sample_rate=sr)

    def __call__(self, x):
        t0 = time.time()
        spec = x.detach().clone().unsqueeze(0)
        spec = self.peak_norm(spec)
        t1 = time.time()
        return spec[0]


def get_raw_transforms_v2(train, size,
                          wrap_pad_prob=0.5,
                          sample_rate=16000,
                          min_duration=2,
                          max_clip_value=0.2,
                          background_noise_path=None,
                          center_crop_val=False):
    if train:
        tfs = [
            OneOf([
                PadToSize(size, mode='wrap'),
                PadToSize(size, mode='constant'),
            ], p=[wrap_pad_prob, 1 - wrap_pad_prob]),
            # Pad
            RandomCrop(size),
        ]
        if background_noise_path:
            noise_gen = BackgroundNoiseGenerator(background_noise_path, sr=sample_rate, min_duration=min_duration,
                                                 num_samples=size,
                                                 waveform_parser=RawAudioParser(normalize_waveform=False))
            bg_tf = AddRandomNoise(noise_gen)
            tfs.append(UseWithProb(bg_tf, prob=0.5))
        # tfs.append(UseWithProb(ClipValue(max_clip_value), prob=0.5))      #TODO CLIP NEEDS WORK

        # tfs.append(UseWithProb(RandomReverb(sampling_rate=sample_rate), prob=0.5))        # TODO: TOO SLOW
        tfs.append(RandomGain(sr=sample_rate))
        tfs.append(PeakNormalization(sr=sample_rate))
        tfs.append(TimeMasking(time_perc=0.1, num_masks=3))
        transforms = Compose(tfs)
    else:
        tfs = [PadToSize(size, "wrap")]
        if center_crop_val:
            tfs.append(CenterCrop(size))
        transforms = Compose(tfs)
    return transforms


def simple_supervised_transforms(is_train, size, sample_rate=8000):
    if is_train:
        tfs = [OneOf([
            PadToSize(size, mode='wrap'),
            PadToSize(size, mode='constant'),
        ], p=[0.5, 1 - 0.5]), RandomCrop(size), UseWithProb(RandomGain(sr=sample_rate), prob=0.5),
            UseWithProb(AddGaussianNoise(), prob=0.5), PeakNormalization(sr=sample_rate),
            TimeMasking(time_perc=0.1, num_masks=3)]
    else:
        tfs = [PadToSize(size, "wrap"), CenterCrop(size), PeakNormalization(sr=sample_rate)]
    transforms = Compose(tfs)
    return transforms


def leaf_supervised_transforms(is_train, size, sample_rate=16000):
    if is_train:
        tfs = [
            OneOf(
                [PadToSize(size, mode='wrap'),
                PadToSize(size, mode='constant')],
                p=[0.5, 1 - 0.5]), 
        RandomCrop(size), 
        UseWithProb(RandomGain(sr=sample_rate), prob=0.5),
        UseWithProb(AddGaussianNoise(), prob=0.5),
        PeakNormalization(sr=sample_rate)]
    else:
        tfs = [
            PadToSize(size, "wrap"), 
            CenterCrop(size), 
            PeakNormalization(sr=sample_rate)
        ]
    transforms = Compose(tfs)
    return transforms
