import os
import torch
import time


class RawAudioParser(object):
    """
    :param normalize_waveform
        whether to N(0,1) normalize audio waveform
    """
    def __init__(self, normalize_waveform=False):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        if self.normalize_waveform: print("ATTENTION!!! Mean/AVG norm on")

    def __call__(self, audio):
        output = torch.from_numpy(audio.astype("float32")).float()
        if self.normalize_waveform:
            mean = output.mean()
            std = output.std()
            output = (output - mean) / (std + 1e-9)
        output = output.unsqueeze(0)
        return output, None
