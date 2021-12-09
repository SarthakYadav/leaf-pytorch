import os
import torch
from torch import nn
from leaf_pytorch.frontend import Leaf


def get_frontend(opt):

    front_end_config = opt['frontend']
    audio_config = opt['audio_config']

    pretrained = front_end_config.get("pretrained", "")
    if os.path.isfile(pretrained):
        pretrained_flag = True
        ckpt = torch.load(pretrained)
    else:
        pretrained_flag = False

    if "leaf" in front_end_config['name'].lower():
        default_args = front_end_config.get("default_args", False)
        use_legacy_complex = front_end_config.get("use_legacy_complex", False)
        initializer = front_end_config.get("initializer", "default")
        if default_args:
            print("Using default Leaf arguments..")
            fe = Leaf(use_legacy_complex=use_legacy_complex, initializer=initializer)
        else:
            sr = int(audio_config.get("sample_rate", 16000))
            window_len_ms = float(audio_config.get("window_len", 25.))
            window_stride_ms = float(audio_config.get("window_stride", 10.))

            n_filters = int(front_end_config.get("n_filters", 40.0))
            min_freq = float(front_end_config.get("min_freq", 60.0))
            max_freq = float(front_end_config.get("max_freq", 7800.0))
            pcen_compress = bool(front_end_config.get("pcen_compress", True))
            mean_var_norm = bool(front_end_config.get("mean_var_norm", False))
            preemp = bool(front_end_config.get("preemp", False))
            fe = Leaf(
                n_filters=n_filters,
                sample_rate=sr,
                window_len=window_len_ms,
                window_stride=window_stride_ms,
                preemp=preemp,
                init_min_freq=min_freq,
                init_max_freq=max_freq,
                mean_var_norm=mean_var_norm,
                pcen_compression=pcen_compress,
                use_legacy_complex=use_legacy_complex,
                initializer=initializer
            )
    else:
        raise NotImplementedError("Other front ends not implemented yet.")
    if pretrained_flag:
        print("attempting to load pretrained frontend weights..", fe.load_state_dict(ckpt))
    return fe
