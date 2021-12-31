import torch
import random
import soundfile as sf
import numpy as np
import io


def _collate_fn_raw_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, channel_size, max_seqlength), dtype=torch.complex64)
    targets = torch.LongTensor(minibatch_size)
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets[x] = target
    return inputs, inputs_complex, targets


def _collate_fn_raw(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    # print("channel size:", channel_size)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, channel_size, max_seqlength), dtype=torch.complex64)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets.append(target.unsqueeze(0))
    targets = torch.cat(targets)
    return inputs, inputs_complex, targets


def _collate_fn_contrastive(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    batch_xi = torch.zeros(minibatch_size, channel_size, max_seqlength)
    batch_xj = torch.zeros(minibatch_size, channel_size, max_seqlength)
    targets = torch.LongTensor(minibatch_size)
    targets_supervised = []
    for ix in range(minibatch_size):
        sample = batch[ix]
        x_i = sample[0]
        x_j = sample[1]
        target = sample[2]
        supervised_target = sample[3]
        seq_length_i = x_i.size(1)
        seq_length_j = x_j.size(1)
        batch_xi[ix].narrow(1, 0, seq_length_i).copy_(x_i)
        batch_xj[ix].narrow(1, 0, seq_length_j).copy_(x_j)
        # print("[_collate]:", x_i.shape)
        # print("[_collate]:", x_j.shape)
        # print("[_collate]:", target)
        targets[ix] = target
        targets_supervised.append(supervised_target.unsqueeze(0))
    targets_supervised = torch.cat(targets_supervised)
    return batch_xi, batch_xj, targets, targets_supervised


def load_audio(f, sr, min_duration: float = 5.,
               read_cropped=False, frames_to_read=-1, audio_size=None):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    # x, clip_sr = torchaudio.load(f, channels_first=False)
    # x = x.squeeze().cpu().numpy()
    if read_cropped:
        assert audio_size
        assert frames_to_read != -1
        if frames_to_read >= audio_size:
            start_idx = 0
        else:
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
        x, clip_sr = sf.read(f, frames=frames_to_read, start=start_idx)
        # print("start_idx: {} | clip size: {} | frames_to_read:{}".format(start_idx, len(x), frames_to_read))
        min_samples = frames_to_read
    else:
        x, clip_sr = sf.read(f)     # sound file is > 3x faster than torchaudio sox_io
    x = x.astype('float32')#.cpu().numpy()
    assert clip_sr == sr

    # min filtering and padding if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x


def load_audio_bytes(buffer, sr, min_duration: float = 5.,
                     read_cropped=False, frames_to_read=-1, audio_size=None):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    load_full = True
    if read_cropped:
        load_full = False
        assert audio_size
        assert frames_to_read != -1
        if frames_to_read >= audio_size:
            with io.BytesIO(buffer) as buf:
                x, clip_sr = sf.read(buf)
        else:
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
            with io.BytesIO(buffer) as buf:
                try:
                    x, clip_sr = sf.read(buf, frames=frames_to_read, start=start_idx)
                except RuntimeError as ex:
                    load_full = True
                    print("{} {} {}. Attempting full read..".format(ex, start_idx, frames_to_read))
            if load_full:
                with io.BytesIO(buffer) as buf:
                    try:
                        x, clip_sr = sf.read(buf)
                        x = x[start_idx:start_idx+frames_to_read]
                    except RuntimeError as ex:
                        print("Catastrophic read failure. {} {} {}".format(ex, start_idx, frames_to_read))
                        return None
        min_samples = frames_to_read
    else:
        with io.BytesIO(buffer) as buf:
            try:
                x, clip_sr = sf.read(buf)
            except RuntimeError as ex:
                # no getting away with this now
                print("Catastrophic read failure. {} {} {}".format(ex, start_idx, frames_to_read))
                return None
    x = x.astype('float32')
    assert clip_sr == sr

    # pad if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x
