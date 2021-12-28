import os
import math
import time
import io
import lmdb
import tqdm
import glob
import numpy as np
import librosa
import torch
import json
import random
import pandas as pd
from google.cloud import storage
from torch.utils.data import Dataset
from typing import Tuple, Optional
from utilities.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass, _collate_fn_contrastive
from utilities.data.raw_waveform_parser import RawAudioParser
import soundfile as sf
from utilities.data.utils import load_audio, load_audio_bytes
import msgpack
import msgpack_numpy as msgnp
from utilities.data.raw_dataset import RawWaveformDataset


def readfile(f):
    with open(f, "rb") as stream:
        data = stream.read()
    return data


def unpack_batch(f):
    data = msgpack.unpackb(readfile(f), object_hook=msgnp.decode)
    return data


class PackedDataset(Dataset):
    def __init__(self, manifest_path, labels_map,
                 audio_config, augment=False,
                 mode='multilabel', delimiter=",",
                 mixer=None, transform=None, is_val=False,
                 cropped_read=False, gcs_bucket_path=None):
        super(PackedDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        self.mode = mode
        self.transform = transform
        self.mixer = mixer
        self.cropped_read = cropped_read
        self.is_val = is_val
        self.gcs_bucket_path = gcs_bucket_path
        if self.gcs_bucket_path:
            self.client = None
        # if gcs_bucket_path:
            # self.use_gcs = True
            # self.storage_client = storage.Client()
            # self.bucket = storage_client.get_bucket(gcs_bucket_path)

        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.labels_delim = delimiter
        self.parse_audio_config(audio_config)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        else:
            self.bg_files = None
        df = pd.read_csv(manifest_path)
        files = df['files'].values.tolist()
        # labels = df['labels'].values.tolist()
        self.files = files
        # self.labels = labels
        # if self.cropped_read:
        #     self.durations = df['durations'].values.tolist()
        self.spec_parser = RawAudioParser(normalize_waveform=self.normalize)
        self.length = len(self.files)

    def parse_audio_config(self, audio_config):
        self.sr = int(audio_config.get("sample_rate", "22050"))
        self.normalize = bool(audio_config.get("normalize", False))
        self.min_duration = float(audio_config.get("min_duration", 2.5))
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.cropped_read:
            self.num_frames = int(audio_config.get('random_clip_size') * self.sr)
        else:
            self.num_frames = -1

        delim = audio_config.get("delimiter", None)
        if delim is not None:
            print("Reassigning delimiter from audio_config")
            self.labels_delim = delim

    def __get_feature__(self, audio) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def __get_item_helper__(self, record) -> Tuple[torch.Tensor, torch.Tensor]:
        lbls = record['label']
        if self.cropped_read and not self.is_val:
            dur = record['duration']
        else:
            dur = None
        preprocessed_audio = load_audio_bytes(record['audio'],
                                              self.sr, self.min_duration,
                                              read_cropped=self.cropped_read,
                                              frames_to_read=self.num_frames,
                                              audio_size=dur)
        real, _ = self.__get_feature__(preprocessed_audio)
        label_tensor = self.__parse_labels__(lbls)
        if self.transform is not None:
            real = self.transform(real)
        return real, label_tensor

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            # print("multiclassssss")
            return self.labels_map[lbls]

    def __len__(self):
        return self.length
    
    def init_gcs(self):
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.gcs_bucket_path)

    def __getitem__(self, item):
        filepath = self.files[item]
        if self.gcs_bucket_path:
            if self.client is None:
                self.init_gcs()
            blob = self.bucket.blob(filepath)
            with blob.open("rb") as fp:
                read_block = msgpack.unpackb(fp.read(), object_hook=msgnp.decode)
        else:
            read_block = unpack_batch(filepath)
        if not self.is_val:
            idxs = np.random.permutation(len(read_block))
        else:
            idxs = np.arange(len(read_block))
        # batch = torch.zeros(len(read_block), 1, )
        batch_tensors = []
        batch_labels = []
        # for record in read_block:
        for idx in idxs:
            record = read_block[idx]
            real, label = self.__get_item_helper__(record)
            if self.mixer is not None:
                real, final_label = self.mixer(self, real, label)
                if self.mode != "multiclass":
                    label = final_label
            batch_tensors.append(real)
            batch_labels.append(label)
        return batch_tensors, batch_labels


def packed_collate_fn_raw_multiclass(batches):
    deflated_batch = []
    for batch in batches:
        for jx in range(len(batch[0])):
            deflated_batch.append((batch[0][jx], batch[1][jx]))
    return _collate_fn_raw_multiclass(deflated_batch)


def packed_collate_fn_raw_multilabel(batches):
    deflated_batch = []
    for batch in batches:
        for jx in range(len(batch[0])):
            deflated_batch.append((batch[0][jx], batch[1][jx]))
    return _collate_fn_raw(deflated_batch)
