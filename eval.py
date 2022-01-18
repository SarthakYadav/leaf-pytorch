import math
import os
import tqdm
import glob
import datetime
import copy
import pickle
from threading import main_thread
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from audio_utils import packed_datasets, transforms_helper
from audio_utils.common import feature_transforms, transforms
from audio_utils.common.audio_config import AudioConfig, Features
from utilities.config_parser import parse_config, get_data_info, get_config
from utilities.training_utils import setup_dataloaders, optimization_helper
from models.classifier import Classifier
import argparse
from sklearn.metrics import accuracy_score
from utilities.metrics_helper import calculate_mAP, calculate_stats, d_prime


def get_val_acc(x):
    x = x.split("/")[-1]
    x = x.replace(".pth", "")
    x = x.split("val_acc=")[-1]
    return float(x)


parser = argparse.ArgumentParser()
parser.add_argument("--test_csv_name", type=str)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--meta_dir", type=str)
parser.add_argument("--device", type=str, default="gpu", help="device to run eval on [tpu/gpu]. Default is gpu")
parser.add_argument("--metrics", type=str, default="multiclass")
parser.add_argument("--separator", type=str, default=",")
parser.add_argument("--gcs_bucket_name", type=str)


def leaf_raw_supervised_transforms(audio_config: AudioConfig):
    # transforms corresponding to initial successful AudioSet supervised training
    # from cola-pytorch commit 21d1df2c
    if audio_config.features == Features.RAW:
        random_clip_size = audio_config.view_size
        val_clip_size = audio_config.view_size
        is_raw = True
    else:
        random_clip_size = audio_config.tr_feature_size
        val_clip_size = audio_config.val_feature_size
        is_raw = False
    pre_transforms = transforms.Compose(
        [
            transforms.UseWithProb(transforms.RandomGain()),
            transforms.UseWithProb(transforms.AddGaussianNoise()),
            transforms.PeakNormalization(),
        ]
    )
    train_tfs = {
        'pre': pre_transforms
    }
    val_tfs = {
        "pre": pre_transforms
    }
    mode = "per_instance"
    if audio_config.features == Features.RAW:
        # Raw waveform processor is called either way
        # just add augmentations
        train_tfs['post'] = transforms.Compose(
            [
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.SPECTROGRAM:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            normalize=audio_config.normalize_features,
            log_compress=True,
            mode=mode)

        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.LOGMEL:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            log_compress=False,
            normalize=audio_config.normalize_features,
            mode=mode
        )
        mel_trans = feature_transforms.ToMelScale(audio_config.sr, audio_config.hop_len,
                                                  n_fft=audio_config.n_fft, n_mels=audio_config.n_mels,
                                                  fmin=audio_config.fmin, fmax=audio_config.fmax)
        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.CenterCrop(val_clip_size)
            ])
    return train_tfs, val_tfs


def pad_input(signal, sr):

    signal = signal[0]
    # print("input signal shape:", signal.shape)
    size = int(math.ceil(signal.shape[1] / sr) * sr)
    # print(size)
    padding = size - signal.shape[1]
    offset = padding // 2
    pad_width = ((0, 0), (offset, padding - offset))
    signal = torch.nn.functional.pad(signal, pad_width[1], "replicate")
    signal = signal.unsqueeze(0)
    # print("padded input shape:", signal.shape)
    signal = signal.reshape(-1, 1, int(sr*1))
    # print("batched input shape:", signal.shape)
    return signal


if __name__ == '__main__':
    args = parser.parse_args()
    hparams_path = os.path.join(args.exp_dir, "hparams.pickle")
    ckpts = sorted(glob.glob(os.path.join(args.exp_dir, "ckpts", "*")), key=get_val_acc)
    print(ckpts)
    if len(ckpts) == 0:
        print(f"Well, no checkpoints found in {args.exp_dir}. Exiting...")
        exit()
    ckpt_path = ckpts[-1]
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path)

    fname = ckpt_path.split("/")[-3]
    ckpt_ext = "/".join(ckpt_path.split("/")[-3:])
    res = os.path.join(args.exp_dir, "results.txt")
    if os.path.exists(res):
        print(f"{res} files exists.. exiting")
        exit()
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    model = Classifier(hparams.cfg)
    if args.device == "tpu":
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.utils.utils as xu
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device("cuda:0")
    print(model.load_state_dict(checkpoint['model_state_dict']))
    model = model.to(device).eval()
    # print(model)
    ac = hparams.cfg['audio_config']
    print(ac)
    cfg = hparams.cfg
    # val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    audio_config = AudioConfig()
    audio_config.parse_from_config(ac)
    audio_config.cropped_read = False
    audio_config.min_duration = 10
    audio_config.random_clip_size = int(10*audio_config.sr)
    # audio_config.val_feature_size = 1000
    tr_tfs, val_tfs = leaf_raw_supervised_transforms(audio_config)

    sr = ac['sample_rate']
    if args.metrics == "multiclass":
        # if ARGS.use_packed_dataset:
        collate_fn = packed_datasets.packed_collate_fn_multiclass
        # else:
        #     collate_fn = _collate_fn_raw_multiclass
    else:
        # if ARGS.use_packed_dataset:
        collate_fn = packed_datasets.packed_collate_fn_multilabel
        # else:
        #     collate_fn = _collate_fn_raw
    val_set = packed_datasets.PackedDataset(
        manifest_path=os.path.join(args.meta_dir, args.test_csv_name),
        labels_map=os.path.join(args.meta_dir, "lbl_map.json"),
        audio_config=audio_config,
        mode=cfg['model']['type'],
        labels_delimiter=args.separator,
        pre_feature_transforms=val_tfs['pre'],
        post_feature_transforms=val_tfs['post'],
        gcs_bucket_path=args.gcs_bucket_name,
        is_val=True
        # cropped_read=False
    )
    loader = DataLoader(val_set, batch_size=2, num_workers=4, collate_fn=collate_fn)
    all_preds = []
    all_gts = []
    for batch in tqdm.tqdm(loader):
        x, y = batch
        # print(x.shape, y.shape)
        # o = pad_input(x, sr)
        # print(o.shape)
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            # print(preds.shape)
            # preds = torch.mean(preds, dim=0, keepdim=True)
            # print(preds.shape)
            if args.metrics == "multiclass":
                preds = torch.argmax(preds, 1).detach().item()
                all_preds.append(preds)
                all_gts.append(y.detach().cpu().float().item())
            else:
                y_pred_sigmoid = torch.sigmoid(preds)
                all_preds.append(y_pred_sigmoid.detach().cpu().float())
                all_gts.append(y.detach().cpu().float())

    if args.metrics == "multiclass":
        acc = accuracy_score(np.asarray(all_gts), np.asarray(all_preds))
        print("Accuracy: {:.4f}".format(acc))
        with open(res, "w") as fd:
            fd.writelines("model,acc,ckpt_ext\n")
            fd.writelines("{},{},{}\n".format(fname, acc, ckpt_ext))
    else:
        macro_mAP = calculate_mAP(all_preds, all_gts)
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        all_gts = torch.cat(all_gts).detach().cpu().numpy()
        stats = calculate_stats(all_preds, all_gts)
        # mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dp = d_prime(mAUC)
        print("mAP: {:.5f}".format(macro_mAP))
        print("mAUC: {:.5f}".format(mAUC))
        print("dprime: {:.5f}".format(dp))
