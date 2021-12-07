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
import torch.nn.functional as F
from utilities.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass
from utilities.data.raw_transforms import get_raw_transforms_v2, simple_supervised_transforms
from utilities.config_parser import parse_config, get_data_info, get_config
from models.classifier import Classifier
from utilities.training_utils import setup_dataloaders, optimization_helper
import argparse
from torch.utils.data import DataLoader
from utilities.data.raw_dataset import RawWaveformDataset as SpectrogramDataset
import wandb
from utilities.data.mixup import do_mixup, mixup_criterion
from utilities.metrics_helper import calculate_mAP
from utilities.data.raw_transforms import Compose, PeakNormalization, PadToSize
from sklearn.metrics import accuracy_score
from utilities.metrics_helper import calculate_stats, d_prime

torch.backends.cudnn.enabled = False


def get_val_acc(x):
    x = x.split("/")[-1]
    x = x.replace(".pth","")
    x = x.split("val_acc=")[-1]
    return float(x)


parser = argparse.ArgumentParser()
parser.add_argument("--test_csv_name", type=str)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--meta_dir", type=str)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--metrics", type=str, default="multiclass")
parser.add_argument("--separator", type=str, default=",")


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
    ckpt_path = ckpts[-1]
    print(ckpt_path)
    checkpoint = torch.load(ckpt_path)
    with open(hparams_path, "rb") as fp:
        hparams = pickle.load(fp)
    model = Classifier(hparams.cfg)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(model.load_state_dict(checkpoint['model_state_dict']))
    model = model.to(device).eval()
    print(model)
    ac = hparams.cfg['audio_config']
    print(ac)
    # val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    val_tfs = Compose([
        # PadToSize(val_clip_size, 'wrap'),
        PeakNormalization(sr=ac['sample_rate'])
    ])
    sr = ac['sample_rate']
    # padder = PadToSize(int(sr * 1.0), "wrap")
    # val_tfs = simple_supervised_transforms(False, val_clip_size,
    #                                        sample_rate=ac['sample_rate'])
    val_set = SpectrogramDataset(os.path.join(args.meta_dir, args.test_csv_name),
                                 os.path.join(args.meta_dir, "lbl_map.json"),
                                 hparams.cfg['audio_config'], mode=args.metrics,
                                 transform=val_tfs, is_val=True, delimiter=args.separator
                                 )
    collate_fn = _collate_fn_raw_multiclass if args.metrics == "multiclass" else _collate_fn_raw
    loader = DataLoader(val_set, batch_size=1, num_workers=2, collate_fn=collate_fn)
    all_preds = []
    all_gts = []
    for batch in tqdm.tqdm(loader):
        x, _, y = batch
        # print(x.shape, y.shape)
        o = pad_input(x, sr)
        # print(o.shape)
        o = o.to(device)
        with torch.no_grad():
            preds = model(o)
            # print(preds.shape)
            preds = torch.mean(preds, dim=0, keepdim=True)
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
    else:
        macro_mAP = calculate_mAP(all_preds, all_gts, mode='macro')
        all_preds = torch.cat(all_preds).detach().cpu().numpy()
        all_gts = torch.cat(all_gts).detach().cpu().numpy()
        stats = calculate_stats(all_preds, all_gts)
        # mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dp = d_prime(mAUC)
        print("mAP: {:.5f}".format(macro_mAP))
        print("mAUC: {:.5f}".format(mAUC))
        print("dprime: {:.5f}".format(dp))
