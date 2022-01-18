import glob
import tqdm
import torch
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, help="Experiment directory containing `ckpts` folder")
parser.add_argument("--topk", type=int, default=5, help="Top n ckpts to keep")


def get_metrics_from_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path)
    metrics = {
        'tr_loss': ckpt['tr_loss'],
        # 'val_loss': ckpt['val_loss'],
        'tr_acc': ckpt['tr_acc'],
        'val_acc': ckpt['val_acc'],
    }
    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.exp_dir, "ckpts", "*.pth"))
    records = []
    for f in tqdm.tqdm(files):
        rec = {
            'filename': f
        }
        metrics = get_metrics_from_ckpt(f)
        for k,v in metrics.items():
            rec[k] = v
        records.append(rec)
    
    df = pd.DataFrame(records)
    print(df)
    df = df.sort_values(["val_acc"], ascending=False)
    df.to_csv(os.path.join(args.exp_dir, "sorted_metrics.csv"), index=False)
    print(df)

    df = df[:args.topk]
    print(df)

    to_keep_files = df['filename'].values.tolist()
    deleted_files_count = 0
    for f in tqdm.tqdm(files):
        if f not in to_keep_files:
            os.remove(f)
            deleted_files_count += 1
    print("Total files: {} | topk: {} | Files removed: {}".format(len(files), args.topk, deleted_files_count))
