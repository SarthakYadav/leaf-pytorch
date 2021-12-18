import os
import datetime
import copy
import pickle
from threading import main_thread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.data import packed_dataset
from utilities.data.utils import _collate_fn_raw, _collate_fn_raw_multiclass
from utilities.data.raw_transforms import get_raw_transforms_v2, simple_supervised_transforms
from utilities.config_parser import parse_config, get_data_info, get_config
from models.classifier import Classifier
from utilities.training_utils import setup_dataloaders, optimization_helper
import argparse
from utilities.data.raw_dataset import RawWaveformDataset as SpectrogramDataset
import wandb
from utilities.data.mixup import do_mixup, mixup_criterion
from utilities.metrics_helper import calculate_mAP


def save_checkpoint(model, optimizer, scheduler, epoch,
                    tr_loss, tr_acc, val_acc):
    archive = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "tr_loss": tr_loss,
        "tr_acc": tr_acc,
        "val_acc": val_acc
    }
    ckpt_path = os.path.join(ARGS.output_directory,
                             "epoch={:03d}_tr_loss={:.6f}_tr_acc={:.6f}_val_acc={:.6f}.pth".format(
                                 epoch, tr_loss, tr_acc, val_acc
                             ))
    torch.save(archive, ckpt_path)
    print("Checkpoint written to -> {}".format(ckpt_path))


parser = argparse.ArgumentParser()
parser.description = "Training script"
parser.add_argument("--cfg_file", type=str,
                    help='path to cfg file')
parser.add_argument("--gpu_id", type=int, help="gpu index", default=0)
parser.add_argument("--expdir", "-e", type=str,
                    help="directory for logging and checkpointing")
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cw", type=str, required=False,
                    help="path to serialized torch tensor containing class weights")
parser.add_argument("--resume_from", type=str,
                    help="checkpoint path to continue training from")
parser.add_argument('--mixer_prob', type=float, default=0.75,
                    help="background noise augmentation probability")
parser.add_argument("--fp16", action="store_true",
                    help='flag to train in FP16 mode')
parser.add_argument("--random_clip_size", type=float, default=5)
parser.add_argument("--val_clip_size", type=float, default=5)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--log_steps", default=10, type=int)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--high_aug", action="store_true")
parser.add_argument("--wandb_project", type=str, default="leaf-pytorch")
parser.add_argument("--wandb_group", type=str, default="dataset")
parser.add_argument("--labels_delimiter", type=str, default=",")
parser.add_argument("--wandb_watch_model", action="store_true")
parser.add_argument("--random_seed", type=int, default=8881)
parser.add_argument("--continue_from_ckpt", type=str, default=None)
parser.add_argument("--cropped_read", action="store_true")
parser.add_argument("--use_packed_dataset", action="store_true")


ARGS = parser.parse_args()
ARGS.output_directory = os.path.join(ARGS.expdir, "ckpts")
ARGS.log_directory = os.path.join(ARGS.expdir, "logs")


def load_checkpoint(ckpt_path, model, optimizer, scheduler):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch']


def train(ARGS):
    np.random.seed(ARGS.random_seed)
    torch.manual_seed(ARGS.random_seed)
    cfg = get_config(ARGS.cfg_file)

    mode = cfg['model']['type']
    # world_size = xm.xrt_world_size()
    # local_rank = xm.get_ordinal()
    # random_clip_size = int(ARGS.random_clip_size * cfg['audio_config']['sample_rate'])
    # val_clip_size = int(ARGS.val_clip_size * cfg['audio_config']['sample_rate'])
    ac = cfg['audio_config']
    random_clip_size = int(ac['random_clip_size'] * ac['sample_rate'])
    val_clip_size = int(ac['val_clip_size'] * ac['sample_rate'])
    if ARGS.high_aug:
        tr_tfs = get_raw_transforms_v2(True, random_clip_size,
                                       sample_rate=ac['sample_rate'])
        val_tfs = get_raw_transforms_v2(False, val_clip_size, center_crop_val=True,
                                        sample_rate=ac['sample_rate'])
    else:
        tr_tfs = simple_supervised_transforms(True, random_clip_size,
                                              sample_rate=ac['sample_rate'])
        val_tfs = simple_supervised_transforms(False, val_clip_size,
                                               sample_rate=ac['sample_rate'])
    if ARGS.use_packed_dataset:
        train_set = packed_dataset.PackedDataset(cfg['data']['train'],
                                                 cfg['data']['labels'],
                                                 cfg['audio_config'],
                                                 mode=mode, augment=True,
                                                 mixer=None, delimiter=ARGS.labels_delimiter,
                                                 transform=tr_tfs, is_val=False,
                                                 cropped_read=ARGS.cropped_read)
        val_set = packed_dataset.PackedDataset(cfg['data']['val'],
                                               cfg['data']['labels'],
                                               cfg['audio_config'],
                                               mode=mode, augment=False,
                                               mixer=None, delimiter=ARGS.labels_delimiter,
                                               transform=val_tfs, is_val=True)
    else:
        train_set = SpectrogramDataset(cfg['data']['train'],
                                       cfg['data']['labels'],
                                       cfg['audio_config'],
                                       mode=mode, augment=True,
                                       mixer=None, delimiter=ARGS.labels_delimiter,
                                       transform=tr_tfs, is_val=False, cropped_read=ARGS.cropped_read)

        val_set = SpectrogramDataset(cfg['data']['val'],
                                     cfg['data']['labels'],
                                     cfg['audio_config'],
                                     mode=mode, augment=False,
                                     mixer=None, delimiter=ARGS.labels_delimiter,
                                     transform=val_tfs, is_val=True)

    batch_size = cfg['opt']['batch_size']

    # device = xm.xla_device()
    device = torch.device(f"cuda:{ARGS.gpu_id}")
    # model = model_helper(cfg['model']).to(device)
    model = Classifier(cfg).to(device)
    if mode == "multiclass":
        if ARGS.use_packed_dataset:
            collate_fn = packed_dataset.packed_collate_fn_raw_multiclass
        else:
            collate_fn = _collate_fn_raw_multiclass
    else:
        if ARGS.use_packed_dataset:
            collate_fn = packed_dataset.packed_collate_fn_raw_multilabel
        else:
            collate_fn = _collate_fn_raw

    train_loader, val_loader = setup_dataloaders(train_set, val_set,
                                                 batch_size=batch_size, collate_fn=collate_fn,
                                                 num_workers=ARGS.num_workers)
    # train_device_loader = pl.MpDeviceLoader(train_loader, device)
    # val_device_loader = pl.MpDeviceLoader(val_loader, device)
    num_steps_per_epoch = len(train_loader)
    optimizer, scheduler, scheduler_name = optimization_helper(model.parameters(), cfg, ARGS.devices,
                                                               reduce_on_plateau_mode="max",
                                                               num_tr_steps_per_epoch=num_steps_per_epoch,
                                                               num_epochs=ARGS.epochs)
    if ARGS.continue_from_ckpt:
        print("Attempting to load checkpoint {}".format(ARGS.continue_from_ckpt))
        start_epoch = load_checkpoint(ARGS.continue_from_ckpt, model, optimizer, scheduler)
        print("Checkpoint loading successful.. Continuing training from Epoch {}".format(start_epoch))
    else:
        start_epoch = 1
    writer = None
    wandb_logger = None
    # if xm.is_master_ordinal():
    if not os.path.exists(ARGS.output_directory):
        os.makedirs(ARGS.output_directory)

    if not os.path.exists(ARGS.log_directory):
        os.makedirs(ARGS.log_directory)
    log_name = ARGS.log_directory.split("/")[-2]
    print("RUN NAME:", log_name)
    if not ARGS.no_wandb:
        wandb_logger = wandb.init(project='{}'.format(ARGS.wandb_project),
                                  group="{}".format(ARGS.wandb_group),
                                  config=cfg, name=log_name)
    print(model)
    with open(os.path.join(ARGS.expdir, "hparams.pickle"), "wb") as handle:
        args_to_save = copy.deepcopy(ARGS)
        args_to_save.cfg = cfg
        pickle.dump(args_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if mode == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
    elif mode == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()

    mixup_enabled = cfg["audio_config"].get("mixup", False)  # and mode == "multilabel"
    if mixup_enabled:
        print("Attention: Will use mixup while training..")

    torch.set_grad_enabled(True)
    if wandb_logger and ARGS.wandb_watch_model:
        wandb_logger.watch(model, log="all", log_freq=100)

    agc_clip = bool(cfg['opt'].get("agc_clipping", False))
    accuracy, max_accuracy = 0.0, 0.0
    end_epoch = ARGS.epochs
    for epoch in range(start_epoch, end_epoch+1):
        print("Epoch {:03d} train begin {}".format(epoch, str(datetime.datetime.now())))
        tr_step_counter = 0
        model.train()
        tr_loss = []
        tr_correct = 0
        tr_total_samples = 0

        tr_preds = []
        tr_gts = []

        for batch in train_loader:
            x, _, y = batch
            x = x.to(device)
            y = y.to(device)
            if mixup_enabled:
                if mode == "multilabel":
                    x, y, _, _ = do_mixup(x, y, mode=mode)
                elif mode == "multiclass":
                    x, y_a, y_b, lam = do_mixup(x, y, mode=mode)
            pred = model(x)
            if mode == "multiclass":
                pred_labels = pred.max(1, keepdim=True)[1]
                tr_correct += pred_labels.eq(y.view_as(pred_labels)).sum()
                tr_total_samples += x.size(0)
                if mixup_enabled:
                    loss = mixup_criterion(loss_fn, pred, y_a, y_b, lam)
                else:
                    loss = loss_fn(pred, y)
            else:
                y_pred_sigmoid = torch.sigmoid(pred)
                tr_preds.append(y_pred_sigmoid.detach().cpu().float())
                tr_gts.append(y.detach().cpu().float())
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if tr_step_counter % ARGS.log_steps == 0:
                print(f"Epoch: {epoch:03d}/{end_epoch:03d} Step:[{tr_step_counter:04d}]/[{num_steps_per_epoch:04d}] Loss: {loss:.4f}")
            tr_loss.append(loss.item())
            tr_step_counter += 1
            if scheduler_name == "warmupcosine":
                scheduler.step()
        epoch_tr_loss = np.mean(tr_loss)
        # epoch_tr_loss = xm.mesh_reduce("tr_loss", mean_tr_loss, np.mean)
        if mode == "multiclass":
            tr_acc = tr_correct.item() / tr_total_samples
        else:
            # calculate mAP
            tr_acc = calculate_mAP(tr_preds, tr_gts, mixup_enabled, mode="weighted")

        # tr_acc = xm.mesh_reduce("train_accuracy", tr_acc, np.mean)
        print('Epoch {} train end {} | Mean Loss: {} | Mean Acc:{}'.format(epoch,
                                                                           str(datetime.datetime.now()),
                                                                           epoch_tr_loss,
                                                                           tr_acc))
        val_step_counter = 0
        model.eval()
        total_samples = 0
        correct = 0
        del tr_gts, tr_preds
        curr_lr = scheduler.get_lr()
        print("Validating..")
        val_preds = []
        val_gts = []
        for batch in val_loader:
            x, _, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                pred = model(x)
                # xm.master_print("pred.shape:", pred.shape)
            if mode == "multiclass":
                pred = pred.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum()
                total_samples += x.size()[0]
            else:
                y_pred_sigmoid = torch.sigmoid(pred)
                val_preds.append(y_pred_sigmoid.detach().cpu().float())
                val_gts.append(y.detach().cpu().float())
        if mode == "multiclass":
            accuracy = correct.item() / total_samples
            # accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        else:
            accuracy = calculate_mAP(val_preds, val_gts)
            # val_preds = torch.cat(val_preds, 0)
            # val_gts = torch.cat(val_gts, 0)
            # all_val_preds = xm.mesh_reduce("all_val_preds", val_preds, torch.cat)
            # xm.master_print("after all reduce, preds shape:", all_val_preds.shape)

        print('Epoch {} test end {}, Accuracy={:.4f}'.format(epoch, str(datetime.datetime.now()), accuracy))
        max_accuracy = max(accuracy, max_accuracy)
        dict_to_write = {
            "tr_loss": epoch_tr_loss,
            "tr_acc": tr_acc,
            "val_acc": accuracy
        }
        del val_gts, val_preds
        if wandb_logger:
            wandb_logger.log(dict_to_write)
        save_checkpoint(model, optimizer, scheduler, epoch, epoch_tr_loss, tr_acc, accuracy)
        if scheduler_name == "reduce":
            scheduler.step(tr_acc)
        else:
            scheduler.step()

    print("Training done, best acc: {}".format(max_accuracy))
    if wandb_logger:
        wandb_logger.finish()
    return max_accuracy


# def _mp_fn(index, flags):
#     # torch.set_default_tensor_type("torch.FloatTensor")
#     acc = train(flags)


if __name__ == "__main__":
    # xmp.spawn(_mp_fn, args=(ARGS,), nprocs=ARGS.tpus)
    # _mp_fn()
    acc = train(ARGS)
