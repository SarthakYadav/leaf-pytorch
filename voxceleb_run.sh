#!/bin/bash

python train_xla.py --cfg_file ./cfgs/voxceleb/efficientnet-b0-leaf-default.cfg -e ~/leaf_experiments/voxceleb/efficientnet-b0_default_leaf_bs1x256_adam_wd_1e-4_rs8881_legacycomplex --epochs 50 --num_workers 4 --log_steps 50 --random_seed 8881 --wandb_group voxceleb
python train_xla.py --cfg_file ./cfgs/voxceleb/efficientnet-b0-leaf-default.cfg -e ~/leaf_experiments/voxceleb/efficientnet-b0_default_leaf_bs1x256_adam_wd_1e-4_rs8882_legacycomplex --epochs 50 --num_workers 4 --log_steps 50 --random_seed 8882 --wandb_group voxceleb
python train_xla.py --cfg_file ./cfgs/voxceleb/efficientnet-b0-leaf-default.cfg -e ~/leaf_experiments/voxceleb/efficientnet-b0_default_leaf_bs1x256_adam_wd_1e-4_rs8883_legacycomplex --epochs 50 --num_workers 4 --log_steps 50 --random_seed 8883 --wandb_group voxceleb
