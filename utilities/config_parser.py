import os
import yaml
from typing import Dict, Optional


def get_data_info(cfg: Dict, augment: Optional[bool] = True) -> Dict:
    try:
        # print("[get_data_info]", cfg)
        meta_root = cfg['meta_root']
        train_manifest = cfg['train_manifest']
        val_manifest = cfg['val_manifest']
        label_map = cfg['label_map']
        is_lmdb = cfg.get("is_lmdb", False)
        if not is_lmdb:
            train_manifest = os.path.join(meta_root, train_manifest)
            val_manifest = os.path.join(meta_root, val_manifest)
            label_map = os.path.join(meta_root, label_map)
            in_memory = cfg.get("in_memory", False)

            results = {
                'train': train_manifest,
                "val": val_manifest,
                "labels": label_map,
                "in_memory": in_memory
            }

            test_manifest = cfg.get("test_manifest", None)
            if test_manifest and test_manifest != "None":
                test_manifest = os.path.join(meta_root, test_manifest)
                results["test"] = test_manifest
            results['bg_files'] = cfg.get("bg_files", None)
            results['background_noise_dir'] = cfg.get("background_noise_dir", None)
        else:
            train_lmdb = cfg['train_lmdb']
            val_lmdb = cfg['val_lmdb']
            label_map = os.path.join(meta_root, label_map) if not os.path.exists(label_map) else label_map
            results = {
                'train': train_lmdb,
                "val": val_lmdb,
                "labels": label_map,
                "is_lmdb": True
            }
            test_lmdb = cfg.get("test_lmdb", None)
            if test_lmdb and test_lmdb != "None":
                results['test_lmdb'] = test_lmdb
            results['background_noise_dir'] = cfg.get("background_noise_dir", None)

        return results

    except KeyError as ex:
        print(ex)
        exit(-1)


__compulsory_keys__ = {
    "frontend": ['name'],
    "model": ["arch", "type"],
    "opt": ["optimizer", "lr", "batch_size"],
    "audio_config": ["feature", "normalize", "sample_rate", "min_duration"],
    "data": ["meta_root", "is_lmdb", "label_map"]
}


__optional_arguments__ = {
    "frontend": {"default_args": False},
    "model": {"activation": "relu"},
    "opt": {
        "scheduler": "step",
        "agc_clip_factor": 0.01,
        "weight_decay": 0.,
        "agc_clipping": True,
        "gamma": 0.1,
        "patience": 15,
        "step_size": 30,
        "warmup_epochs": 15,
    },
    "audio_config": {
        "random_clip_size": 2.5,
        "val_clip_size": 2.5,
        "mixup": False
    },
    "data": {"background_noise_dir": None}
}


def check_and_fill_optional_arguments(cfg):
    # by section
    for k in __compulsory_keys__.keys():
        assert k in cfg.keys()

    for k, v in cfg.items():
        # make sure compulsory keys are present
        assert k in __compulsory_keys__.keys()
        rkeys = __compulsory_keys__[k]
        subkeys = v.keys()
        for rkey in rkeys:
            assert rkey in subkeys, f"{rkey} not found"
        optional_args = __optional_arguments__[k]
        for optk, optv in optional_args.items():
            if optk not in subkeys:
                v[optk] = optv

    if cfg['model']['type'] == "contrastive":
        assert "proj_out_dim" in cfg['model'].keys()
    print(cfg)


# left for compatibility
def parse_config(config_file: str) -> Dict:
    with open(config_file, "r") as fd:
        cfg = yaml.load(fd, yaml.FullLoader)
    return cfg


def get_config(config_file):
    cfg = parse_config(config_file)
    check_and_fill_optional_arguments(cfg)
    cfg['data'] = get_data_info(cfg['data'])
    return cfg
