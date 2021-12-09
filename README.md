# leaf-pytorch

- [Sponsors](#sponsors)
- [About](#about) 
- [Key Points](#key-points) 
- [Dependencies](#dependencies) 
- [Results](#results) 
- [Loading Pretrained Models](#loading-pretrained-models)
- [References](#references)

## Sponsors

This work would not be possible without cloud resources provided by Google's [TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/). I also thank the TRC support team for quickly resolving whatever issues I had: you're awesome!

## About

This is a PyTorch implementation of the [LEAF audio frontend](https://openreview.net/pdf?id=jM76BCb6F9m) [1], made using the [official tensorflow implementation](https://github.com/google-research/leaf-audio) as a direct reference.  
This implementation supports training on TPUs using `torch-xla`.

## Key Points

* Will be evaluated on AudioSet, SpeechCommands and Voxceleb1 datasets, and pretrained weights will be made available.
* Currently, `torch-xla` has some issues with certain `complex64` operations: `torch.view_as_real(comp)`, `comp.real`, `comp.imag` as highlighted in [#Issue 3070](https://github.com/pytorch/xla/issues/3070). 
These are used primarily for generating gabor impulse responses. To bypass this shortcoming, an alternate implementation using manual complex number operations is provided.
* Matched performance on SpeechCommands, experiments on other datasets ongoing
* More details for commands to replicate experiments will be added shortly


## Dependencies
```
torch >= 1.9.0
torchaudio >= 0.9.0
torch-audiomentations==0.9.0
SoundFile==0.10.3.post1
[Optional] torch_xla == 1.9
```

Additional dependencies include
```
[WavAugment](https://github.com/facebookresearch/WavAugment)
```

## Results
| Model | Dataset | Metric | features | Official | This repo | weights |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|       |       |       |       |       |       |       |
| EfficientNet-b0 | SpeechCommands | Accuracy | LEAF | 93.4±0.3 | 94.5±0.4 | [ckpt](https://drive.google.com/drive/folders/1bPQGE23boXXNkCr2AtDMi8xoZRElpZUK?usp=sharing) 
| EfficientNet-b0 | VoxCeleb1 | Accuracy | LEAF | 33.1±0.7 | 40.9±1.8 | [ckpt](https://drive.google.com/drive/folders/1J4dn6QskJ4YicbCJdti680abIxaAQqTN?usp=sharing)
| ResNet-18 | VoxCeleb1 | Accuracy | LEAF | N/A | 44.7±2.9 | [ckpt](https://drive.google.com/drive/folders/1pWBKaWVDNaI8NusiML91UPHdxTgzP8sd?usp=sharing)


## Loading Pretrained Models

- download and extract desired ckpt from [Results](#results).
```python
import os
import torch
import pickle
from models.classifier import Classifier

results_dir = "<path to results folder>"
hparams_path = os.path.join(results_dir, "hparams.pickle")
ckpt_path = os.path.join(results_dir, "ckpts", "<checkpoint.pth>")
checkpoint = torch.load(ckpt_path)
with open(hparams_path, "rb") as fp:
    hparams = pickle.load(fp)
model = Classifier(hparams.cfg)
print(model.load_state_dict(checkpoint['model_state_dict']))

# to access just the pretrained LEAF frontend
frontend = model.features
```

# References

[1] If you use this repository, kindly cite the LEAF paper:

```
@article{zeghidour2021leaf,
  title={LEAF: A Learnable Frontend for Audio Classification},
  author={Zeghidour, Neil and Teboul, Olivier and de Chaumont Quitry, F{\'e}lix and Tagliasacchi, Marco},
  journal={ICLR},
  year={2021}
}
``` 

Please also consider citing this implementation using the citation widget in the sidebar.
