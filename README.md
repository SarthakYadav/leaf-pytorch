# leaf-pytorch

- [Sponsors](#sponsors)
- [About](#about) 
- [Key Points](#key-points) 
- [Dependencies](#dependencies) 
- [Results](#results) 
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
* Default parameters of `LEAF` are the most thoroughly tested. Will test/add other configurations over time.
* Haven't added `SincNet`, `SincNet+` implementations are not done yet. Might add them in the future depending on availability

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

| Model | Dataset | features | Official | This repo | weights |
| ----- | ----- | ----- | ----- | ----- | ----- |
|       |       |       |       |       |       |
| EfficientNet-b0 | SpeechCommands | LEAF | 93.4±0.3 | 94.52±0.4 | [ckpt](https://drive.google.com/file/d/1E9ZsR4TqGXLdl0mqOFUV7H0qelOCDCVI/view?usp=sharing) 


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
