import torch
import numpy as np


def do_mixup(inputs, targets, alpha=1., random_seed=1233, mode="multilabel"):
    """
    Applies Mixup to the input. Supports doing all computations on the GPU itself.

    :param inputs: input batch tensor of shape (bsize, C, H, W)
    :param targets: input gt tensor of shape (bsize, num_classes)
    :param alpha: alpha parameter for beta distribution generation. can be left as is
    :param random_seed: random seed value
    :return: tuple of mixed_inputs, mixed_outputs
    """
    random_state = np.random.RandomState(random_seed)
    bsize = len(inputs)
    lam = torch.from_numpy(random_state.beta(alpha, alpha, bsize)).to(inputs.device).float()
    perms = torch.randperm(bsize).to(inputs.device)
    if mode == "multilabel":
        mixed_x = inputs * lam.view(bsize, 1, 1) + inputs[perms] * (1 - lam.view(bsize, 1, 1))
        mixed_y = targets * lam.view(bsize, 1) + targets[perms] * (1 - lam.view(bsize, 1))
        return mixed_x, mixed_y, None, None
    else:
        mixed_x = inputs * lam.view(bsize, 1, 1) + inputs[perms] * (1 - lam.view(bsize, 1, 1))
        y_a, y_b = targets, targets[perms]
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (criterion(pred, y_a) * lam + criterion(pred, y_b) * (1 - lam)).mean()
