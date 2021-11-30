# credits: https://github.com/hche11
# https://github.com/hche11/VGGSound/blob/master/utils.py

import csv
import torch
from sklearn.metrics import average_precision_score
from scipy import stats
from sklearn import metrics
import numpy as np


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def reverseTransform(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if len(img.shape) == 5:
        for i in range(3):
            img[:, i, :, :, :] = img[:, i, :, :, :]*std[i] + mean[i]
    else:
        for i in range(3):
            img[:, i, :, :] = img[:, i, :, :]*std[i] + mean[i]
    return img


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def calculate_mAP(preds, gts, mixup=False, mode="macro"):
    preds = torch.cat(preds, 0).numpy()
    gts = torch.cat(gts, 0).numpy()
    if mixup:
        gts[gts >= 0.5] = 1
        gts[gts < 0.5] = 0
    map_value = average_precision_score(gts, preds, average=mode)
    return map_value
