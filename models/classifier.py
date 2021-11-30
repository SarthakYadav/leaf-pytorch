import torch
from torch import nn
from leaf_pytorch import get_frontend
from models.model_helper import get_classifier


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.features = get_frontend(cfg)
        self.model = get_classifier(cfg['model'])

    def forward(self, x):
        out = self.features(x)
        out = out.unsqueeze(1)
        out = self.model(out)
        return out
