import torch
from leaf_pytorch.frontend import Leaf
import numpy as np


if __name__ == '__main__':
    fe = Leaf()
    x = torch.randn(1, 1, 16000)
    x = torch.from_numpy(np.load("/home/sarthak/Workspace/x.npy"))
    x = x.unsqueeze(0)
    print(x.shape)
    o = fe(x)
    print(o.shape)
    print(o[0][1])
