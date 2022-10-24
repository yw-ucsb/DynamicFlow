'''
Implementation of multi-dimension monotonic sigmoid network;
'''

import math
import numpy as np
import torch
import torch.nn as nn

from transforms.base import Transform
from utils import torchutils
from torch import optim

import torch
from torch.nn import functional as F
import nn.nets.blockmade as bmade
import nn.nets.mlp

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

class BasicLattice(torch.nn.Module):
    def __init__(self, feature, hidden_feature):
        super().__init__()
