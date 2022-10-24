"""
Implementation of piecewise calibrator.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import pwl_calibrator_lib as calib

from transforms.base import Transform
from nn.nets import FlexibleMade
from nn.nets import ResidualNet
from utils import torchutils

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

class UniformCalibrator(Transform):
    def __init__(self, feature, x_min, x_max, n_knots):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.n_knots = n_knots
        self.slope_raw = torch.nn.Parameter(torch.randn(feature, n_knots - 1))

    def forward(self, inputs, context=None):
        # if torch.min(inputs) < torch.min(x_min) or torch.max(inputs) > torch.min(x_max):
        #     raise ValueError('Input outside domain!')

        outputs, derivative = calib._uniform_pwl_calibrator(inputs, self.slope_raw, self.x_min, self.x_max, self.n_knots)

        logabsdet = torch.sum(torch.log(derivative), dim=1)

        return outputs, logabsdet

if __name__ == '__main__':
    bs = 3
    feature = 2
    x_min = torch.tensor([-2., -2.])
    x_max = torch.tensor([2., 2.])
    n_knots = 5

    x = 2 * torch.rand(bs, feature)

    cali = UniformCalibrator(x_min, x_max, n_knots)

    y, logabsdet = cali(x)

    j = torch.autograd.functional.jacobian(cali.forward, x)
    real_j = torch.zeros(size=[bs, feature, feature])
    for i in range(bs):
        real_j[i, ...] = j[0][i, :, i, :]

    real_det = torch.log(torch.det(real_j))