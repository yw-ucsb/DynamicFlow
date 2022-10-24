import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transforms.base import Transform
from utils import torchutils
from torch import optim
import nn.nets.blockmade as bmade

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


class PiecewiseLinearActivation(nn.Module):
    def __init__(self, feature, num_knot, require_slope=True):
        super().__init__()
        self.feature = feature
        self.num_knot = num_knot
        self.num_spline = num_knot + 1
        self.require_slope = require_slope

        self.x_pos = nn.Parameter(torch.randn(feature, self.num_knot))
        # Slope for each splines, initialized as 1;
        self.slope = nn.Parameter(torch.ones(feature, self.num_spline))
        # Bias of y_pos[0] from x_pos[0], initialized as 0;
        self.y_bias = nn.Parameter(torch.zeros(feature, 1))

        self._epsilon = 1e-3

    def _get_sorted_x_pos(self):
        return torch.sort(self.x_pos)[0]

    def _reset_x_pos(self):
        nn.init.normal_(self.x_pos, std=2.0)

    def _get_y_pos(self):
        self.x_pos_sorted = self._get_sorted_x_pos()
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(self.x_pos_sorted, -1, dims=1)
        delta_x = x_pos_roll - self.x_pos_sorted
        delta_y = delta_x * self.slope[:, 1:]
        tmp = torch.cat((self.x_pos_sorted[:, 0].unsqueeze(-1) + self.y_bias, delta_y[:, :-1]), dim=1)
        y_pos = torch.cumsum(tmp, dim=1)
        return y_pos

    def forward(self, inputs):
        # Constrain the slope to be positive values;
        slope_constrained = F.softplus(self.slope) + self._epsilon

        # Calculate y position of splines;
        x_pos_sorted = self._get_sorted_x_pos()
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(x_pos_sorted, -1, dims=1)
        delta_x = x_pos_roll - x_pos_sorted
        delta_y = delta_x * slope_constrained[:, 1:]
        tmp = torch.cat((x_pos_sorted[:, 0].unsqueeze(-1) + self.y_bias, delta_y[:, :-1]), dim=1)
        y_pos = torch.cumsum(tmp, dim=1)

        # Find the spline id the inputs belong to;
        slope_indices = torchutils.searchsorted(x_pos_sorted, inputs).T
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, 1, slope_indices).T
        x_selected = torch.gather(x_pos_sorted, 1, x_indices).T
        y_selected = torch.gather(y_pos, 1, x_indices).T

        # Calculate the spline output;
        outputs = y_selected + (inputs - x_selected) * slope_selected

        if self.require_slope:
            return outputs, slope_selected
        else:
            return outputs

    def inverse(self, outputs):
        # Constrain the slope to be positive values;
        slope_constrained = F.softplus(self.slope) + self._epsilon

        # Calculate y position of splines;
        x_pos_sorted = self._get_sorted_x_pos()
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(x_pos_sorted, -1, dims=1)
        delta_x = x_pos_roll - x_pos_sorted
        delta_y = delta_x * slope_constrained[:, 1:]
        tmp = torch.cat((x_pos_sorted[:, 0].unsqueeze(-1) + self.y_bias, delta_y[:, :-1]), dim=1)
        y_pos = torch.cumsum(tmp, dim=1)

        # Find the spline id the inputs belong to;
        slope_indices = torchutils.searchsorted(y_pos, outputs).T
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, 1, slope_indices).T
        x_selected = torch.gather(x_pos_sorted, 1, x_indices).T
        y_selected = torch.gather(y_pos, 1, x_indices).T

        # Calculate the spline output;
        inputs = x_selected + (outputs - y_selected) / slope_selected

        if self.require_slope:
            return inputs, 1 / slope_selected
        else:
            return inputs

    def forward_(self, inputs):
        # Constrain the slope to be positive values;
        slope_constrained = F.softplus(self.slope) + self._epsilon

        # Calculate y position of splines;
        x_pos_sorted = self._get_sorted_x_pos()
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(x_pos_sorted, -1, dims=1)
        delta_x = x_pos_roll - x_pos_sorted
        delta_y = delta_x * slope_constrained[:, 1:]
        tmp = torch.cat((x_pos_sorted[:, 0].unsqueeze(-1) + self.y_bias, delta_y[:, :-1]), dim=1)
        y_pos = torch.cumsum(tmp, dim=1)

        # Find the spline id the inputs belong to;
        slope_indices = torchutils.searchsorted(x_pos_sorted, inputs).T
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, 1, slope_indices).T
        x_selected = torch.gather(x_pos_sorted, 1, x_indices).T
        y_selected = torch.gather(y_pos, 1, x_indices).T

        # Calculate the spline output;
        outputs = y_selected + (inputs - x_selected) * slope_selected
        return outputs


class PiecewiseLinearTransformation(Transform):
    def __init__(self, feature, num_knot):
        super().__init__()
        self.pwl = PiecewiseLinearActivation(feature=feature, num_knot=num_knot, require_slope=True)

    def forward(self, inputs, context=None):
        outputs, slope_inputs = self.pwl(inputs)
        logabsdet = torch.sum(torch.log(slope_inputs), dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        outputs, slope_inputs = self.pwl.inverse(inputs)
        logabsdet = torch.sum(torch.log(slope_inputs), dim=1)
        return outputs, logabsdet


class BlockPiecewiseLinear(nn.Module):
    def __init__(self, feature, block_feature, hidden_feature, num_knot, require_slope=True):
        super().__init__()
        self.feature = feature
        self.block_feature = block_feature
        # Calculate number of blocks.
        self.num_block = math.ceil(feature / block_feature)

        self.num_knot = num_knot
        self.num_spline = num_knot + 1
        self.require_slope = require_slope

        # BlockMADE that generates x position and slopes for each spline based on inputs;
        self.BMADE = bmade.BlockMADE(
            in_feature=[self.feature],
            hidden_feature=[hidden_feature],
            out_feature=[self.feature * self.num_knot, self.feature * self.num_spline],
            out_block_feature=[self.block_feature * self.num_knot, self.block_feature * self.num_spline],
            block_feature=[block_feature]
        )

        # Bias of y_pos[0] from x_pos[0], initialized as 0;
        self.y_bias = nn.Parameter(torch.zeros(feature, 1))

        self._epsilon = 1e-3

    def _get_x_pos_slope(self, inputs):
        batch_size = inputs.shape[0]
        output_MADE = self.BMADE(inputs)
        x_pos = output_MADE[..., :self.feature * self.num_knot].reshape(batch_size, self.feature, self.num_knot)
        slope = output_MADE[..., self.feature * self.num_knot:].reshape(batch_size, self.feature, self.num_spline)
        return x_pos, slope

    def _get_sorted_x_pos(self, x_pos):
        return torch.sort(x_pos)[0]

    def forward(self, inputs):
        '''
        Accepting 2-D input as bs * feature, need modification on image dataset;
        '''
        x_pos, slope = self._get_x_pos_slope(inputs)

        # Constrain the slope to be positive values;
        slope_constrained = F.softplus(slope) + self._epsilon

        # Calculate y position of splines;
        x_pos_sorted = self._get_sorted_x_pos(x_pos)
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(x_pos_sorted, -1, dims=-1)
        delta_x = x_pos_roll - x_pos_sorted
        delta_y = delta_x * slope_constrained[..., 1:]
        tmp = torch.cat((x_pos_sorted[..., 0].unsqueeze(-1) + self.y_bias, delta_y[..., :-1]), dim=-1)
        y_pos = torch.cumsum(tmp, dim=-1)

        # Find the spline id the inputs belong to;
        slope_indices = torchutils.searchsorted(x_pos_sorted, inputs)
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, -1, slope_indices.unsqueeze(-1)).squeeze(-1)
        x_selected = torch.gather(x_pos_sorted, -1, x_indices.unsqueeze(-1)).squeeze(-1)
        y_selected = torch.gather(y_pos, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Calculate the spline output;
        outputs = y_selected + (inputs - x_selected) * slope_selected

        if self.require_slope:
            return outputs, slope_selected
        else:
            return outputs

    def inverse(self, outputs):
        pass

    def forward_(self, inputs):
        '''
        Accepting 2-D input as bs * feature, need modification on image dataset;
        '''
        x_pos, slope = self._get_x_pos_slope(inputs)

        # Constrain the slope to be positive values;
        slope_constrained = F.softplus(slope) + self._epsilon

        # Calculate y position of splines;
        x_pos_sorted = self._get_sorted_x_pos(x_pos)
        # Get the tensor x_{i} - x_{i-1}, i = 1, 2, ...n_knot;
        x_pos_roll = torch.roll(x_pos_sorted, -1, dims=-1)
        delta_x = x_pos_roll - x_pos_sorted
        delta_y = delta_x * slope_constrained[..., 1:]
        tmp = torch.cat((x_pos_sorted[..., 0].unsqueeze(-1) + self.y_bias, delta_y[..., :-1]), dim=-1)
        y_pos = torch.cumsum(tmp, dim=-1)

        # Find the spline id the inputs belong to;
        slope_indices = torchutils.searchsorted(x_pos_sorted, inputs)
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, -1, slope_indices.unsqueeze(-1)).squeeze(-1)
        x_selected = torch.gather(x_pos_sorted, -1, x_indices.unsqueeze(-1)).squeeze(-1)
        y_selected = torch.gather(y_pos, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Calculate the spline output;
        outputs = y_selected + (inputs - x_selected) * slope_selected

        return outputs

class BlockPiecewiseLienarTransformation(Transform):
    def __init__(self, feature, block_feature, hidden_feature, num_knot):
        super().__init__()
        self.blockpwl = BlockPiecewiseLinear(
            feature=feature, block_feature=block_feature, hidden_feature=hidden_feature, num_knot=num_knot)

    def forward(self, inputs, context=None):
        outputs, slope_inputs = self.blockpwl(inputs)
        logabsdet = torch.sum(torch.log(slope_inputs), dim=1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        pass

if __name__ == '__main__':
    batch_size = 1
    feature = 7
    block_feature = 3
    hidden_feature = 64
    num_knot = 4

    inputs = torch.randn(batch_size, feature)
    print('Original Input:\n', inputs)

    # f = PiecewiseLinearActivation(feature=feature, num_knot=num_knot)
    f = BlockPiecewiseLinear(feature=feature, block_feature=block_feature, hidden_feature=hidden_feature, num_knot=num_knot)

    y, s_f = f.forward(inputs)

    # x, s_i = f.inverse(y)

    # print(f.x_pos_sorted)

    # print(y)
    # print('Reconstruct:\n', x)

    # y = f.test_f(inputs)


    j = torch.autograd.functional.jacobian(f.forward_, inputs)
    # j = torch.autograd.functional.jacobian(f.test_f, inputs)
    real_j = torch.zeros(size=[batch_size, feature, feature])
    for i in range(batch_size):
        real_j[i, ...] = j[i, :, i, :]
    print(real_j.detach())

    print('Real det:\n', torch.log(torch.det(real_j)))

    print('Estimated det:\n', torch.sum(torch.log(s_f), dim=1))