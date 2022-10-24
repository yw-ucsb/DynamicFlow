'''
Implementation of blocked invertible mapping with BlockMADE.
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
import nn.nde.made as made

from tqdm import tqdm
import matplotlib.pyplot as plt

class BlockInvertibleTransformation(Transform):
    def __init__(self,
                 feature,
                 hidden_feature,
                 block_feature,
                 num_knot=4,
                 num_hidden_layer=2,
                 use_res_layer=True,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False):
        super().__init__()

        self.feature = feature
        self.b_feature_bias = block_feature
        self.block_feature = block_feature
        self.num_block = math.ceil(feature / block_feature)
        # Calculate number of blocks.
        self.b_feature_bias_last = self.feature - (self.num_block - 1) * self.b_feature_bias

        # Number of triangular elements;
        self.b_feature_tri = (self.b_feature_bias * (self.b_feature_bias - 1)) // 2
        self.b_feature_tri_last = (self.b_feature_bias_last * (self.b_feature_bias_last - 1)) // 2
        # MADE network only parameterize entries on off-diagonal positions;
        self.MADE_out_feature = (self.num_block - 1) * self.b_feature_tri + self.b_feature_tri_last

        # Activation layer setup;
        self.num_knot = num_knot
        self.num_spline = num_knot + 1

        # BMADEs to generate lower and upper triangular matrices, as well spline knots and slopes;
        # The sequence is: L off diagonal, U off-diagonal, U diagonal, bias, spline x position, spline slope;
        self.BMADE = bmade.BlockMADE(
            in_feature=[self.feature],
            hidden_feature=[hidden_feature],
            out_feature=[self.MADE_out_feature, self.MADE_out_feature, self.feature,
                         self.feature, self.feature * self.num_knot, self.feature * self.num_spline],
            out_block_feature=[self.b_feature_tri, self.b_feature_tri, self.b_feature_bias,
                               self.b_feature_bias, self.block_feature * self.num_knot, self.block_feature * self.num_spline],
            block_feature=[block_feature],
            num_hidden_layer=num_hidden_layer,
            use_res_layer=use_res_layer,
            rand_mask=rand_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

        # Bias of y_pos[0] from x_pos[0], initialized as 0;
        self.y_bias = nn.Parameter(torch.zeros(feature, 1))

        self.diag = None
        self._epsilon = 1e-3

    def _create_LUbias(self, inputs):
        batch_size = inputs.shape[0]
        output_MADE = self.BMADE(inputs)
        L = torch.zeros(batch_size, self.feature, self.feature).to(inputs.device)
        U = torch.zeros(batch_size, self.feature, self.feature).to(inputs.device)

        # Get matrices indices;
        L_indices, U_indices, D_indices = self._create_LUD_indices()

        # Get matrices entries from BMADEs;
        l_flatten = output_MADE[..., :self.MADE_out_feature]
        u_flatten = output_MADE[..., self.MADE_out_feature:(2 * self.MADE_out_feature)]
        diag_unconstrained = output_MADE[...,
                             (2 * self.MADE_out_feature):(2 * self.MADE_out_feature + self.feature)]
        bias = output_MADE[..., (2 * self.MADE_out_feature + self.feature):2 * (self.MADE_out_feature + self.feature)]

        # Constrained diagonal elements to be positive;
        self.diag = F.softplus(diag_unconstrained) + self._epsilon

        # Fill corresponding matrices;
        L[:, L_indices[0], L_indices[1]] = l_flatten
        L[:, D_indices[0], D_indices[1]] = 1.0
        U[:, U_indices[0], U_indices[1]] = u_flatten
        U[:, D_indices[0], D_indices[1]] = self.diag

        return L, U, bias

    def _create_LUD_indices(self):
        index_bias_last = (self.num_block - 1) * self.b_feature_bias
        L_indices_expect_last = torch.tril_indices(self.b_feature_bias, self.b_feature_bias, -1)
        U_indices_expect_last = torch.triu_indices(self.b_feature_bias, self.b_feature_bias, 1)
        L_indices_last = torch.tril_indices(self.b_feature_bias_last, self.b_feature_bias_last, -1)
        U_indices_last = torch.triu_indices(self.b_feature_bias_last, self.b_feature_bias_last, 1)

        L_indices = torch.empty(2, 0, dtype=torch.int)
        U_indices = torch.empty(2, 0, dtype=torch.int)
        D_indices = np.diag_indices(self.feature)
        for i in range(self.num_block - 1):
            L_indices = torch.cat((L_indices, (L_indices_expect_last + i * self.b_feature_bias)), dim=1)
            U_indices = torch.cat((U_indices, (U_indices_expect_last + i * self.b_feature_bias)), dim=1)
        L_indices = torch.cat((L_indices, (L_indices_last + index_bias_last)), dim=1)
        U_indices = torch.cat((U_indices, (U_indices_last + index_bias_last)), dim=1)
        return L_indices, U_indices, D_indices

    def _get_x_pos_slope(self, inputs):
        batch_size = inputs.shape[0]
        output_MADE = self.BMADE(inputs)
        x_pos = output_MADE[..., 2 * (self.MADE_out_feature + self.feature):2 * (self.MADE_out_feature + self.feature) + self.feature * self.num_knot].reshape(batch_size, self.feature, self.num_knot)
        slope = output_MADE[..., 2 * (self.MADE_out_feature + self.feature) + self.feature * self.num_knot:].reshape(batch_size, self.feature, self.num_spline)
        return x_pos, slope

    def _get_sorted_x_pos(self, x_pos):
        return torch.sort(x_pos)[0]

    def forward(self, inputs, context=None):
        # Affine layer;
        L, U, bias = self._create_LUbias(inputs)
        logabsdet = torch.sum(torch.log(self.diag), dim=1)
        affine_outputs = torch.einsum('bij, bj->bi', U, inputs)
        affine_outputs = torch.einsum('bij, bj->bi', L, affine_outputs) + bias

        # Activation layer;
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
        slope_indices = torchutils.searchsorted(x_pos_sorted, affine_outputs)
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, -1, slope_indices.unsqueeze(-1)).squeeze(-1)
        x_selected = torch.gather(x_pos_sorted, -1, x_indices.unsqueeze(-1)).squeeze(-1)
        y_selected = torch.gather(y_pos, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Calculate the spline output;
        outputs = y_selected + (affine_outputs - x_selected) * slope_selected

        logabsdet += torch.sum(torch.log(slope_selected), dim=1)

        return outputs, logabsdet

    def forward_(self, inputs, context=None):
        # Affine layer;
        L, U, bias = self._create_LUbias(inputs)
        logabsdet = torch.sum(torch.log(self.diag), dim=1)
        affine_outputs = torch.einsum('bij, bj->bi', U, inputs)
        affine_outputs = torch.einsum('bij, bj->bi', L, affine_outputs) + bias

        # Activation layer;
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
        slope_indices = torchutils.searchsorted(x_pos_sorted, affine_outputs)
        x_indices = F.relu((slope_indices - 1))

        # Select the slope and left x boundary with spline id;
        slope_selected = torch.gather(slope_constrained, -1, slope_indices.unsqueeze(-1)).squeeze(-1)
        x_selected = torch.gather(x_pos_sorted, -1, x_indices.unsqueeze(-1)).squeeze(-1)
        y_selected = torch.gather(y_pos, -1, x_indices.unsqueeze(-1)).squeeze(-1)

        # Calculate the spline output;
        outputs = y_selected + (affine_outputs - x_selected) * slope_selected

        return outputs

# Testing code;
if __name__ == '__main__':
    batch_size = 2
    in_feature = 7
    hidden_feature = 256
    block_feature = 3
    num_knot = 4

    inputs = torch.randn(batch_size, in_feature)

    num_iter = 5000
    val_interval = 250
    lr = 0.001


    net = BlockInvertibleTransformation(
        feature=in_feature,
        hidden_feature=hidden_feature,
        block_feature=block_feature,
        num_knot=num_knot
    )

    a, b, c = net._create_LUD_indices()
    # print(a, b, c)

    L, U, bias = net._create_LUbias(inputs)
    diag = net.diag
    # print(L)
    # print(U)

    _, logabsdet = net(inputs)

    # Print out the real Jacobian matrix, should observe diagonal block pattern;
    j = torch.autograd.functional.jacobian(net.forward_, inputs)
    real_j = torch.zeros(size=[batch_size, in_feature, in_feature])
    for i in range(batch_size):
        real_j[i, ...] = j[i, :, i, :]
    print(real_j.detach())

    # print('Real det:', torch.log(torch.abs(torch.det(real_j))))
    print('Real det:', torch.log(torch.det(real_j)))

    # print('Estimated Jacobian is:', torch.sum(torch.log(LUNet.diag), dim=1))
    print('Estimated Jacobian is:', logabsdet)