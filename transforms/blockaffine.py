'''
Implementation of blocked affine transformation with BlockMADE.
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


class BlockLUAffineTransformation(Transform):
    def __init__(self,
                 feature,
                 hidden_feature,
                 block_feature,
                 num_hidden_layer=2,
                 use_res_layer=True,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False
                 ):
        super().__init__()
        self.feature = feature
        self.b_feature_bias = block_feature
        self.num_block = math.ceil(feature / block_feature)
        # Calculate number of blocks.
        self.b_feature_bias_last = self.feature - (self.num_block - 1) * self.b_feature_bias

        # Number of triangular elements;
        self.b_feature_tri = (self.b_feature_bias * (self.b_feature_bias - 1)) // 2
        self.b_feature_tri_last = (self.b_feature_bias_last * (self.b_feature_bias_last - 1)) // 2
        # MADE network only parameterize entries on off-diagonal positions;
        self.MADE_out_feature = (self.num_block - 1) * self.b_feature_tri + self.b_feature_tri_last

        # BMADEs to generate lower and upper triangular matrices;
        # The sequence is: L off diagonal, U off-diagonal, U diagonal, bias;
        self.BMADE = bmade.BlockMADE(
            in_feature=[self.feature],
            hidden_feature=[hidden_feature],
            out_feature=[self.MADE_out_feature, self.MADE_out_feature, self.feature, self.feature],
            out_block_feature=[self.b_feature_tri, self.b_feature_tri, self.b_feature_bias, self.b_feature_bias],
            block_feature=[block_feature],
            num_hidden_layer=num_hidden_layer,
            use_res_layer=use_res_layer,
            rand_mask=rand_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

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
        diag_unconstrained = output_MADE[..., (2 * self.MADE_out_feature):(2 * self.MADE_out_feature + self.feature)]
        bias = output_MADE[..., (2 * self.MADE_out_feature + self.feature):]

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

    def forward(self, inputs, context=None):
        L, U, bias = self._create_LUbias(inputs)
        logabsdet = torch.sum(torch.log(self.diag), dim=1)
        outputs = torch.einsum('bij, bj->bi', U, inputs)
        outputs = torch.einsum('bij, bj->bi', L, outputs) + bias
        return outputs, logabsdet

    def forward_(self, inputs, context=None):
        L, U, bias = self._create_LUbias(inputs)
        outputs = torch.einsum('bij, bj->bi', U, inputs)
        outputs = torch.einsum('bij, bj->bi', L, outputs) + bias
        return outputs



# Testing code;
if __name__ == '__main__':
    batch_size = 2
    in_feature = 6
    hidden_feature = 256
    block_feature = 3

    inputs = torch.randn(batch_size, in_feature)

    num_iter = 5000
    val_interval = 250
    lr = 0.001


    LUNet = BlockLUAffineTransformation(
        feature=in_feature,
        hidden_feature=hidden_feature,
        block_feature=block_feature
    )

    a, b, c = LUNet._create_LUD_indices()
    # print(a, b, c)

    L, U, bias = LUNet._create_LUbias(inputs)
    diag = LUNet.diag
    # print(L)
    # print(U)

    _, logabsdet = LUNet(inputs)

    # Print out the real Jacobian matrix, should observe diagonal block pattern;
    j = torch.autograd.functional.jacobian(LUNet.forward_, inputs)
    real_j = torch.zeros(size=[batch_size, in_feature, in_feature])
    for i in range(batch_size):
        real_j[i, ...] = j[i, :, i, :]
    print(real_j.detach())

    # print('Real det:', torch.log(torch.abs(torch.det(real_j))))
    print('Real det:', torch.log(torch.det(real_j)))

    # print('Estimated Jacobian is:', torch.sum(torch.log(LUNet.diag), dim=1))
    print('Estimated Jacobian is:', logabsdet)