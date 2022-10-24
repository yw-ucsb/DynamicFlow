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

from tqdm import tqdm
import matplotlib.pyplot as plt



class BlockLUAffineTransformation(Transform):
    def __init__(self, feature, block_feature, hidden_feature):
        super().__init__()
        self.feature = feature
        self.block_feature = block_feature
        # Calculate number of blocks.
        self.num_block = math.ceil(feature / block_feature)
        self.block_feature_last = self.feature - (self.num_block - 1) * self.block_feature

        # Number of triangular elements;
        self.n_triangular_entries_expect_last = (self.block_feature * (self.block_feature - 1)) // 2
        self.n_triangular_entries_last = (self.block_feature_last * (self.block_feature_last - 1)) // 2
        # MADE network only parameterize entries on off-diagonal positions;
        self.MADE_out_feature = (self.num_block - 1) * self.n_triangular_entries_expect_last +\
                                self.n_triangular_entries_last

        # Indices of lower and upper triangular matrices
        self.lower_indices = np.tril_indices(feature, k=-1)
        self.upper_indices = np.triu_indices(feature, k=1)
        self.diag_indices = np.diag_indices(feature)

        # BMADEs to generate lower and upper triangular matrices;
        self.LMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.MADE_out_feature,
            num_block=self.num_block
        )
        self.UMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.MADE_out_feature,
            num_block=self.num_block
        )
        self.DiagMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.feature,
            num_block=self.num_block
        )
        self.BiasMADE = bmade.BlockMADE(
            in_feature=self.feature,
            hidden_feature=hidden_feature,
            out_feature=self.feature,
            num_block=self.num_block
        )

        self._epsilon = 1e-3

    def _create_LUbias(self, inputs):
        batch_size = inputs.shape[0]
        L = torch.zeros(batch_size, self.feature, self.feature)
        U = torch.zeros(batch_size, self.feature, self.feature)
        self.diag = torch.zeros(batch_size, self.feature)

        # Get matrices indices;
        L_indices, U_indices, D_indices = self._create_LUD_indices()

        # Get matrices entries from BMADEs;
        l_flatten = self.LMADE(inputs)
        u_flatten = self.UMADE(inputs)
        diag_unconstrained = self.DiagMADE(inputs)
        bias = self.BiasMADE(inputs)

        # Constrained diagonal elements to be positive;
        self.diag = F.softplus(diag_unconstrained) + self._epsilon

        # Fill corresponding matrices;
        L[:, L_indices[0], L_indices[1]] = l_flatten
        L[:, D_indices[0], D_indices[1]] = 1.0
        U[:, U_indices[0], U_indices[1]] = u_flatten
        U[:, D_indices[0], D_indices[1]] = self.diag

        return L, U, bias

    def _create_LUD_indices(self):
        if self.block_feature == 1:
            print('Warning, you are creating autoregressive flow.')
        index_bias_last = (self.num_block - 1) * self.block_feature
        L_indices_expect_last = torch.tril_indices(self.block_feature, self.block_feature, -1)
        U_indices_expect_last = torch.triu_indices(self.block_feature, self.block_feature, 1)
        L_indices_last = torch.tril_indices(self.block_feature_last, self.block_feature_last, -1)
        U_indices_last = torch.triu_indices(self.block_feature_last, self.block_feature_last, 1)

        L_indices  = torch.empty(2, 0, dtype=int)
        U_indices  = torch.empty(2, 0, dtype=int)
        D_indices = np.diag_indices(self.feature)
        for i in range(self.num_block - 1):
            L_indices = torch.cat((L_indices, (L_indices_expect_last + i * self.block_feature)), dim=1)
            U_indices = torch.cat((U_indices, (U_indices_expect_last + i * self.block_feature)), dim=1)
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
        logabsdet = torch.sum(torch.log(self.diag), dim=1)
        outputs = torch.einsum('bij, bj->bi', U, inputs)
        outputs = torch.einsum('bij, bj->bi', L, outputs) + bias
        return outputs


# Testing code;
if __name__ == '__main__':
    batch_size = 2
    in_feature = 6
    hidden_feature = 256
    block_feature = 2

    inputs = torch.randn(batch_size, in_feature)

    num_iter = 5000
    val_interval = 250
    lr = 0.001

    LUNet = BlockLUAffineTransformation(
        feature=in_feature,
        block_feature=block_feature,
        hidden_feature=hidden_feature
    )

    a, b, c = LUNet._create_LUD_indices()
    # print(a, b, c)

    L, U, bias = LUNet._create_LUbias(inputs)
    print(L)
    print(U)

    _, logabsdet = LUNet(inputs)

    # Print out the real Jacobian matrix, should observe diagonal block pattern;
    # j = torch.autograd.functional.jacobian(BlockNet.forward_, inputs)
    j = torch.autograd.functional.jacobian(LUNet.forward_, inputs)
    # j = torch.autograd.functional.jacobian(block_made.forward, inputs)
    real_j = torch.zeros(size=[batch_size, in_feature, in_feature])
    for i in range(batch_size):
        real_j[i, ...] = j[i, :, i, :]
    print(real_j.detach())

    # print('Real det:', torch.log(torch.abs(torch.det(real_j))))
    print('Real det:', torch.log(torch.det(real_j)))

    print('Estimated Jacobian is:', torch.sum(torch.log(LUNet.diag), dim=1))
