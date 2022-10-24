"""
Implementation of lattice network;
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import lattice_lib as llib
import lattice_lib_coupling as llib_c

from transforms.base import Transform
from nn.nets import FlexibleMade
from nn.nets import ResidualNet
from utils import torchutils

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


class BasicDenseLattice(Transform):
    def __init__(self, feature, lattice_size=None):
        super().__init__()
        self.feature = feature

        if lattice_size is not None:
            self.lattice_size = lattice_size
        else:
            self.lattice_size = [[2 for i in range(j + 1)] for j in range(feature)]

        self.theta_size = sum([math.prod(self.lattice_size[i]) for i in range(len(self.lattice_size))])

        self.theta_raw = torch.nn.Parameter(torch.rand(self.theta_size))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        theta_raw = self.theta_raw.repeat(batch_size, 1)

        outputs, derivative = llib._get_interpolation_and_derivative(inputs, theta_raw, self.lattice_size)

        logabsdet = torch.sum(torch.log(derivative), dim=1)

        return outputs, logabsdet

    def _get_current_monotonic_theta(self):
        theta_mono = []
        # Calculates the split size.
        split_size = [math.prod(self.lattice_size[i]) for i in range(len(self.lattice_size))]
        theta_split = torch.split(self.theta_raw.unsqueeze(0), split_size, dim=1)

        for i in range(self.feature):
            theta_mono.append(torch.sigmoid(llib._create_monotonic_theta_one_d(theta_split[i], self.lattice_size[i])))
        return theta_mono


class AutoregressiveDenseLattice(Transform):
    def __init__(self, feature, hidden_feature, lattice_size=None):
        super().__init__()
        self.feature = feature

        if lattice_size is not None:
            self.lattice_size = lattice_size
        else:
            self.lattice_size = [[2 for i in range(j + 1)] for j in range(feature)]

        self.out_degree_multiplier = [math.prod(self.lattice_size[i]) for i in range(len(self.lattice_size))]

        self.autoregressive_net = FlexibleMade.FlexibleMADE(
            feature=feature,
            hidden_feature=hidden_feature,
            out_degree_multiplier=self.out_degree_multiplier
        )

    def _get_theta_raw(self, inputs):
        return self.autoregressive_net(inputs)

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        theta_raw = self._get_theta_raw(inputs)

        outputs, derivative = llib._get_interpolation_and_derivative(inputs, theta_raw, self.lattice_size)

        logabsdet = torch.sum(torch.log(derivative), dim=1)

        return outputs, logabsdet


class BasicCouplingLattice(Transform):
    def __init__(self, lattice_size, prior_index=None, random_x=False, theta_constraint=True):
        super().__init__()
        self.lattice_size = lattice_size
        self.prior_index = prior_index
        self.random_x = random_x
        self.theta_constraint = theta_constraint

        # Initialize theta_raw for basic lattice.
        self.theta_raw = torch.nn.Parameter(torch.randn(math.prod(self.lattice_size)))

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        theta_raw = self.theta_raw.repeat(batch_size, 1)
        dim_dataset = inputs.shape[1]
        midpoint = dim_dataset // 2 if dim_dataset % 2 == 0 else dim_dataset // 2 + 1

        outputs = torch.empty_like(inputs)

        # Split the inputs to identity group, prior group and transformation group
        x_identity, x_prior, x_transform = llib_c._split_x(inputs, self.lattice_size, self.prior_index, self.random_x)

        y, derivative = llib_c._get_interpolation_derivative(x_prior, x_transform, theta_raw, self.lattice_size)

        outputs[:, 0:midpoint, ...] = x_identity
        outputs[:, midpoint:dim_dataset, ...] = y

        logasbdet = torch.sum(torch.log(derivative), dim=1)

        return outputs, logasbdet


class CouplingLattice(Transform):
    def __init__(self, feature, hidden_feature, lattice_size, prior_index, random_x=False):
        super().__init__()
        self.feature = feature
        self.midpoint = feature // 2 if feature % 2 == 0 else feature // 2 + 1
        self.lattice_size = lattice_size
        self.prior_index = prior_index
        self.random_x = random_x

        # Parameterization with ResidualNet.
        self.out_feature = math.prod(lattice_size)
        self.net = ResidualNet(in_features=self.midpoint,
                               out_features=self.out_feature,
                               hidden_features=hidden_feature,
                               num_blocks=2,
                               activation=F.relu,
                               dropout_probability=0.,
                               use_batch_norm=False
                               )

    def forward(self, inputs, context=None):
        # Split the inputs to identity group, prior group and transformation group.
        x_identity, x_prior, x_transform = llib_c._split_x(inputs, self.lattice_size, self.prior_index, self.random_x)

        # Generates theta_raw and do interpolation.
        theta_raw = self.net(x_identity)
        y, derivative = llib_c._get_interpolation_derivative(x_prior, x_transform, theta_raw, self.lattice_size)

        outputs = torch.empty_like(inputs)
        outputs[:, 0:self.midpoint, ...] = x_identity
        outputs[:, self.midpoint:self.feature, ...] = y

        logasbdet = torch.sum(torch.log(derivative), dim=1)

        return outputs, logasbdet


if __name__ == '__main__':
    # bs = 20
    # lattice_size = [[10], [10, 10]]
    #
    # x = torch.rand(bs, len(lattice_size))

    # # Test of BasicDenseLattice.
    #
    # lattice = BasicDenseLattice(len(lattice_size), lattice_size)
    #
    # j = torch.autograd.functional.jacobian(lattice.forward, x)
    # real_j = torch.zeros(size=[bs, len(lattice_size), len(lattice_size)])
    # for i in range(bs):
    #     real_j[i, ...] = j[0][i, :, i, :]
    #
    # y, logabsdet = lattice(x)
    #
    # real_det = torch.log(torch.det(real_j))

    # # Test of AutoregressiveDenseLattice.
    #
    # a_lattice = AutoregressiveDenseLattice(len(lattice_size), 64, lattice_size)
    #
    # j = torch.autograd.functional.jacobian(a_lattice.forward, x)
    # real_j = torch.zeros(size=[bs, len(lattice_size), len(lattice_size)])
    # for i in range(bs):
    #     real_j[i, ...] = j[0][i, :, i, :]
    #
    # y, logabsdet = a_lattice(x)
    #
    # real_det = torch.log(torch.det(real_j))
    
    # Test of BasicCouplingLattice.

    bs = 20
    lattice_size = [11, 11]

    x = torch.rand(bs, len(lattice_size))
    
    c_lattice = BasicCouplingLattice(lattice_size=lattice_size, prior_index=[0, 1], random_x=False)

    j = torch.autograd.functional.jacobian(c_lattice.forward, x)
    real_j = torch.zeros(size=[bs, len(lattice_size), len(lattice_size)])
    for i in range(bs):
        real_j[i, ...] = j[0][i, :, i, :]

    y, logabsdet = c_lattice(x)

    real_det = torch.log(torch.det(real_j))