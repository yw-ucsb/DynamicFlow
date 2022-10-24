"""
Implementation of lattice network;
"""

import math
import numpy as np
import torch
import copy
import collections

from transforms.base import Transform
from utils import torchutils

import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

def _partial_mono_projection(w, lattice_size, monotonicity, dim, constraint_group):
    layers = list(torch.unbind(w, dim=dim))
    for i in range(constraint_group, lattice_size[dim] - 1, 2):
        average = (layers[i] + layers[i + 1]) / 2.0
        if monotonicity == 1:
            layers[i] = torch.minimum(layers[i], average)
            layers[i + 1] = torch.maximum(layers[i + 1], average)

    return torch.stack(tuple(layers), dim=dim)

def _dykstra_projection(w, lattice_size, monotonicity, num_iteration):
    # TODO: Modify the code such that it supports batched weights;
    # TODO: Verify the difference between two projection methods;
    # TODO: Constraint in Tensorflow seems very different from Pytorch, check
    # TODO: how to modify the code;
    if num_iteration == 0:
        return w

    def _dykstra_loop(w, last_change):
        last_change = copy.copy(last_change)
        for dim in range(len(lattice_size)):
            for constraint_group in [0, 1]:
                if constraint_group + 1 >= lattice_size[dim]:
                    continue
                w_roll_back = w - last_change[(dim, constraint_group)]
                w = _partial_mono_projection(w_roll_back, lattice_size, monotonicity, dim, constraint_group)
                last_change[(dim, constraint_group)] = w - w_roll_back
        return w, last_change

    # This part is used to generate the keys for last_change tuple;
    zeros = torch.zeros(lattice_size)
    last_change = collections.defaultdict(lambda: zeros)
    _, last_change = _dykstra_loop(w, last_change)

    # Major Dykstra projection loop:
    last_change = {k: zeros for k in last_change}
    for i in range(num_iteration):
        w, last_change = _dykstra_loop(w, last_change)
        # print(w)
    return w

class Basic2DLattice(torch.nn.Module):
    def __init__(self, bond, n, batch_size, eps=1e-3):
        """
        v is a free parameter
        Member:
        bond: input range of x, 1-D tensor;
        n: number of segments per dimension;
        d: length of each mini-square;
        """
        super().__init__()
        self.bond = bond
        self.n = n
        self.d = (bond[1] - bond[0])/n
        self.batch_size = batch_size
        self.eps = eps

        self.v_raw = torch.nn.Parameter(torch.randn(batch_size, 2 * (n + 1) ** 2))

    def _getParameters(self):
        return self.v_raw
        # raise NotImplementedError

    def _assignVertices(self, v_raw, monotonicity):
        """
        Step 1: add a minimal distance between adjacent vertices' values to guarantee
        strict monotonicity in mini-squares;

        Step 2: assign vertices values for the whole square based on given monotonicity;
        1: monotonically increasing, 0: monotonically decreasing;

        Note: more advanced implementation should be done with binary division
        currently, only 2-D cases with [1, 1] and [1, 0] is considered;
        """

        batch_size = v_raw.shape[0]
        v_sort = torch.sort(v_raw)[0]
        v = v_sort.reshape(batch_size, self.n + 1, self.n + 1)
        # Construct the stable matrix for v;
        tmp1, tmp2 = torch.meshgrid(torch.linspace(0, self.eps * self.n, self.n + 1), torch.linspace(0, self.eps * self.n, self.n + 1))
        m_stable = (tmp1 + tmp2).to(v_raw.device)
        v_stable = v + m_stable

        if monotonicity == [1, 1]:
            return v_stable
        elif monotonicity == [1, 0]:
            return torch.flip(v_stable, dims=[1])
        else:
            raise ValueError('Not supported yet!')

    def _getInterpolationParameter(self, v_all, x):
        """
        Given batched input x, determine which mini-square each x is in and return the
        corresponding interpolation parameters for each x;
        Input:
        v_all: vertices of the whole square, batch_size * (n + 1)^2;
        x: batch_size * 2;
        Output:
        v: vertices for each x, batch_size * 4;
        b: boundary for each x, batch_size * 2;

        This is a naive version of implementation, think about more efficient code later;
        """
        batch_size = x.shape[0]
        block_feature = x.shape[1]
        # Boundary tensor for comparison, batch_size * 2 * (n + 1), 2 is the block feature size;
        r_bond = torch.arange(self.bond[0], self.bond[1], self.d)
        x_id = torch.sum((r_bond <= x.unsqueeze(-1)).long(), dim=-1) - 1

        # Gather v and b values based on input id;
        r = torch.arange(batch_size)
        v0 = v_all[r, x_id[:, 1]][r, x_id[:, 0]].unsqueeze(-1)
        v1 = v_all[r, x_id[:, 1] + 1][r, x_id[:, 0]].unsqueeze(-1)
        v2 = v_all[r, x_id[:, 1]][r, x_id[:, 0] + 1].unsqueeze(-1)
        v3 = v_all[r, x_id[:, 1] + 1][r, x_id[:, 0] + 1].unsqueeze(-1)

        v = torch.cat((v0, v1, v2, v3), dim=1)
        b = r_bond[x_id]

        return v, b

    def _2DInterpolation(self, v, b, x):
        """
        2D Interpolation within a single square;
        Input:
        v: vertex value: batch_size * 4, v0  v2
                                         v1  v3;
        b: vertex coordinates of v0: [b0_x0, b0_x1], bs * 2;
        x: batch_size * 2; vertical: x1, horizontal: x0;
        Output:
        y: batch_size;
        """
        b0 = b
        b1 = b0 + torch.tensor([0., self.d], device=x.device)
        b2 = b0 + torch.tensor([self.d, 0.], device=x.device)
        b3 = b0 + torch.tensor([self.d, self.d], device=x.device)

        s0 = (b3[:, 0] - x[:, 0]) * (b3[:, 1] - x[:, 1])
        s1 = (b2[:, 0] - x[:, 0]) * (x[:, 1] - b2[:, 1])
        s2 = (x[:, 0] - b1[:, 0]) * (b1[:, 1] - x[:, 1])
        s3 = (x[:, 0] - b0[:, 0]) * (x[:, 1] - b0[:, 1])

        y = s0 * v[:, 0] + s1 * v[:, 1] + s2 * v[:, 2] + s3 * v[:, 3]

        dx0 = (v[:, 2] - v[:, 0]) * (b1[:, 1] - x[:, 1]) + (v[:, 3] - v[:, 1]) * (x[:, 1] - b0[:, 1])
        dx1 = (v[:, 1] - v[:, 0]) * (b2[:, 0] - x[:, 0]) + (v[:, 3] - v[:, 2]) * (x[:, 0] - b0[:, 0])

        return y, dx0, dx1

    def _2DSingleBlockInterpolation(self, v0, v1, bond, x):
        """
        2D interpolation for both dimension within a single square;
        x0 and x1 use same bound value but are interpolated with different v;
        """
        batch_size = x.shape[0]
        J = torch.zeros(batch_size, 2, 2)
        y = torch.zeros(batch_size, 2)
        y[:, 0], a, b = self._2DInterpolation(v0, bond, x)
        y[:, 1], c, d = self._2DInterpolation(v1, bond, x)

        logabsdet = torch.log(a * d - b * c)
        return y, logabsdet

    def forward(self, x):
        v_raw = self._getParameters()
        v0_raw, v1_raw = v_raw.chunk(2, dim=1)

        v0_stable = self._assignVertices(v0_raw, [1, 0])
        v1_stable = self._assignVertices(v1_raw, [1, 1])
        v0, b0 = self._getInterpolationParameter(v0_stable, x)
        v1, b1 = self._getInterpolationParameter(v1_stable, x)
        assert torch.equal(b0, b1)
        y, logabsdet = self._2DSingleBlockInterpolation(v0, v1, b0, x)
        return y, logabsdet



class Basci2DLatticeTransformation(Transform):
    def __init__(self, bond, n, batch_size, eps=1e-6):
        """
        v is a free parameter
        Member:
        bond: input range of x, 1-D tensor;
        n: number of segments per dimension;
        d: length of each mini-square;
        """
        super().__init__()
        self.bond = bond
        self.n = n
        self.d = (bond[1] - bond[0]) / n
        self.batch_size = batch_size
        self.eps = eps

        self.v_raw = torch.nn.Parameter(torch.randn(2 * (n + 1) ** 2))
        # self.v_raw = torch.nn.Parameter(torch.tensor([0., 1., 0., 1., 0., 0., 1., 1.]))
        # self.v_raw = torch.tensor([0, 0, 1, 1]).repeat(batch_size, 2)

    def _getParameters(self):
        v0_raw, v1_raw = self.v_raw.chunk(2)
        v0_raw = v0_raw.reshape(self.n + 1, self.n + 1)
        v1_raw = v1_raw.reshape(self.n + 1, self.n + 1)
        v0_project = _dykstra_projection(v0_raw, [self.n + 1, self.n + 1], 1, 5)
        v1_project = _dykstra_projection(v1_raw, [self.n + 1, self.n + 1], 1, 5)
        v0_project = torch.flip(v0_project, dims=[0])
        return v0_project, v1_project
        # return self.v_raw.repeat(self.batch_size, 1)
        # raise NotImplementedError

    def _assignVertices(self, v_raw, monotonicity):
        """
        Step 1: add a minimal distance between adjacent vertices' values to guarantee
        strict monotonicity in mini-squares;

        Step 2: assign vertices values for the whole square based on given monotonicity;
        1: monotonically increasing, 0: monotonically decreasing;

        Note: more advanced implementation should be done with binary division
        currently, only 2-D cases with [1, 1] and [1, 0] is considered;
        """

        batch_size = v_raw.shape[0]
        v_sort = torch.sort(v_raw)[0]
        v = v_sort.reshape(batch_size, self.n + 1, self.n + 1)
        # # Construct the stable matrix for v;
        # tmp1, tmp2 = torch.meshgrid(torch.linspace(0, self.eps * self.n, self.n + 1),
        #                             torch.linspace(0, self.eps * self.n, self.n + 1))
        # m_stable = (tmp1 + tmp2).to(v_raw.device)
        # v_stable = v + m_stable
        v_stable = v

        if monotonicity == [1, 1]:
            return v_stable
        elif monotonicity == [1, 0]:
            return torch.flip(torch.transpose(v_stable, dim0=1, dim1=2), dims=[1])
        else:
            raise ValueError('Not supported yet!')

    def _getInterpolationParameter(self, v_all, x):
        """
        Given batched input x, determine which mini-square each x is in and return the
        corresponding interpolation parameters for each x;
        Input:
        v_all: vertices of the whole square, batch_size * (n + 1)^2;
        x: batch_size * 2;
        Output:
        v: vertices for each x, batch_size * 4;
        b: boundary for each x, batch_size * 2;

        This is a naive version of implementation, think about more efficient code later;
        """
        batch_size = x.shape[0]
        block_feature = x.shape[1]
        # Boundary tensor for comparison, batch_size * 2 * (n + 1), 2 is the block feature size;
        r_bond = torch.arange(self.bond[0], self.bond[1], self.d)
        x_id = torch.sum((r_bond <= x.unsqueeze(-1)).long(), dim=-1) - 1

        # Gather v and b values based on input id;
        r = torch.arange(batch_size)
        v0 = v_all[r, x_id[:, 1]][r, x_id[:, 0]].unsqueeze(-1)
        v1 = v_all[r, x_id[:, 1] + 1][r, x_id[:, 0]].unsqueeze(-1)
        v2 = v_all[r, x_id[:, 1]][r, x_id[:, 0] + 1].unsqueeze(-1)
        v3 = v_all[r, x_id[:, 1] + 1][r, x_id[:, 0] + 1].unsqueeze(-1)

        v = torch.cat((v0, v1, v2, v3), dim=1)
        b = r_bond[x_id]

        return v, b

    def _2DInterpolation(self, v, b, x):
        """
        2D Interpolation within a single square;
        Input:
        v: vertex value: batch_size * 4, v0  v2
                                         v1  v3;
        b: vertex coordinates of v0: [b0_x0, b0_x1], bs * 2;
        x: batch_size * 2; vertical: x1, horizontal: x0;
        Output:
        y: batch_size;
        """
        b0 = b
        b1 = b0 + torch.tensor([0., self.d], device=x.device)
        b2 = b0 + torch.tensor([self.d, 0.], device=x.device)
        b3 = b0 + torch.tensor([self.d, self.d], device=x.device)

        s0 = (b3[:, 0] - x[:, 0]) * (b3[:, 1] - x[:, 1])
        s1 = (b2[:, 0] - x[:, 0]) * (x[:, 1] - b2[:, 1])
        s2 = (x[:, 0] - b1[:, 0]) * (b1[:, 1] - x[:, 1])
        s3 = (x[:, 0] - b0[:, 0]) * (x[:, 1] - b0[:, 1])

        y = (s0 * v[:, 0] + s1 * v[:, 1] + s2 * v[:, 2] + s3 * v[:, 3]) / self.d ** 2

        dx0 = ((v[:, 2] - v[:, 0]) * (b1[:, 1] - x[:, 1]) + (v[:, 3] - v[:, 1]) * (x[:, 1] - b0[:, 1]) ) / self.d ** 2
        dx1 = ((v[:, 1] - v[:, 0]) * (b2[:, 0] - x[:, 0]) + (v[:, 3] - v[:, 2]) * (x[:, 0] - b0[:, 0]) ) / self.d ** 2

        return y, dx0, dx1

    def _2DSingleBlockInterpolation(self, v0, v1, bond, x):
        """
        2D interpolation for both dimension within a single square;
        x0 and x1 use same bound value but are interpolated with different v;
        """
        batch_size = x.shape[0]
        J = torch.zeros(batch_size, 2, 2)
        y = torch.zeros(batch_size, 2)
        y[:, 0], a, b = self._2DInterpolation(v0, bond, x)
        y[:, 1], c, d = self._2DInterpolation(v1, bond, x)

        # J = torch.cat((a,b,c,d), dim=0).reshape(batch_size, 2, 2)
        # print('Jacobian is:\n', J)
        #
        # print(torch.log(torch.det(J)))

        logabsdet = torch.log(torch.abs(a * d - b * c) + self.eps)
        return y, logabsdet

    def forward(self, inputs, context=None):
        # v_raw = self._getParameters()
        # v0_raw, v1_raw = v_raw.chunk(2, dim=1)
        # v0_stable = torch.sigmoid(self._assignVertices(v0_raw, [1, 0]))
        # v1_stable = torch.sigmoid(self._assignVertices(v1_raw, [1, 1]))


        v0_raw, v1_raw = self._getParameters()
        v0_stable = torch.sigmoid(v0_raw.repeat(self.batch_size, 1, 1))
        v1_stable = torch.sigmoid(v1_raw.repeat(self.batch_size, 1, 1))
        v0, b0 = self._getInterpolationParameter(v0_stable, inputs)
        v1, b1 = self._getInterpolationParameter(v1_stable, inputs)
        assert torch.equal(b0, b1)
        # print('v0:', v0_stable)
        # print('v1:', v1_stable)
        outputs, logabsdet = self._2DSingleBlockInterpolation(v0, v1, b0, inputs)
        return outputs, logabsdet


class Test2DLattice (Transform):
    def __init__(self, n):
        super(Test2DLattice, self).__init__()
        self.n = n
        self.v0 = torch.nn.Parameter(torch.randn(2, 2))
        self.v1 = torch.nn.Parameter(torch.randn(2, 2))

    def forward(self, inputs, context=None):
        x = inputs
        bs = x.shape[0]
        y = torch.zeros(bs, 2)
        J = torch.zeros(bs, 2, 2)
        # First project the vertices;
        s_v0 = torch.tensor([[0, 0], [1e-4, 1e-4]])
        s_v1 = torch.tensor([[0, 1e-4], [0, 1e-4]])
        v0 = _dykstra_projection(self.v0, [self.n + 1, self.n + 1], 1, 5) + s_v0
        v1 = _dykstra_projection(self.v1, [self.n + 1, self.n + 1], 1, 5) + s_v1
        v0 = torch.sigmoid(torch.flip(v0, dims=[1]))
        v1 = torch.sigmoid(v1)

        # Calculates the interpolation weights;
        s0 = (1 - x[:, 0]) * (1 - x[:, 1])
        s1 = x[:, 0] * (1 - x[:, 1])
        s2 = (1 - x[:, 0]) * x[:, 1]
        s3 = x[:, 0] * x[:, 1]

        y[:, 0] = s0 * v0[0, 0] + s1 * v0[1, 0] + s2 * v0[0, 1] + s3 * v0[1, 1]
        y[:, 1] = s0 * v1[0, 0] + s1 * v1[1, 0] + s2 * v1[0, 1] + s3 * v1[1, 1]

        J[:, 0, 0] = (v0[1, 0] - v0[0, 0]) * (1 - x[:, 1]) + (v0[1, 1] - v0[0, 1]) * x[:, 1]
        J[:, 0, 1] = (v0[0, 1] - v0[0, 0]) * (1 - x[:, 0]) + (v0[1, 1] - v0[1, 0]) * x[:, 0]
        J[:, 1, 0] = (v1[1, 0] - v1[0, 0]) * (1 - x[:, 1]) + (v1[1, 1] - v1[0, 1]) * x[:, 1]
        J[:, 1, 1] = (v1[0, 1] - v1[0, 0]) * (1 - x[:, 0]) + (v1[1, 1] - v1[1, 0]) * x[:, 0]

        # print('J is:', J)

        return y, torch.log(torch.abs(torch.det(J)))


if __name__ == '__main__':
    # np.random.seed(1137)
    # torch.manual_seed(114514)

    # n = 1
    # bs = 2
    # bond = torch.tensor([0, 1])
    # x = torch.rand(bs, 2)
    #
    # lattice = Basci2DLatticeTransformation(bond, n, bs)
    #
    # j = torch.autograd.functional.jacobian(lattice.forward, x)
    # real_j = torch.zeros(size=[bs, 2, 2])
    # for i in range(bs):
    #     real_j[i, ...] = j[0][i, :, i, :]
    #
    #
    # y, logabsdet = lattice(x)
    #
    # err = torch.log(torch.det(real_j)) - logabsdet


    test2dlattice = Test2DLattice(1)
    bs = 2
    bond = torch.tensor([0, 1])
    x = torch.rand(bs, 2)

    j = torch.autograd.functional.jacobian(test2dlattice.forward, x)
    real_j = torch.zeros(size=[bs, 2, 2])
    for i in range(bs):
        real_j[i, ...] = j[0][i, :, i, :]

    y, logabsdet = test2dlattice(x)






