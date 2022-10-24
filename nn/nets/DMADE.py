'''
Implementation of MADE considering the state action pair as input.
Input:
    S = [s', s, a]
'''

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions, nn
from torch.nn import functional as F
from torch.nn import init

from utils import torchutils

torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)

def _get_in_degree(dim_s, dim_a):
    """
    Function to assign the degrees to the inputs;
    :param s: Tensor with shape: [bs, dim_s]
    :param a: Tensor with shape: [bs, dim_a]
    :param dim_s: Int value, dimension of the state
    :param dim_a: Int value, dimension of the action
    :return: Tensor with value from [0, dim_s], the degree of the inputs
    """
    degree = torch.cat((torch.arange(1, dim_s + 1), torch.zeros(dim_s), torch.zeros(dim_a)))
    return degree

def _get_out_degree(dim_s, dim_next, next_layer='hidden', random_mask=False):
    """
    Function to assign degree to the current layer
    :param dim_s: Int value, dimension of the state
    :param dim_next: Int value, dimension of the next layer
    :param next_layer: String of 'hidden' or 'out'
    :param random_mask: Boolean value
    :return: Tensor of size [dim_next]
    """
    if next_layer == 'hidden':
        if random_mask:
            out_degree = torch.randint(
                low=0,
                high=dim_s - 1,
                size=[dim_next],
                dtype=torch.long,
            )
            return out_degree
        else:
            n = math.ceil(dim_next / dim_s)
            out_degree = torch.arange(0, dim_s).repeat(n)
            return out_degree[0:dim_next]
    elif next_layer == 'out':
        return torch.arange(1, dim_s + 1)

class MaskedLinear(nn.Linear):
    def __init__(self,
                 dim_s,
                 in_degree,
                 dim_out,
                 next_layer='hidden',
                 random_mask=False,
                 bias=True):
        super().__init__(
            in_features=len(in_degree), out_features=dim_out, bias=bias
        )
        self.dim_s = dim_s
        self.in_degree = in_degree
        self.dim_out = dim_out
        self.next_layer = next_layer
        self.random_mask = random_mask

        mask, out_degree = self._create_mask()
        self.register_buffer('mask', mask)
        self.register_buffer('out_degree', out_degree)

        # print('In:', self.in_degree)
        # print('Out:', self.out_degree)
        # print('mask:', self.mask)

    def _create_mask(self):
        if self.next_layer == 'out':
            out_degree = _get_out_degree(dim_s=self.dim_s, dim_next=self.dim_out, next_layer=self.next_layer,
                                         random_mask=self.random_mask)
            mask = (out_degree[..., None] > self.in_degree).float()
        elif self.next_layer == 'hidden':
            out_degree = _get_out_degree(dim_s=self.dim_s, dim_next=self.dim_out, next_layer=self.next_layer,
                                         random_mask=self.random_mask)
            mask = (out_degree[..., None] >= self.in_degree).float()
        else:
            raise ValueError
        return mask, out_degree

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

class MaskedResidualBlock(nn.Module):
    def __init__(self,
                 dim_s,
                 in_degree,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=False,
                 ):
        if random_mask:
            raise ValueError('Masked Residual block does not support random degree.')
        super().__init__()
        dim = len(in_degree)

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(dim, eps=1e-3) for _ in range(2)]
            )

        # Initialize linear blocks.
        linear_0 = MaskedLinear(
            dim_s=dim_s,
            in_degree=in_degree,
            dim_out=dim,
            next_layer='hidden',
            random_mask=False
        )
        linear_1 = MaskedLinear(
            dim_s=dim_s,
            in_degree=linear_0.out_degree,
            dim_out=dim,
            next_layer='hidden',
            random_mask=False
        )
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        self.out_degree = linear_1.out_degree

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization of weights.
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, a=-1e-3, b=1e-3)
            init.uniform_(self.linear_layers[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, inputs):
        residual = inputs
        if self.use_batch_norm:
            residual = self.batch_norm_layers[0](residual)
        residual = self.activation(residual)
        residual = self.linear_layers[0](residual)
        if self.use_batch_norm:
            residual = self.batch_norm_layers[1](residual)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.linear_layers[1](residual)
        return inputs + residual

class DMADE(nn.Module):
    def __init__(self,
                 dim_s,
                 dim_a,
                 dim_hidden,
                 random_mask=False,
                 num_blocks=0,
                 use_residual_blocks=True,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 ):
        super().__init__()
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random degree.")
        super().__init__()

        # First layer.
        self.first_layer = MaskedLinear(
            dim_s=dim_s,
            in_degree=_get_in_degree(dim_s, dim_a),
            dim_out=dim_hidden,
            random_mask=random_mask
        )

        prev_out_degree = self.first_layer.out_degree

        # Hidden layers.
        blocks = []
        block_constructor = MaskedResidualBlock

        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    dim_s=dim_s,
                    in_degree=prev_out_degree,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    zero_initialization=True,
                )
            )
            prev_out_degree = blocks[-1].out_degree
        self.blocks = nn.ModuleList(blocks)

        # Last Layer:
        self.last_layer = MaskedLinear(
            dim_s=dim_s,
            in_degree=prev_out_degree,
            dim_out=dim_s,
            next_layer='out'
        )

    # def forward(self, s_next, s, a):
    #     inputs = torch.cat((s_next, s, a), dim=1)
    #     tmp = self.first_layer(inputs)
    #     for block in self.blocks:
    #         tmp = block(tmp)
    #     outputs = self.last_layer(tmp)
    #     return outputs

    def forward(self, inputs):
        tmp = self.first_layer(inputs)
        for block in self.blocks:
            tmp = block(tmp)
        outputs = self.last_layer(tmp)
        return outputs


if __name__ == '__main__':
    dim_s = 3
    dim_a = 2
    bs = 1

    s_next = torch.randn(bs, dim_s)
    s = torch.randn(bs, dim_s)
    a = torch.randn(bs, dim_a)

    dmade = DMADE(dim_s=dim_s, dim_a=dim_a, dim_hidden=16)

    x = torch.cat((s_next, s, a), dim=1)
    y = dmade(x)

    j = torch.autograd.functional.jacobian(dmade.forward, x)
    real_j = torch.zeros(size=[bs, dim_s, dim_s * 2 + dim_a])
    for i in range(bs):
        real_j[i, ...] = j[i, :, i, :]


    print(real_j)
