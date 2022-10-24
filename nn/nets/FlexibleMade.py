'''
Implementation of flexible made network, which allows different output feature
for different output degrees.
'''

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions, nn
from torch.nn import functional as F
from torch.nn import init

from utils import torchutils

def _get_input_degree(in_dim):
    return torch.arange(1, in_dim + 1)

def _get_degree(feature, out_dim, next_layer_type='hidden', out_degree_multiplier=None, random_degree=False):
    """
    This function returns the degree of the output neurons of a layer.
    Args:
        feature: Int value, the dimension of dataset
        out_dim: Int value, output dimension
        next_layer_type: String with value 'hidden' or 'out'
        out_degree_multiplier: None or List of d integers with value
        greater or equal to 1
        random_degree: Boolean value, whether to apply random degree to
        hidden layers
    Return:
        out_degree: Tensor of size:
        [out_dim] if 'hidden'
        [sum(out_degree_multiplier)] if 'out'
    """
    if next_layer_type == 'hidden':
        if random_degree:
            out_degree = torch.randint(
                    low=1,
                    high=feature - 1,
                    size=[out_dim],
                    dtype=torch.long,
                )
            return out_degree
        else:
            n = math.ceil(out_dim / (feature - 1))
            out_degree = torch.arange(1, feature).repeat(n)
            return out_degree[0:out_dim]
    elif next_layer_type == 'out':
        if out_degree_multiplier is None:
            return torch.arange(1, feature + 1)
        else:
            if len(out_degree_multiplier) != feature:
                raise ValueError('Size of output degree multiplier incorrect, expect {} but get {}'.format(feature,
                                  len(out_degree_multiplier)))
            tmp = []
            for i in range(feature):
                out_degree_i = torch.tensor([i+1]).repeat(out_degree_multiplier[i])
                tmp.append(out_degree_i)
            out_degree = torch.cat(tmp)
            return out_degree
    else:
        raise ValueError('Invalid layer type, should be hidden or out.')


class MaskedLinear(nn.Linear):
    def __init__(self,
                 feature,
                 in_degree,
                 out_dim,
                 next_layer_type='hidden',
                 out_degree_multiplier=None,
                 random_degree=False,
                 bias=True):
        super().__init__(
            in_features=len(in_degree), out_features=out_dim, bias=bias
        )
        self.feature = feature
        self.in_degree = in_degree
        self.out_dim = out_dim
        self.next_layer_type = next_layer_type
        self.out_degree_multiplier = out_degree_multiplier
        self.random_degree = random_degree

        mask, out_degree = self._get_mask_and_degree()
        self.register_buffer('mask', mask)
        self.register_buffer('out_degree', out_degree)

    def _get_mask_and_degree(self):
        # Calculates the output degree and weight mask of the current layer.
        if self.next_layer_type == 'out':
            out_degree = _get_degree(self.feature, self.out_dim, self.next_layer_type,
                                     self.out_degree_multiplier, self.random_degree)
            mask = (out_degree[..., None] > self.in_degree).float()

        else:
            out_degree = _get_degree(self.feature, self.out_dim, self.next_layer_type,
                                     self.out_degree_multiplier, self.random_degree)
            mask = (out_degree[..., None] >= self.in_degree).float()
        return mask, out_degree

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedForwardBlock(nn.Module):
    """
    Hidden blocks of MADE, the input and output dimensions are the same.
    """
    def __init__(self,
                 feature,
                 in_degree,
                 random_degree=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=False,
                 ):
        super().__init__()
        dim = len(in_degree)

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim, eps=1e-3)
        else:
            self.batch_norm = None

        self.linear = MaskedLinear(
            feature=feature,
            in_degree=in_degree,
            out_dim=dim,
            next_layer_type='hidden',
            random_degree=random_degree
        )
        self.out_degree = self.linear.out_degree

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs, context=None):
        if self.batch_norm:
            outputs = self.batch_norm(inputs)
        else:
            outputs = inputs
        outputs = self.mask_linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MaskedResidualBlock(nn.Module):
    def __init__(self,
                 feature,
                 in_degree,
                 random_degree=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=False,
                 ):
        if random_degree:
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
            feature=feature,
            in_degree=in_degree,
            out_dim=dim,
            next_layer_type='hidden',
            random_degree=False
        )
        linear_1 = MaskedLinear(
            feature=feature,
            in_degree=linear_0.out_degree,
            out_dim=dim,
            next_layer_type='hidden',
            random_degree=False
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


class FlexibleMADE(nn.Module):
    """
    Implementation of FlexibleMADE which allows arbitrary output features for output neurons.
    """
    def __init__(self,
                 feature,
                 hidden_feature,
                 out_degree_multiplier=None,
                 random_degree=False,
                 num_blocks=2,
                 use_residual_blocks=True,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 ):
        super().__init__()
        if use_residual_blocks and random_degree:
            raise ValueError("Residual blocks can't be used with random degree.")
        super().__init__()

        # First layer.
        self.first_layer = MaskedLinear(
            feature=feature,
            in_degree=_get_input_degree(feature),
            out_dim=hidden_feature,
            random_degree=random_degree
        )

        prev_out_degree = self.first_layer.out_degree

        # Hidden layers.
        blocks = []
        if use_residual_blocks:
            block_constructor = MaskedResidualBlock
        else:
            block_constructor = MaskedFeedForwardBlock

        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    feature=feature,
                    in_degree=prev_out_degree,
                    random_degree=random_degree,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    zero_initialization=True,
                )
            )
            prev_out_degree = blocks[-1].out_degree
        self.blocks = nn.ModuleList(blocks)

        # Last Layer:
        if out_degree_multiplier is not None:
            out_dim = sum(out_degree_multiplier)
        else:
            out_dim = feature
        self.last_layer = MaskedLinear(
            feature=feature,
            in_degree=prev_out_degree,
            out_dim=out_dim,
            next_layer_type='out',
            out_degree_multiplier=out_degree_multiplier
        )

    def forward(self, inputs):
        tmp = self.first_layer(inputs)
        for block in self.blocks:
            tmp = block(tmp)
        outputs = self.last_layer(tmp)
        return outputs



if __name__ == '__main__':
    # degree = _get_degree(4, 5, next_layer_type='init', out_degree_multiplier=None)
    # print(degree)
    # degree = _get_degree(4, 5, next_layer_type='hidden', out_degree_multiplier=None)
    # print(degree)
    # degree = _get_degree(4, 5, next_layer_type='out', out_degree_multiplier=None)
    # print(degree)
    # degree = _get_degree(4, 5, next_layer_type='out', out_degree_multiplier=[1,2,3,4])
    # print(degree)


    # Test of MADE.
    bs = 2
    in_feature = 4
    hidden_feature = 64

    # out_degree_multiplier = None
    out_degree_multiplier = [1, 2, 3, 7]


    x = torch.randn(bs, in_feature)

    made = FlexibleMADE(in_feature, hidden_feature, out_degree_multiplier=out_degree_multiplier)

    y = made(x)

    # j = torch.autograd.functional.jacobian(made.forward, x)
    # real_j = torch.zeros(size=[bs, in_feature, in_feature])
    # for i in range(bs):
    #     real_j[i, ...] = j[i, :, i, :]
    #
    #
    # print(real_j)

