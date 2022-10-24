'''
Implementation of Blocked version of MADE
'''

import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import distributions, nn
from torch.nn import functional as F
from torch.nn import init

from utils import torchutils

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)

# Basic function to create the block degree given input size and number of blocks;
def _create_block_degree(feature, out_block_feature, num_block, mode='stack'):
    # The dimensions within the same block id is stacked together;
    out_feature_all = sum(feature)
    if mode == 'stack':
        tmp = []
        for i in range(len(feature)):
            block_degree_i = torch.arange(1, num_block + 1).repeat_interleave(out_block_feature[i])
            tmp.append(block_degree_i[0:feature[i]])
        block_degree = torch.cat(tmp)
    elif mode == 'seq':
        tmp = []
        for i in range(len(feature)):
            block_feature_i = math.ceil(feature[i] / (num_block - 1))
            block_degree_i = torch.arange(1, num_block).repeat(block_feature_i)
            tmp.append(block_degree_i[0:feature[i]])
        block_degree = torch.cat(tmp)
    elif mode == 'random':
        block_degree = torch.randint(
            low=1,
            high=num_block,
            size=[out_feature_all],
            dtype=torch.long,
        )
    else:
        raise ValueError('Invalid create degree mode. Use stack, seq or random.')
    return block_degree[0:out_feature_all]


class BlockMaskedLinear(nn.Linear):
    def __init__(self, in_feature, out_feature, out_block_feature, in_block_degree, num_block, rand_mask=False, is_output=False, bias=True):
        # Check the input block feature and num of blocks are legal:
        # for o_feature, b_feature in (out_feature, block_feature):
        #     if num_block != math.ceil(o_feature / b_feature):
        #         raise ValueError('Invalid block feature size and number of blocks!')
        out_feature_all = sum(out_feature)
        in_feature_all = sum(in_feature)
        super().__init__(in_features=in_feature_all, out_features=out_feature_all, bias=bias)
        self.out_feature = out_feature
        self.out_block_feature = out_block_feature
        self.in_block_degree = in_block_degree
        self.num_block = num_block
        self.rand_mask = rand_mask
        self.is_output = is_output
        # Get the mask and block degree of the current layer;
        mask, block_degree = self._get_mask_and_block_degree()
        self.register_buffer('mask', mask)
        self.register_buffer('block_degree', block_degree)

    def _get_mask_and_block_degree(self):
        # Assign sequential block degree for 'block_feature' times for every output unit in each block from 0 to 'num_blocks - 1';
        if self.is_output:
            block_degree = _create_block_degree(feature=self.out_feature,
                                                out_block_feature=self.out_block_feature, num_block=self.num_block, mode='stack')
            mask = (block_degree[..., None] > self.in_block_degree).float()
        else:
            if isinstance(self.out_features, list):
                raise ValueError('Output feature cannot be a list unless the current layer is final output!')
            # Assign random mask from 1 to 'num_blocks' for each hidden unit;
            if self.rand_mask:
                block_degree = _create_block_degree(feature=self.out_feature,
                                                    out_block_feature=self.out_block_feature, num_block=self.num_block, mode='random')
                mask = (block_degree[..., None] >= self.in_block_degree).float()
            # Assign sequential mask from 1 to 'num_blocks' for each hidden unit;
            else:
                block_degree = _create_block_degree(feature=self.out_feature,
                                                    out_block_feature=self.out_block_feature, num_block=self.num_block, mode='seq')
                mask = (block_degree[..., None] >= self.in_block_degree).float()
        return mask, block_degree

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class BlockMaskedFeedForwardLayer(nn.Module):
    def __init__(self,
                 in_feature,
                 out_feature,
                 out_block_feature,
                 in_block_degree,
                 num_block,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=False
                 ):
        super().__init__()

        # Batch norm;
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(in_feature, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear layer;
        self.mask_linear = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=out_feature,
            out_block_feature=out_block_feature,
            in_block_degree=in_block_degree,
            num_block=num_block,
            rand_mask=rand_mask,
            is_output=False
        )
        self.block_degree = self.mask_linear.block_degree

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


class BlockMaskedResidualLayer(nn.Module):
    def __init__(self,
                 in_feature,
                 out_feature,
                 out_block_feature,
                 in_block_degree,
                 num_block,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False,
                 zero_initialization=True
                 ):
        if rand_mask:
            raise ValueError('Masked residual block cant be used with random masks.')
        if in_feature != out_feature:
            raise ValueError('In and out feature should be the same.')
        super().__init__()

        # Batch norm.
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(in_feature, eps=1e-3) for _ in range(2)]
            )

        # Linear layers;
        linear_layer_0 = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=in_feature,
            out_block_feature=out_block_feature,
            in_block_degree=in_block_degree,
            num_block=num_block,
            rand_mask=rand_mask,
            is_output=False
        )
        linear_layer_1 = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=in_feature,
            out_block_feature=out_block_feature,
            in_block_degree=linear_layer_0.block_degree,
            num_block=num_block,
            rand_mask=rand_mask,
            is_output=False
        )
        self.linear_layers = nn.ModuleList([linear_layer_0, linear_layer_1])
        self.block_degree = linear_layer_1.block_degree

        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

        # Initialization.
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


class BlockMADE(nn.Module):
    def __init__(self,
                 in_feature,
                 hidden_feature,
                 out_feature,
                 out_block_feature,
                 block_feature,
                 num_hidden_layer=2,
                 use_res_layer=True,
                 rand_mask=False,
                 activation=F.relu,
                 dropout_probability=0.0,
                 use_batch_norm=False
                 ):
        if use_res_layer and rand_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        self.num_block = math.ceil(in_feature[0] / block_feature[0])

        # Construct the BlockMADE network;
        self.first_layer = BlockMaskedLinear(
            in_feature=in_feature,
            out_feature=hidden_feature,
            out_block_feature=out_block_feature,
            # Input and output should be in the same block degree.
            in_block_degree=_create_block_degree(in_feature, block_feature, num_block=self.num_block),
            num_block=self.num_block,
            rand_mask=rand_mask,
            is_output=False
        )
        prev_block_degree = self.first_layer.block_degree

        # Hidden layers;
        self.hidden_layers = nn.ModuleList([])
        if use_res_layer:
            hidden_layer = BlockMaskedFeedForwardLayer
        else:
            hidden_layer = BlockMaskedResidualLayer
        for _ in range(num_hidden_layer):
            self.hidden_layers.append(hidden_layer(
                in_feature=hidden_feature,
                out_feature=hidden_feature,
                out_block_feature=out_block_feature,
                in_block_degree=prev_block_degree,
                num_block=self.num_block,
                rand_mask=rand_mask,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                zero_initialization=True
            ))
            if num_hidden_layer != 0:
                prev_block_degree = self.hidden_layers[-1].block_degree

        # Last layer;
        self.last_layer = BlockMaskedLinear(
            in_feature=hidden_feature,
            out_feature=out_feature,
            out_block_feature=out_block_feature,
            in_block_degree=prev_block_degree,
            num_block=self.num_block,
            rand_mask=rand_mask,
            is_output=True
        )

    def forward(self, inputs):
        tmp = self.first_layer(inputs)
        for hidden_layer in self.hidden_layers:
            tmp = hidden_layer(tmp)
        outputs = self.last_layer(tmp)
        return outputs


if __name__ == '__main__':
    batch_size = 2
    in_feature = 6
    hidden_feature = 64
    block_feature = 1
    num_block = math.ceil(in_feature / block_feature)
    out_feature = in_feature

    inputs = torch.randn(batch_size, in_feature)

    # print(_create_block_degree([6, 6], 2))


    block_made = BlockMADE(
        in_feature=[in_feature],
        hidden_feature=[hidden_feature],
        out_feature=[out_feature],
        out_block_feature=[block_feature],
        block_feature=[block_feature],
        use_res_layer=True,
        num_hidden_layer=2
    )

    print(_create_block_degree([in_feature], [block_feature], num_block, 'stack'))
    print(block_made.first_layer.block_degree)
    print(block_made.last_layer.block_degree)
    # print(block_made.last_layer.num_block)
    print(block_made.hidden_layers[-1].block_degree)

    # print(block_made.first_layer.mask.shape)

    def b_forward(inputs):
        outputs = block_made(inputs)
        return outputs

    print(b_forward(inputs))

    j = torch.autograd.functional.jacobian(b_forward, inputs)
    # j = torch.autograd.functional.jacobian(block_made.forward, inputs)
    real_j = torch.zeros(size=[batch_size, in_feature, in_feature])
    for i in range(batch_size):
        real_j[i, ...] = j[i, :, i, :]
    print('Torch Jacobian:\n', real_j)

    outputs = block_made(inputs)
