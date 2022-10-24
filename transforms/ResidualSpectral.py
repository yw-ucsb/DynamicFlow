import torch

torch.set_printoptions(
    precision=4,
    sci_mode=False
)


import math
import numpy as np
import torch
import MaskedPWA as MPWA
import torch.nn.functional as F

from transforms.base import Transform

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)


class ResidualSpectralTransformation(Transform):
    def __init__(self, feature, hidden_feature, num_hidden_layer, use_resnet=False):
        super().__init__()
        self.PWANet = MPWA.MaskedPWANet(
            feature=feature,
            hidden_feature=hidden_feature,
            num_hidden_layer=num_hidden_layer,
            use_resnet=use_resnet
        )

    def forward(self, inputs, context=None):
        outputs, deri = self.PWANet(inputs)
        # output_real = self.PWANet.direct_forward(inputs)

        # print('error mean:', (output_real-outputs).mean())

        # assert ~torch.any(deri <= 0), 'deri:{}, {}, {}'.format(deri[0], deri[2], deri[3])

        # assert torch.equal(output_real, outputs), 'r:{},\n f:{}'.format(output_real, outputs)
        logabsdet = torch.sum(torch.log(torch.abs(deri)), dim=1)

        return outputs, logabsdet




if __name__ == "__main__":
    # np.random.seed(1137)
    # torch.manual_seed(114514)
    batch_size = 4
    num_layer = 5
    feature = 4
    hidden_feature = 32
    inputs = torch.randn(batch_size, feature, requires_grad=True)
    net = ResidualSpectralTransformation(feature=feature, hidden_feature=hidden_feature,  num_hidden_layer=num_layer)

    y, logabsdet = net(inputs)

    real_output = net.PWANet.direct_forward(inputs)
    print('Real output:\n', real_output.detach())
    print('PWA output:\n', y.detach())

    j = torch.autograd.functional.jacobian(net.forward, inputs)
    real_j = torch.zeros(size=[batch_size, feature, feature])
    for i in range(batch_size):
        real_j[i, ...] = j[0][i, :, i, :]
    print('Real jacobian:\n', real_j)

    print('Real abslogDet:\n', torch.log(torch.abs(torch.det(real_j))))

    print('Estimated abslogDet:\n', logabsdet)




