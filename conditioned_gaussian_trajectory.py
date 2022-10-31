import numpy as np
import torch
import nflows
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import colors

class ConditionalGaussian(nn.Module):
    def __init__(self, shape, context_net=None):
        super().__init__()
        self.shape = torch.Size(shape)
        # self.dim_context = dim_context
        if context_net is None:
            self.context_net = lambda x: x
        else:
            self.context_net = context_net
        self.register_buffer('_log_z',
                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _compute_parameters(self, context):
        raise NotImplementedError

    def _log_prob(self, inputs, context):
        raise NotImplementedError

    def _sample(self, num_samples, context):
        raise NotImplementedError

class MeanConditionalGaussian(ConditionalGaussian):
    def __init__(self, shape, log_covs, context_net=None):
        super().__init__(shape, context_net)
        self.log_covs = log_covs

    def _compute_parameters(self, context):
        if context is None:
            raise ValueError('Context needs to be a tensor')
        means = self.context_net(context)

        return means

    def _log_prob(self, inputs, context):
        if inputs.shape[1:] != self.shape:
            raise ValueError(
                "Shape missmatch. Expected input of shape {}, got {}".format(self.shape, inputs.shape[1:])
            )
        # Compute the distribution parameters
        means = self._compute_parameters(context)
        # Calculate the log density
        log_prob = -0.5 * torch.sum(
            (inputs - means) ** 2, dim=1
        )
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        # print(self.log_covs)
        batch_size = context.shape[0]
        means = self._compute_parameters(context).repeat_interleave(num_samples, dim=0)
        covs = torch.exp(self.log_covs).repeat_interleave(num_samples, dim=0)

        # print(means)
        # print(covs)

        noise = torch.randn(batch_size * num_samples, *
                            self.shape, device=means.device
                            )
        samples = means + covs * noise
        return samples.reshape(batch_size, num_samples, -1)

class LinearDynamic(nn.Module):
    """
    Linear dynamics with specified matrix: s' = A * s + b
    """
    def __init__(self, A=None, b=None):
        super().__init__()
        self.A = A
        self.b = b

    def forward(self, inputs):
        if self.A.shape[0] != inputs.shape[-1] or self.b.shape[0] != inputs.shape[-1]:
            raise ValueError('Size of dynamics and inputs mismatch.')
        if self.A is None or self.b is None:
            # Default case is identity transformation
            return inputs
        else:
            return inputs @ self.A + self.b

if __name__ == '__main__':
    # pc = 'desktop'
    # pc = 'laptop'
    pc = 'gauss'
    if pc == 'desktop':
        FILEPATH = 'D:\/Study_Files\/UCSB\/Projects\/WorkSpace\/DynamicFlow\/trajectory_data\/'
    elif pc == 'laptop':
        FILEPATH = 'D:\/Coding\/WorkSpace\/DynamicFlow\/trajectory_data\/'
    elif pc == 'gauss':
        FILEPATH = '/home/yuwang/Coding/WorkSpace/DynamicFlow/'
    else:
        FILEPATH = 'D:\/Study_Files\/UCSB\/Projects\/WorkSpace\/DynamicFlow\/trajectory_data\/'

    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    np.random.seed(1137)
    torch.manual_seed(114514)

    index = 9
    bs = 5000
    dim = 2
    n_sample = 20000
    context = torch.zeros([bs, dim])
    A = torch.eye(dim) * 1.005
    # A = torch.rand([dim, dim])
    b = torch.zeros(dim) + 0.005
    # b = torch.zeros(dim) + torch.randn(dim)
    print('A:', A)
    print('b:', b)
    context_net = LinearDynamic(A, b)
    # log_covs = torch.zeros([bs, dim])
    log_covs = torch.zeros([bs, dim]) - 3
    dist = MeanConditionalGaussian(shape=[dim], log_covs=log_covs, context_net=context_net)

    # x = dist._sample(n_sample, context)
    #
    # x0 = x.squeeze(dim=0)
    #
    # plt.scatter(x0[:, 0], x0[:, 1])
    # plt.show()

    # y = torch.zeros([bs, dim])
    # print(dist._log_prob(y, context))

    t_length = 100
    trajectory = torch.zeros(t_length + 1, bs, dim)
    t_data = torch.zeros(t_length, bs, 2, dim)
    for i in range(t_length):
        s = dist._sample(1, context).squeeze(dim=1)
        s_pair = torch.stack((s, context), dim=1)
        t_data[i] = s_pair
        context = s
        trajectory[i + 1] = context

    t_plot = torch.transpose(trajectory, dim0=0, dim1=1)
    t_mean_plot = torch.mean(t_plot, dim=0)
    # plt.scatter(trajectory.squeeze()[:, 0], trajectory.squeeze()[:, 1])
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j in range(bs):
        axes[0].plot(t_plot[j, :, 0], t_plot[j, :, 1])
    axes[0].set_title('True training trajectories, t{}'.format(index))

    t_plot_hist = t_plot.reshape(-1, 2).numpy()
    # limit = 4
    # bounds = np.array([
    #     [-limit, limit],
    #     [-limit, limit]
    # ])
    axes[1].hist2d(t_plot_hist[:, 0], t_plot_hist[:, 1], bins=500, cmax=100, density=True, cmap='summer', norm=colors.LogNorm())
    axes[2].scatter(t_mean_plot[:, 0], t_mean_plot[:, 1])
    plt.show()


    t_data = t_data.reshape(bs * t_length, 2, dim)



    torch.save(t_data, FILEPATH + 't{}.pt'.format(index))
    torch.save(torch.flip(t_plot, dims=[1]), FILEPATH + 't_whole{}.pt'.format(index))