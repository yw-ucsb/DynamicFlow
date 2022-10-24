import numpy as np
import torch
import time
import nflows
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.nn.nets import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from nflows.utils import torchutils
from nflows.nn import nets
from nflows import transforms

from torch import optim

import matplotlib.pyplot as plt
from matplotlib import colors


pc = 'desktop'
# pc = 'laptop'
if pc == 'desktop': FILEPATH = 'D:\/Study_Files\/UCSB\/Projects\/WorkSpace\/DynamicFlow\/trajectory_data\/'
elif pc == 'laptop': FILEPATH = 'D:\/Coding\/WorkSpace\/DynamicFlow\/trajectory_data\/'
else: FILEPATH = 'D:\/Study_Files\/UCSB\/Projects\/WorkSpace\/DynamicFlow\/trajectory_data\/'

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

np.random.seed(1137)
torch.manual_seed(114514)

# Setup device;
assert torch.cuda.is_available()
device = torch.device('cpu')
# torch.set_default_tensor_type(torch.FloatTensor)

bs = 1024
batch_size = bs
dim = 2
dim_hidden = 64
num_layers = 2
lr = 1e-4
plt.figure(figsize=(12, 12))
plot_interval = 3

num_iterations = 10000
val_interval = 250


# Load data and pack as dataset;
data_index = 9
data = torch.load(FILEPATH + 't{}.pt'.format(data_index))
N_test = int(0.1 * data.shape[0])
data_test = data[-N_test:]
data = data[0:-N_test]
N_validate = int(0.1 * data.shape[0])
data_val = data[-N_validate:]
data_train = data[0:-N_validate]

class t_dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n = self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

def batch_generator(loader, num_batches=int(1e10)):
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return

train = t_dataset(data_train)
train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, drop_last=True)
train_generator = batch_generator(train_loader)
print('num_train:', len(data_train))

val = t_dataset(data_val)
val_loader = DataLoader(data_val, batch_size=bs, shuffle=True, drop_last=True)
val_generator = batch_generator(val_loader)
print('num_val:', len(data_val))

test = t_dataset(data_test)
test_loader = DataLoader(data_test, batch_size=bs, shuffle=True, drop_last=True)
test_generator = batch_generator(test_loader)
print('num_test:', len(data_test))



base_dist = ConditionalDiagonalNormal(shape=[dim],
                               context_encoder=ResidualNet(
                                   in_features=dim,
                                   out_features=dim * 2,
                                   hidden_features=dim_hidden
                               )
                               )

transform = []

for _ in range(num_layers):
    transform.append(MaskedAffineAutoregressiveTransform(features=dim, hidden_features=dim_hidden, context_features=dim))
    # transform.append(
    #     transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
    #         features=dim,
    #         hidden_features=dim_hidden,
    #         context_features=dim,
    #         num_bins=8,
    #         tails='linear',
    #         tail_bound=3.0
    #     )
    # )
    transform.append(ReversePermutation(features=dim))

Transform = CompositeTransform(transform)

flow = Flow(Transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters(), lr=lr)


train_loss = np.zeros(shape=(num_iterations))
val_score = np.zeros(shape=(int(num_iterations / val_interval)))

start = time.time()

t_val = 0
count_val = 0
best_val_score = 1000

tbar = tqdm(range(num_iterations))
for i in tbar:
    # Training iterations;
    flow.train()
    batch = next(train_generator).to(device)
    # print(batch.shape)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=batch[:, 0, :], context=batch[:, 1, :]).mean()
    # print('Current loss:', loss.detach().cpu().numpy())
    # train_loss[i] = loss.detach().numpy()
    loss.backward()
    optimizer.step()

    if (i + 1) % val_interval == 0:
        flow.eval()
        val_start = time.time()
        with torch.no_grad():
            # compute validation score
            running_val_log_density = 0
            for val_batch in val_loader:
                val_inputs = val_batch[:, 0, :]
                val_context = val_batch[:, 1, :]
                log_density_val = -flow.log_prob(inputs=val_inputs.to(device).detach(), context=val_context.to(device).detach())
                mean_log_density_val = torch.mean(log_density_val).detach()
                running_val_log_density += mean_log_density_val
            running_val_log_density /= len(val_loader)
        if running_val_log_density < best_val_score:
            best_val_score = running_val_log_density
            torch.save(flow.state_dict(), FILEPATH + 'cgnf_t{}_bs{}_hd{}_nl{}.t'.format(data_index, bs, dim_hidden, num_layers))
        val_score[count_val] = running_val_log_density.cpu().detach().numpy()
        print('Current validation score is:', val_score[count_val])
        val_end = time.time()
        t_val = t_val + (val_end - val_start)
        count_val += 1

    # Plot the unrolled trajectories during training;
    if (i + 1) % 1000 == 0:
        bs = 3000
        context_f = torch.zeros([bs, dim])
        context_base = torch.zeros([bs, dim])
        t_length = 100
        trajectory_f = torch.zeros(t_length, bs, dim)
        trajectory_base = torch.zeros(t_length, bs, dim)
        cov = torch.zeros(t_length, bs, dim, dim)
        mean = torch.zeros(t_length, bs, dim)
        for j in range(t_length):
            trajectory_f[j] = context_f.detach().cpu()
            s_f = flow._sample(1, context_f.to(device)).squeeze(dim=1)
            trajectory_base[j] = context_base.detach().cpu()
            s_base = base_dist._sample(1, context_base.to(device)).squeeze(dim=1)
            context_f = s_f
            context_base = s_base
        cov = cov.transpose(dim0=0, dim1=1)
        mean_tr_f = torch.mean(trajectory_f, dim=1)
        mean_tr_base = torch.mean(trajectory_base, dim=1)

        plt.scatter(mean_tr_f[:, 0][::plot_interval], mean_tr_f[:, 1][::plot_interval], label='{}th plot'.format((i+1)//1000))
        # plt.scatter(mean_tr_base[:, 0], mean_tr_base[:, 1], edgecolors='g', label='{}th plot'.format(i))
        # plt.plot(mean_tr_base[:, 0], mean_tr_base[:, 1])



flow.load_state_dict(torch.load(FILEPATH + 'cgnf_t{}_bs{}_hd{}_nl{}.t'.format(data_index, bs, dim_hidden, num_layers)))
flow.eval()

# # Evaluation of generalization, mean evolution;
# bs = 5
# context = torch.zeros([bs, dim])
# t_length = 100
# trajectory0 = torch.zeros(t_length, bs, dim)
# cov = torch.zeros(t_length, bs, dim, dim)
# mean = torch.zeros(t_length, bs, dim)
# for i in range(t_length):
#     trajectory0[i] = context.detach().cpu()
#     samples = flow._sample(5000, context.to(device)).detach().cpu()
#     # for j in range(bs):
#         # sample = torch.transpose(samples[j], dim0=0, dim1=1)
#         # cov[i, j] = torch.tensor(np.cov(sample.numpy()))
#     mean[i] = samples.mean(dim=1)
#     context = mean[i]
# cov = cov.transpose(dim0=0, dim1=1)
# mean_tr = torch.mean(trajectory0, dim=1)
#
# real = torch.zeros(t_length, dim)
# A = torch.eye(dim) * 1.005
# # A = torch.rand([dim, dim])
# b = torch.zeros(dim) + 0.005
# for i in range(t_length - 1):
#     real[i + 1] = A @ real[i] + b
#
# plt.scatter(real[:, 0], real[:, 1])
# for i in range(bs):
#     plt.scatter(mean[:, i, 0], mean[:, i, 1])
# plt.show()

# Evaluation of generalization, trajectory mean;
# for k in range(10):
bs = 3000
context_f = torch.zeros([bs, dim])
context_base = torch.zeros([bs, dim])
t_length = 100
trajectory_f = torch.zeros(t_length, bs, dim)
trajectory_base = torch.zeros(t_length, bs, dim)
cov = torch.zeros(t_length, bs, dim, dim)
mean = torch.zeros(t_length, bs, dim)
for i in range(t_length):
    trajectory_f[i] = context_f.detach().cpu()
    s_f = flow._sample(1, context_f.to(device)).squeeze(dim=1)
    trajectory_base[i] = context_base.detach().cpu()
    s_base = base_dist._sample(1, context_base.to(device)).squeeze(dim=1)
    # samples = flow._sample(5000, context.to(device)).detach().cpu()
    # for j in range(bs):
        # sample = torch.transpose(samples[j], dim0=0, dim1=1)
        # cov[i, j] = torch.tensor(np.cov(sample.numpy()))
    # mean[i] = samples.mean(dim=1)
    context_f = s_f
    context_base = s_base
    # context = mean[i]
cov = cov.transpose(dim0=0, dim1=1)
mean_tr_f = torch.mean(trajectory_f, dim=1)
mean_tr_base = torch.mean(trajectory_base, dim=1)

real = torch.zeros(t_length, dim)
A = torch.eye(dim) * 1.005
# A = torch.rand([dim, dim])
b = torch.zeros(dim) + 0.005
for i in range(t_length - 1):
    real[i + 1] = A @ real[i] + b

plt.title('cgnf_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers))
plt.scatter(real[:, 0][::plot_interval], real[:, 1][::plot_interval], edgecolors='b')
# for i in range(bs):
#     plt.scatter(mean[:, i, 0], mean[:, i, 1])
#     plt.scatter(trajectory_f[:, i, 0], trajectory_f[:, i, 1])
plt.scatter(mean_tr_f[:, 0][::plot_interval], mean_tr_f[:, 1][::plot_interval], edgecolors='r', label='Best')
plt.scatter(mean_tr_base[:, 0][::plot_interval], mean_tr_base[:, 1][::plot_interval], edgecolors='g', label='latent')
# plt.plot(mean_tr_base[:, 0], mean_tr_base[:, 1])
plt.legend()
plt.show()

#
# # Evaluation of trajectories
# bs = 10
# context = torch.zeros([bs, dim])
# t_length = 100
# trajectory = torch.zeros(t_length, bs, dim)
# t_data = torch.zeros(t_length, bs, 2, dim)
# for i in range(t_length):
#     trajectory[i] = context.detach().cpu()
#     s = flow._sample(1, context.to(device)).squeeze(dim=1)
#     context = s
#
# t_plot = torch.transpose(trajectory, dim0=0, dim1=1)
# t_plot[t_plot > 100.0] = 100.0
# for j in range(bs):
#     plt.plot(t_plot[j, :, 0], t_plot[j, :, 1])
# plt.title('Trajectories from NF fitting t{}'.format(data_index))
# plt.show()

# # Evaluation of the log density and samples;
# step = 100
# rangee = 4
# x_bias = 0.2
# y_bias = 0.1
# bias = torch.tensor([x_bias, y_bias])
# x = torch.linspace(-rangee + x_bias, rangee + x_bias, step)
# y = torch.linspace(-rangee + y_bias, rangee + y_bias, step)
# # x = torch.linspace(-1, 1, step)
# # y = torch.linspace(-3, -1, step)
# log_context = torch.zeros(step ** 2, 2) + bias
# X, Y = torch.meshgrid(x, y)
# xy = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
# # z = flow._log_prob(xy, log_context.to(device)).detach().cpu().exp()
# z = base_dist._log_prob(xy, log_context.to(device)).detach().cpu().exp()
# z_max = z.max()
# Z = z.reshape(step, step)
#
# figure, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# # plt.title('Max: {:.3f}'.format(z_max.numpy()))
# axes[0].set_title('Density, log scaled')
# # axes[0].contourf(X.numpy(), Y.numpy(), Z.numpy(), cmap='summer', norm=colors.LogNorm())
# axes[0].contourf(X.numpy(), Y.numpy(), Z.numpy(), cmap='summer')
#
#
# sample_context = torch.zeros([1, 2]) + bias
# # s = flow._sample(10000, sample_context.to(device)).detach().numpy()
# s = base_dist._sample(10000, sample_context.to(device)).detach().numpy()
# s_mean = torch.tensor([s[0, :, 0].mean(), s[0, :, 1].mean()])
# # plt.title('Mean: [{:3f}, {:3f}]'.format(s_mean.numpy()[0], s_mean.numpy()[1]))
# axes[1].set_title('Samples')
# plt.tight_layout()
#
# axes[1].scatter(s[0, :, 0], s[0, :, 1], alpha=0.02)
# plt.show()