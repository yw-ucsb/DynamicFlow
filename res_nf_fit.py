import numpy as np
import torch
import time
import nflows
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.utils import torchutils
from nflows.nn.nets import ResidualNet
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
dim_hidden = 16
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

base_dist = StandardNormal(shape=[dim])

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

RESNET = ResidualNet(in_features=dim, out_features=dim, hidden_features=dim_hidden)

optimizer = optim.Adam([
    {'params': flow.parameters()},
    {'params': RESNET.parameters()}],
    lr=lr)


train_loss = np.zeros(shape=(num_iterations))
val_score = np.zeros(shape=(int(num_iterations / val_interval)))

start = time.time()

t_val = 0
count_val = 0
best_val_score = 1000

tbar = tqdm(range(num_iterations))
for i in tbar:
    # Training iterations;
    RESNET.train()
    flow.train()
    batch = next(train_generator).to(device)
    # print(batch.shape)
    optimizer.zero_grad()

    predict = RESNET(batch[:, 1, :])
    true = batch[:, 0, :]
    loss = -flow.log_prob(inputs=true, context=predict).mean()

    # print('Current loss:', loss.detach().cpu().numpy())
    train_loss[i] = loss.detach().numpy()
    loss.backward()
    optimizer.step()

    if (i + 1) % val_interval == 0:
        RESNET.eval()
        flow.train()
        val_start = time.time()
        with torch.no_grad():
            # compute validation score
            running_val_log_density = 0
            for val_batch in val_loader:
                val_predict = RESNET(val_batch[:, 1, :])
                val_true = val_batch[:, 0, :]
                val_loss = -flow.log_prob(inputs=val_true, context=val_predict).mean()
                mean_log_density_val = torch.mean(val_loss).detach()
                running_val_log_density += mean_log_density_val
            running_val_log_density /= len(val_loader)
        if running_val_log_density < best_val_score:
            best_val_score = running_val_log_density
            torch.save(flow.state_dict(), FILEPATH + 'resnf_nf_model_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers))
            torch.save(RESNET.state_dict(), FILEPATH + 'resnf_res_model_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers))
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
                trajectory_base[j] = context_base.detach().cpu()
                s_base = RESNET(context_f)
                s_f = flow._sample(1, context=s_base.to(device)).squeeze(dim=1)
                context_f = s_f
                context_base = s_base
            cov = cov.transpose(dim0=0, dim1=1)
            mean_tr_f = torch.mean(trajectory_f, dim=1)
            mean_tr_base = torch.mean(trajectory_base, dim=1)

            plt.scatter(mean_tr_f[:, 0][::plot_interval], mean_tr_f[:, 1][::plot_interval],
                        label='{}th plot'.format((i + 1) // 1000))
            # plt.scatter(mean_tr_base[:, 0], mean_tr_base[:, 1], edgecolors='g', label='{}th plot'.format(i))
            # plt.plot(mean_tr_base[:, 0], mean_tr_base[:, 1])

flow.load_state_dict(torch.load(FILEPATH + 'resnf_nf_model_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers)))
flow.eval()
RESNET.load_state_dict(torch.load(FILEPATH + 'resnf_res_model_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers)))
RESNET.eval()

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
    trajectory_base[i] = context_base.detach().cpu()
    s_base = RESNET(context_f)
    s_f = flow._sample(1, context=s_base.to(device)).squeeze(dim=1)
    context_f = s_f
    context_base = s_base
cov = cov.transpose(dim0=0, dim1=1)
mean_tr_f = torch.mean(trajectory_f, dim=1)
mean_tr_base = torch.mean(trajectory_base, dim=1)

real = torch.zeros(t_length, dim)
A = torch.eye(dim) * 1.005
# A = torch.rand([dim, dim])
b = torch.zeros(dim) + 0.005
for i in range(t_length - 1):
    real[i + 1] = A @ real[i] + b

plt.title('resnf_t{}_bs{}_hd{}_nl{}.t'.format(data_index, batch_size, dim_hidden, num_layers))
plt.scatter(real[:, 0][::plot_interval], real[:, 1][::plot_interval], edgecolors='b')
# for i in range(bs):
#     plt.scatter(mean[:, i, 0], mean[:, i, 1])
#     plt.scatter(trajectory_f[:, i, 0], trajectory_f[:, i, 1])
plt.scatter(mean_tr_f[:, 0][::plot_interval], mean_tr_f[:, 1][::plot_interval], edgecolors='r', label='Best')
# plt.scatter(mean_tr_base[:, 0][::plot_interval], mean_tr_base[:, 1][::plot_interval], edgecolors='g', label='latent')
# plt.plot(mean_tr_base[:, 0], mean_tr_base[:, 1])
plt.legend()
plt.show()