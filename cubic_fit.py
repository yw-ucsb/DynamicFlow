import os
import argparse
import time
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('Cubic Fit')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--total_steps', type=int, default=1000)
parser.add_argument('--train_mode', type=str, choices=['uniform', 'random'], default='uniform')
parser.add_argument('--batch_num_steps', type=int, default=200)
parser.add_argument('--batch_step_size', type=int, default=1)
parser.add_argument('--hidden_neuron', type=int, default=50)
parser.add_argument('--pc', type=str, default='desktop')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--adjoint', action='store_true', default=True)
parser.add_argument('--num_train', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--test_freq', type=int, default=200)
args = parser.parse_args()

class Lambda(nn.Module):
    def forward(self, t, y):
        return y ** 3 @ true_A

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, args.hidden_neuron),
            nn.Tanh(),
            # nn.Linear(args.hidden_neuron, args.hidden_neuron),
            # nn.Tanh(),
            # nn.Linear(args.hidden_neuron, args.hidden_neuron),
            # nn.Tanh(),
            nn.Linear(args.hidden_neuron, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

def get_batch():
    s = np.random.choice(np.arange(args.num_train, dtype=np.int64), args.batch_size, replace=False)
    batch_y = y[s]
    batch_fy = true_fy[s]
    return batch_y.to(device), batch_fy.to(device)


def visualize(true_y, pred_y, odefunc):
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('y0,y1')
    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'g-', label='true y0')
    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'b-', label='true y1')
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], 'g--', label='pred y0')
    ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--', label='pred y1')
    ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label='true')
    ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--', label='pred')
    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)
    ax_phase.legend()

    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
    mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    figure.tight_layout()
    plt.savefig(FILEPATH + 'demo_pretrain.png')
    # plt.draw()
    # plt.pause(0.001)



if __name__ == '__main__':
    np.random.seed(114)
    torch.manual_seed(514)

    if args.pc == 'gauss':
        FILEPATH = '/home/yuwang/Coding/WorkSpace/DynamicFlow/node/'
    elif args.pc == 'desktop':
        FILEPATH = 'D:\/Study_Files\/UCSB\/Projects\/WorkSpace\/DynamicFlow\/node\/'
    elif args.pc == 'colab':
        FILEPATH = '/drive/MyDrive/node'
    else: raise ValueError('Invalid equipment!')

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    print('Using adjoint method:', args.adjoint)
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Generate training data;
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
    y = ((torch.rand(args.num_train, 2) - 0.5) * 4).to(device)
    true_fy = Lambda()(None, y)

    mlp = mlp().to(device)

    optimizer = optim.RMSprop(mlp.parameters(), lr=1e-3)

    tbar = tqdm(range(1, args.niters + 1))
    for itr in tbar:
        optimizer.zero_grad()
        batch_y, batch_fy = get_batch()
        batch_pred_fy = mlp(None, batch_y)
        loss = torch.abs(batch_pred_fy - batch_fy).mean()
        # loss = (torch.norm(batch_pred_fy - batch_fy, dim=1) ** 2).mean()
        loss.backward()
        optimizer.step()

        val_loss = 1000
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_fy = mlp(None, y)
                total_loss = torch.abs(pred_fy - true_fy).mean()
                # total_loss = (torch.norm(pred_fy - true_fy, dim=1) ** 2).mean()
                print('Iter {:04d} | Total Loss {:.4f}'.format(itr, total_loss.item()))
                if total_loss <= val_loss:
                    val_loss = total_loss
                    torch.save(mlp.state_dict(), FILEPATH + 'cubic_fit.t')

    # Evaluate pretrained ODE;
    true_y0 = torch.tensor([[2., 0.]]).to(device)
    t = torch.linspace(0., 25., args.total_steps).to(device)
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    print('Computing true y...')
    t_start_true_y = time.time()
    with torch.no_grad():
        true_y = odeint(Lambda().to(device), true_y0, t, method='dopri5')
    t_end_true_y = time.time()
    print('True y takes:{:.3f}s'.format(t_end_true_y - t_start_true_y))

    # print('Base draw.')
    figure, axes = plt.subplots(1, 3, figsize=(24, 8))
    ax_traj = axes[0]
    ax_phase = axes[1]
    ax_vecfield = axes[2]
    # plt.show(block=False)

    loss_plot = torch.zeros(args.niters)

    ii = 0

    mlp.load_state_dict(torch.load(FILEPATH + 'cubic_fit.t'))

    with torch.no_grad():
        pred_y = odeint(mlp, true_y0, t)
        visualize(true_y, pred_y, mlp)
        plt.show()
