# (Rami) Modified
# (Rami & Gaffar) Done 1st reading, Feb 22, 2020


# import numpy as np
# import tensorflow as tf


# Imports
# General
import os
from typing import Tuple

## ML & RL Impoorts
import numpy as np
# import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy

from spinup.utils.logx import colorize


EPS = 1e-8


# Helper functions

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



LOG_STD_MAX = 2
LOG_STD_MIN = -20


class RLDataset(IterableDataset):

    def __init__(self, buffer, sample_size: int):
        self.buffer = buffer
        self.sample_size = sample_size # Time of learning

    def __iter__(self) -> Tuple:
        data = self.buffer.sample_batch(self.sample_size)

        obs = data['obs']
        obs2 = data['obs2']
        act = data['act']
        rew = data['rew']
        done = data['done']

        for i in range(len(done)):
            yield obs[i], obs2[i], act[i], rew[i], done[i]



"""
Actor-Critics
"""

# Agent's Neural Networks (Classes) #

## Actor Network (Pi) ##
class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim) # Last layer of Actoe mean
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim) # Last layer of Actoe std
        self.act_limit = act_limit # Action constraint

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)

        mu = self.mu_layer(net_out)

        log_std = self.log_std_layer(net_out) # 
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) # 

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # Squash policy:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

## Critic Networks (Q,V) ##
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        v = self.v(torch.cat([obs], dim=-1))
        return torch.squeeze(v, -1) # Critical to ensure v has right shape.


## Actor-Critic Network (TD3 Style) ##
class MLPActorCritic(nn.Module):

    def __init__(self,
                 observation_space, action_space,
                 hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Build policy and value functions #
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)

        # create value functions, Q_phi(s,a) , Qpi_phi(s,pi(a|s)) and V_psi(s), for
        ## Jv(psi) = Expt_st~D[0.5*( V_psi(st) - Expt_at~pi[Qpi_phi(st,pi(at|st))-logpi(at|st)] )^2] -->eq#5
        ## Jq(phi) = Expt_(st,at)~D[0.5 ( Q_phi(st,at) - rt - lamda*V_psi(st+1) )^2] -->eq#6

        # value(s) = NN(x, unit: [hid_list]+1, act, out_act):
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.v = MLPVFunction(obs_dim, hidden_sizes, activation) # For V, V_targ, V'    Rami

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()



"""
Reward and transition dynamic model construction, the hidden_size for each model could
be adjusted in each subnetworks.
"""

## Model Network ##
class MLPDynModel(pl.LightningModule): # Rami (Done)

    def __init__(
                self, obs_dim, act_dim,
                lr, hidden_sizes, activation=nn.ReLU, output_activation=None,
                ):
        super().__init__()
        self.lr = lr
        # self.loss = nn.CrossEntropyLoss()

        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], obs_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], obs_dim)

    def forward(self, obs, act):

        # st+1 ~ f(st+1|st,at;omega) = N(mu,std|st,at;omega)
        net_out = self.net(torch.cat([obs, act], dim=-1))

        mu = self.mu_layer(net_out)

        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) 

        # Transition distribution 
        delta = Normal(mu, std)
        delta  = delta.rsample()

        return delta
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)
        
    def JofDynModel(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss J
        """
        pass

    def training_step(self, batch, batch_idx):

        # o = batch['obs']
        # o2 = batch['obs2']
        # a = batch['act']
        # r= batch['rew']
        # # d = batch['done']
        o, o2, a, r, _ = batch

        delta = self(o,a)
        # transition = o + delta
        delta_backup = o2 - o

        Jdyn = ((delta_backup - delta)**2).mean()
        # Jdyn = self.loss(delta, delta_backup)
        # acc = accuracy(delta, delta_backup)

        return {'loss': Jdyn}
        # pbar = {'train_acc': acc}
        # return {'loss': Jdyn, 'progress_bar': pbar}


class MLPRewModel(pl.LightningModule): # Rami (Done)

    def __init__(
            self, obs_dim, act_dim,
            lr, hidden_sizes, activation=nn.ReLU, output_activation=None,
            ):
        super().__init__()
        self.lr = lr
        # self.loss = nn.CrossEntropyLoss()

        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], 1)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, act):

        # rt ~ N(mu,std|st,at;ph)
        net_out = self.net(torch.cat([obs, act], dim=-1))

        mu = self.mu_layer(net_out)

        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) 

        # Reward distribution
        reward = Normal(mu, std)
        reward = reward.rsample()

        return reward

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):

        # o = batch['obs']
        # o2 = batch['obs2']
        # a = batch['act']
        # r= batch['rew']
        # d = batch['done']
        o, o2, a, r, _ = batch

        r_rm = self(o,a)
        r_backup = r

        Jrew = ((r_backup-r_rm)**2).mean()
        # Jdyn = self.loss(r_rm, r_backup)
        # acc = accuracy(r_rm, r_backup)

        # return pl.TrainResult(Jrew)
        return {'loss': Jrew}
        # pbar = {'train_acc': acc}
        # return {'loss': Jrew, 'progress_bar': pbar}


class MLPModel(pl.LightningModule): # Rami (Done)

    def __init__(self,
                 observation_space, action_space,
                 lr,
                 hidden_sizes=(256,128),
                 activation=nn.ReLU,
                 output_activation=None):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        # replay_buffer =  replay_buffer

        self.delta = MLPDynModel(obs_dim, act_dim, lr, hidden_sizes, activation, output_activation)
        
        self.reward = MLPRewModel(obs_dim, act_dim, lr, hidden_sizes, activation, output_activation)

    def train_model(
            self,
            replay_buffer,
            dataset_size,
            model_batch_size,
            model_epochs,
            # pbar_rr,
            # ws
            ):
        print(colorize('replay buffer size', color='cyan'), replay_buffer.size)
        print(colorize('dataset_size', color='cyan'), dataset_size)
        dataset = RLDataset(replay_buffer, dataset_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=model_batch_size,
            sampler=None,
            num_workers=12
        )

        # Pbar_rr = 10
        ws = None

        # model_epochs = t // batch_size
        # model_epochs = 5
        # if t <= 5e5:
        #     model_epochs = 5
        # elif 2e2 < t:
        #     model_epochs = 10

        # if t > 2000:
        #     print(colorize('Load parameters..', color='grey', bold=True))
        #     Ckpt_Path = self.check_point_path()
        #     nn_model3 = ResNet().load_from_checkpoint(checkpoint_path=Ckpt_Path)

        print(colorize('Dynamics model training..', color='red', bold=True))
        dyn_trainer = pl.Trainer(
            max_epochs=model_epochs,
            gpus=1,
            weights_summary=ws,
            # progress_bar_refresh_rate=pbar_rr
            )
        dyn_trainer.fit(self.delta, dataloader)
        print(colorize('Reward model training..', color='red', bold=True))
        dyn_trainer = pl.Trainer(
            max_epochs=model_epochs,
            gpus=1,
            weights_summary=ws,
            # progress_bar_refresh_rate=pbar_rr
            )
        dyn_trainer.fit(self.delta, dataloader)
        # return loss_model

    # def check_point_path(self):
            
    #         cwd = os.getcwd()
    #         Folder_Path = cwd + "/lightning_logs"

    #         versions = []
    #         for f in os.listdir(Folder_Path):
    #             if f.startswith('version'):
    #                 versions.append(f)
    #         v = len(versions)
    #         Version_Path = Folder_Path + f"/version_{v-1}"

    #         for root, dirnames, files in os.walk(Version_Path):
    #             for f in files:
    #                 if f.endswith('.ckpt'):
    #                     ckpt = f

    #         Ckpt_Path = Version_Path + '/checkpoints/' + ckpt
            
    #         return Ckpt_Path


