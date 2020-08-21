"""
OpenAI: Original from spinningup/spinup/algos/pytorch/sac/core.py,
(Gaffar): Added Deterministic Ensemles of Dynamics Model,
# >>> Added by Gaffar >>>
# <<< Added by Gaffar <<<
(Rami): Modefied
"""

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


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

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

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

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


# >>> Added by Gaffar >>>
class DynamicsModel0(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes_dyn):
        super().__init__()
        self.Fc1 = nn.Linear(obs_dim + act_dim, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc1.weight)

        self.Fc2 = nn.Linear(hidden_sizes_dyn, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc2.weight)
     
        self.delta = nn.Linear(hidden_sizes_dyn, obs_dim)
        nn.init.xavier_normal_(self.delta.weight)
    def forward(self, obs, act):
        net_out = F.relu(self.Fc1(torch.cat([obs, act], dim=-1)))
        net_out = F.relu(self.Fc2(net_out))

        delta = self.delta(net_out)
        return delta

class DynamicsModel1(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes_dyn):
        super().__init__()
        self.Fc1 = nn.Linear(obs_dim + act_dim, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc1.weight)

        self.Fc2 = nn.Linear(hidden_sizes_dyn, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc2.weight)
     
        self.delta = nn.Linear(hidden_sizes_dyn, obs_dim)
        nn.init.xavier_normal_(self.delta.weight)
    def forward(self, obs, act):
        net_out = F.relu(self.Fc1(torch.cat([obs, act], dim=-1)))
        net_out = F.relu(self.Fc2(net_out))

        delta = self.delta(net_out)
        return delta

class DynamicsModel2(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes_dyn):
        super().__init__()
        self.Fc1 = nn.Linear(obs_dim + act_dim, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc1.weight)

        self.Fc2 = nn.Linear(hidden_sizes_dyn, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc2.weight)
     
        self.delta = nn.Linear(hidden_sizes_dyn, obs_dim)
        nn.init.xavier_normal_(self.delta.weight)
    def forward(self, obs, act):
        net_out = F.relu(self.Fc1(torch.cat([obs, act], dim=-1)))
        net_out = F.relu(self.Fc2(net_out))

        delta = self.delta(net_out)
        return delta

class DynamicsModel3(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes_dyn):
        super().__init__()
        self.Fc1 = nn.Linear(obs_dim + act_dim, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc1.weight)

        self.Fc2 = nn.Linear(hidden_sizes_dyn, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc2.weight)
     
        self.delta = nn.Linear(hidden_sizes_dyn, obs_dim)
        nn.init.xavier_normal_(self.delta.weight)
    def forward(self, obs, act):
        net_out = F.relu(self.Fc1(torch.cat([obs, act], dim=-1)))
        net_out = F.relu(self.Fc2(net_out))

        delta = self.delta(net_out)
        return delta

class DynamicsModel4(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes_dyn):
        super().__init__()
        self.Fc1 = nn.Linear(obs_dim + act_dim, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc1.weight)

        self.Fc2 = nn.Linear(hidden_sizes_dyn, hidden_sizes_dyn)
        nn.init.xavier_normal_(self.Fc2.weight)
     
        self.delta = nn.Linear(hidden_sizes_dyn, obs_dim)
        nn.init.xavier_normal_(self.delta.weight)
    def forward(self, obs, act):
        net_out = F.relu(self.Fc1(torch.cat([obs, act], dim=-1)))
        net_out = F.relu(self.Fc2(net_out))

        delta = self.delta(net_out)
        return delta
# <<< Added by Gaffar <<<


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes_dyn= 500, 
                 hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        device = torch.device("cuda:0")

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.dyn_delta0 = DynamicsModel0(obs_dim, act_dim, hidden_sizes_dyn).to(device)
        self.dyn_delta1 = DynamicsModel1(obs_dim, act_dim, hidden_sizes_dyn).to(device)
        self.dyn_delta2 = DynamicsModel2(obs_dim, act_dim, hidden_sizes_dyn).to(device)
        self.dyn_delta3 = DynamicsModel3(obs_dim, act_dim, hidden_sizes_dyn).to(device)
        self.dyn_delta4 = DynamicsModel4(obs_dim, act_dim, hidden_sizes_dyn).to(device)
 
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

    def get_next_state(self, obs, act):
        with torch.no_grad():
            delta0 = self.dyn_delta0(obs, act)
            '''
            delta1 = self.dyn_delta1(obs, act)
            delta2 = self.dyn_delta2(obs, act)
            delta3 = self.dyn_delta3(obs, act)
            delta4 = self.dyn_delta4(obs, act)
            delta = sum([delta0, delta1, delta2, delta3, delta4]) / 5.0
            '''
            next_state = delta0 + obs
            return next_state.cpu().data.numpy()  
