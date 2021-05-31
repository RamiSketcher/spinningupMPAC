# (Rami) Modified
# (Rami & Gaffar) Done 1st reading, Feb 22, 2020


# Imports
## ML & RL Impoorts
import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda:0")

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
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)

        # create value functions, Q_phi(s,a) , Qpi_phi(s,pi(a|s)) and V_psi(s), for
        ## Jv(psi) = Expt_st~D[0.5*( V_psi(st) - Expt_at~pi[Qpi_phi(st,pi(at|st))-logpi(at|st)] )^2] -->eq#5
        ## Jq(phi) = Expt_(st,at)~D[0.5 ( Q_phi(st,at) - rt - lamda*V_psi(st+1) )^2] -->eq#6

        # value(s) = NN(x, unit: [hid_list]+1, act, out_act):
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.v = MLPVFunction(obs_dim, hidden_sizes, activation).to(device) # For V, V_targ, V'    Rami

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs.to(device), deterministic, False)
            return a.cpu().numpy()



"""
Reward and Transition Dynamic Models,
the hidden_size for each model could be adjusted in each subnetworks.
"""

## Model Network ##
class MLPDynModel(nn.Module): # Rami (Done)

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], obs_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], obs_dim)

    def forward(self, obs, act, deterministic=False, with_logprob=True):

        # st+1 ~ f(st+1|st,at;omega) = N(mu,std|st,at;omega)
        net_out = self.net(torch.cat([obs, act], dim=-1))

        mu = self.mu_layer(net_out)

        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) 

        # Transition distribution 
        delta = Normal(mu, std)

        return delta.rsample()

class MLPDynModelEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.delta1 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        self.delta2 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        self.delta3 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.delta4 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.delta5 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)

    def forward(self, obs, act, deterministic=False, with_logprob=True):
        delta1 = self.delta1(obs, act)
        delta2 = self.delta2(obs, act)
        delta3 = self.delta3(obs, act)
        # delta4 = self.delta4(obs, act)
        # delta5 = self.delta5(obs, act)
        delta = (delta1 + delta2 + delta3)/3
        return delta
        

class MLPRewModel(nn.Module): # Rami (Done)

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.net = mlp([obs_dim + act_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], 1)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs, act, deterministic=False, with_logprob=True):

        # rt ~ R(rt|st,at;eqsi) =  N(mu,std|st,at;ph)
        net_out = self.net(torch.cat([obs, act], dim=-1))

        mu = self.mu_layer(net_out)

        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) 

        # Reward distribution
        reward = Normal(mu, std)

        return reward.rsample()

class MLPRewModelEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, output_activation=None):
        super().__init__()
        self.reward1 = MLPRewModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.reward2 = MLPRewModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.reward3 = MLPRewModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.delta4 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)
        # self.delta5 = MLPDynModel(obs_dim, act_dim, hidden_sizes, activation, output_activation)

    def forward(self, obs, act, deterministic=False, with_logprob=True):
        reward1 = self.reward1(obs, act)
        # reward2 = self.reward2(obs, act)
        # reward3 = self.reward3(obs, act)
        # delta4 = self.delta4(obs, act)
        # delta5 = self.delta5(obs, act)
        # reward = (reward1 + reward2 + reward3 + reward4 + reward5)/5
        # reward = (reward1 + reward2 + reward3)/3
        reward = reward1
        return reward



class MLPModel(nn.Module): # Rami (Done)

    def __init__(self,
                 observation_space, action_space,
                 hidden_sizes=(256,128),
                 activation=nn.ReLU,
                 output_activation=None):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]


        self.delta = MLPDynModelEnsemble(obs_dim, act_dim, hidden_sizes, activation, output_activation).to(device)

        self.reward = MLPRewModelEnsemble(obs_dim, act_dim, hidden_sizes, activation, output_activation).to(device)


