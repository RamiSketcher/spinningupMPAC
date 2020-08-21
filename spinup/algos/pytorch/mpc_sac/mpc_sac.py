"""
OpenAI: Original from spinningup/spinup/algos/pytorch/sac/core.py,
(Gaffar): Added Ensemles of Dynamics Model,
(Rami): Modefied
"""

from copy import deepcopy
import itertools
import sys
import math 
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import gym
import time
#from spinup.algos.pytorch.sac.envs.gym_env import GymEnv
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
#from tensorboardX import SummaryWriter
#import multiprocessing as mp
import torch.multiprocessing as mp
import contextlib


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

'''
seeds:
12542
14063
19432
'''

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


class DynReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, seed = 1000, j=0):
        device = torch.device("cuda:0")
        k = j * batch_size
        idxs = []
        for i in range(batch_size):
            idxs.append(k)
            k += 1
        np.array(idxs)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}






def mpc_sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=1000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2,
        batch_size=100, start_steps=2000, 
        update_after=1000, update_every=50,
        num_test_episodes=10, max_ep_len=200, 
        logger_kwargs=dict(), save_freq=1):
    """
    Model Predictive Control: Soft Actor-Critic (MPC-SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    name = logger_kwargs['exp_name']
    #writer = SummaryWriter(comment="mpc-sac_" + name + "Policy")
    #writer_mpc = SummaryWriter(comment = "mpc-sac_" + name + "MPC")
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    device = torch.device("cuda:0")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    Dyn_epochs = 100

    env, test_env, eval_env = env_fn(), env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    dyn_model0 = ac.dyn_delta0
    dyn_model1 = ac.dyn_delta1    
    dyn_model2 = ac.dyn_delta2
    dyn_model3 = ac.dyn_delta3
    dyn_model4 = ac.dyn_delta4

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Experience buffer for dynamics model
    dyn_buffer = DynReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(5e5))
   

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
        
    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        
        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    dyn_optimizer0 = Adam(ac.dyn_delta0.parameters(), lr=lr)
    dyn_optimizer1 = Adam(ac.dyn_delta1.parameters(), lr=lr)
    dyn_optimizer2 = Adam(ac.dyn_delta2.parameters(), lr=lr)
    dyn_optimizer3 = Adam(ac.dyn_delta3.parameters(), lr=lr)
    dyn_optimizer4 = Adam(ac.dyn_delta4.parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

        

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def compute_loss_dyn_norm(dyn_data, norm_data):
        data = dyn_data[0]
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        mu_o, std_o, mu_a, std_a, mu_delta, std_delta = \
        norm_data[0], norm_data[1], norm_data[2], norm_data[3], norm_data[4], norm_data[5]
        norm_o = (o - mu_o) / std_o
        norm_a = (a - mu_a) / std_a
        delta = o2 - o
        norm_delta = (delta - mu_delta) / std_delta
        dyn_delta0 = ac.dyn_delta0(norm_o, norm_a)
        dyn_delta1 = ac.dyn_delta1(norm_o, norm_a)
        dyn_delta2 = ac.dyn_delta2(norm_o, norm_a)
        dyn_delta3 = ac.dyn_delta3(norm_o, norm_a)
        dyn_delta4 = ac.dyn_delta4(norm_o, norm_a)
        loss_dyn0 = ((norm_delta - dyn_delta0)**2).mean()
        loss_dyn1 = ((norm_delta - dyn_delta1)**2).mean()
        loss_dyn2 = ((norm_delta - dyn_delta2)**2).mean()
        loss_dyn3 = ((norm_delta - dyn_delta3)**2).mean()
        loss_dyn4 = ((norm_delta - dyn_delta4)**2).mean()
        loss_dyn = [loss_dyn0, loss_dyn1, loss_dyn2, loss_dyn3, loss_dyn4]
        loss_all = sum(loss_dyn) / 5.0
        #loss_ens3 = [loss_dyn0, loss_dyn1, loss_dyn2]
        #loss_ens3 = sum(loss_ens3) / 3.0
        return loss_dyn, loss_all



    def update_dyn(dyn_data, i, norm_data,t, j, k):

        loss_dyn, loss_all  = compute_loss_dyn_norm(dyn_data, norm_data)

        loss_final0 = 0

        dyn_optimizer0.zero_grad()
        loss_dyn0 = loss_dyn[0]
        loss_dyn0.backward()
        dyn_optimizer0.step()

        
        dyn_optimizer1.zero_grad()
        loss_dyn1 = loss_dyn[1]
        loss_dyn1.backward()
        dyn_optimizer1.step()


        dyn_optimizer2.zero_grad()
        loss_dyn2 = loss_dyn[2]
        loss_dyn2.backward()
        dyn_optimizer2.step()

        
        dyn_optimizer3.zero_grad()
        loss_dyn3 = loss_dyn[3]
        loss_dyn3.backward()
        dyn_optimizer3.step()

        dyn_optimizer4.zero_grad()
        loss_dyn4 = loss_dyn[4]
        loss_dyn4.backward()
        dyn_optimizer4.step()
        
        #logger.store(LossDyn=loss_ens3.item())
        #logger.store(LossEns5=loss_all.item())
        if i == (Dyn_epochs - 1) and (j+1) == k:
            #print(loss_dyn0.item())
            writer.add_scalar("Loss0", loss_dyn0.item(), t)
            writer.add_scalar("Loss1", loss_dyn1.item(), t)
            writer.add_scalar("Loss2", loss_dyn2.item(), t)
            writer.add_scalar("Loss3", loss_dyn3.item(), t)
            writer.add_scalar("Loss4", loss_dyn4.item(), t)
            writer.add_scalar("Loss All", loss_all.item(), t)        
  
    def weight_reset(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()


    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                if type(d) is not bool:
                    d = d.astype(bool)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def test_policy(count = 4):
        policy_ret = 0
        policy_steps = 0
        policy_scores = 0.0
        for j in range(count):
            o, d, ep_ret, ep_len, scores = eval_env.reset(), False, 0, 0, 0.0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o2, r, d, _ = eval_env.step(get_action(o, True))
                score = eval_env.get_score(o2)
                o = o2 
                if type(d) is not bool:
                    d = d.astype(bool)
                ep_ret += r
                ep_len += 1
                scores += score
            policy_ret += ep_ret
            policy_steps += ep_len
            policy_scores += scores
        policy_scores /= policy_steps
        logger.store(PolicyEpRet=policy_ret / count, PolicyEpLen = policy_steps/count,\
        PolicyScore=policy_scores/count)
        return policy_ret / count, ep_len / count, policy_scores/count


    def get_reward(env, o, a):
        return env.get_reward(o, a)

    def get_next_state(o, a, norm_data):
        with torch.no_grad():
            o = torch.as_tensor(o, dtype=torch.float32).to(device)
            a = torch.as_tensor(a, dtype=torch.float32).to(device)
            mu_o, std_o, mu_a, std_a, mu_delta, std_delta = \
            norm_data[0], norm_data[1], norm_data[2], norm_data[3], norm_data[4], norm_data[5]
            norm_o = (o - mu_o) / std_o
            norm_a = (a - mu_a) / std_a
            dyn_delta = ac.dyn_delta0(norm_o, norm_a)
            dyn_delta += ac.dyn_delta1(norm_o, norm_a)
            dyn_delta += ac.dyn_delta2(norm_o, norm_a)
            dyn_delta += ac.dyn_delta3(norm_o, norm_a)
            dyn_delta += ac.dyn_delta4(norm_o, norm_a)
            dyn_delta = dyn_delta / 5.0
            dyn_pred = (dyn_delta * std_delta) + mu_delta
            next_state = dyn_pred + o
            return next_state.cpu().data.numpy()

    def eval_dyn(env, norm_data):
        steps = 0
        eval_loss = 0
        o = env.reset()
        while True:
            a = get_action(o)
            o2d = get_next_state(o, a, norm_data)
            o2, reward, done, _ = env.step(a)
            if type(done) is not bool:
                done = done.astype(bool)
            eval_loss += ((o2d - o2)**2).mean()
            o = o2
            steps += 1
            if done:
                break
        return eval_loss / steps

    def get_mpc_action(o, norm_data, env = env , N_Trajectories = 1500, H = 20, E = 10):
        action_dim = env.action_space.shape[0]
        ot = []
        for _ in range(N_Trajectories):
            ot.append(o)
        ot = np.array(ot)
        a = get_action(ot)
        r, _ = env.get_reward(ot, a)
        o2 = get_next_state(ot, a, norm_data)
        for h in range(H):
            a2 = get_action(o2)
            next_r, _ = get_reward(env, o2, a2) 
            r += (next_r * (0.99 **(h+1)))
            o2 = get_next_state(o2, a2, norm_data)
        act_rew = np.column_stack((a, r))
        act_rew = act_rew[act_rew[:,action_dim].argsort()]
        best_action_values = act_rew[-E:]
        best_actions = sum(best_action_values) / E
        best_action = best_actions[:action_dim]
        return best_action


    def MPCPlannerOpt(env, norm_data, N_Trajectories = 1500, count=10, H = 20, \
    update = False, E = 10):
        action_dim = env.action_space.shape[0]
        mpc_scores = 0.0
        EpRet = 0.0
        EpLen = 0.0
        for _ in range(count):
            rewards = 0.0
            scores = 0.0
            steps = 0.0
            o = env.reset()
            #print(o)
            while True:
                ot = []
                for _ in range(N_Trajectories):
                    ot.append(o)
                ot = np.array(ot)
                a = get_action(ot)
                r, _ = env.get_reward(ot, a)
                o2 = get_next_state(ot, a, norm_data)
                for h in range(H):
                    a2 = get_action(o2)
                    next_r, _ = get_reward(env, o2, a2) 
                    r += (next_r * (0.99 **(h+1)))
                    o2 = get_next_state(o2, a2, norm_data)
                act_rew = np.column_stack((a, r))
                act_rew = act_rew[act_rew[:,action_dim].argsort()]
                best_action_values = act_rew[-E:]
                best_actions = sum(best_action_values) / E
                best_action = best_actions[:action_dim]
                o2, reward, done, _ = env.step(best_action)
                score = env.get_score(o2)
                if type(done) is not bool:
                    done = done.astype(bool)

                if update:
                    replay_buffer.store(o, best_action, reward, o2, done)

                    dyn_buffer.store(o, best_action, reward, o2, done)
                rewards += reward
                scores += score
                steps += 1
                o = o2
                if done or (steps==max_ep_len):
                    break
            EpRet += rewards
            EpLen += steps
            mpc_scores += scores
            print(rewards)
            sys.stdout.flush()
        print(EpRet/count)
        mpc_scores /= EpLen 
        logger.store(MPCEpRet=EpRet / count, MPCEpLen = EpLen/count,\
        MPCScore=mpc_scores/count)
        return EpRet / count, EpLen / count, mpc_scores/count 

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
   
    def stats_calc(batch):
        obs_test = batch['obs']
        act_test = batch['act']
        obs2_test = batch['obs2']
        delta = obs2_test - obs_test
        mu_obs = torch.mean(obs_test, 0, True).cpu().data.numpy()
        mu_obs[mu_obs == 0] = 0.000001
        std_obs = torch.std(obs_test, 0).cpu().data.numpy()
        std_obs[std_obs == 0] = 0.000001
        mu_act = torch.mean(act_test, 0, True)#\.cpu().data.numpy()
        std_act = torch.std(act_test, 0)#\.cpu().data.numpy()
        mu_delta = torch.mean(delta, 0, True).cpu().data.numpy()
        mu_delta[mu_delta == 0] = 0.000001
        std_delta = torch.std(delta, 0).cpu().data.numpy()
        std_delta[std_delta == 0] = 0.000001
        mu_obs = torch.as_tensor(mu_obs, dtype=torch.float32).to(device)
        std_obs = torch.as_tensor(std_obs, dtype=torch.float32).to(device)
        mu_delta = torch.as_tensor(mu_delta, dtype=torch.float32).to(device)
        std_delta = torch.as_tensor(std_delta, dtype=torch.float32).to(device)
        norm_data = [mu_obs, std_obs, mu_act, std_act, mu_delta, std_delta]
        return norm_data
    
    update_counter = 0
    updating = True
    
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if updating and t % 4000 == 0 and (t+1) > 20000:
            update_counter += 1
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        if update_counter != 0:
            update_counter += 1 
        if t > start_steps:
            if update_counter > 0:
                if (t+1) > 500000:
                    x = 500000
                else:
                    x = t
                batch_norm = dyn_buffer.sample_batch(int(x))
                norm_data = stats_calc(batch_norm)
                a = get_mpc_action(o, norm_data, N_Trajectories = 1500, H = 10, E = 10)
            else:
                a = get_action(o)
        else:
            a = env.action_space.sample()
            

        # Step the env
        o2, r, d, _ = env.step(a)
        if t == 1:
            print(d)
            print(type(d))
        if type(d) is not bool:
            d = d.astype(bool)
        if t == 2:
            print(o2)
            print(r)
            print(d)
            
        ep_ret += r
        ep_len += 1

        if (t+1) % 1000 == 0:
            print(1000)
        
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d
      
        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        dyn_buffer.store(o, a, r, o2, d)
       
        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2
      
        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            #writer.add_scalar("Reward", ep_ret, t)
            #writer.add_scalar("Steps", ep_len, t)
            o, ep_ret, ep_len = env.reset(), 0, 0


        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

       
        if (update_counter + 1) == 2000:
            update_counter = 0


        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch



            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

 
            # Test the performance of the deterministic version of the agent.
            test_agent()

            policy_rewards, policy_steps, _ = test_policy(count = 5)
            writer.add_scalar("Reward", policy_rewards, t)
            writer.add_scalar("Steps", policy_steps, t)            


            if (t+1) < 100000:
                Dyn_epochs = 120

            else: 
                Dyn_epochs = 60


            
            print(Dyn_epochs)
            # Update the dynamics model
            #if (t+1) % 4000== 0:
            if (t+1) % 4000 == 0:  
                dyn_model0.apply(weight_reset)
                dyn_model1.apply(weight_reset)
                dyn_model2.apply(weight_reset)
                dyn_model3.apply(weight_reset)
                dyn_model4.apply(weight_reset)
                if (t+1) > 500000:
                    x = 500000
                else:
                    x = t
                batch_norm = dyn_buffer.sample_batch(int(x))
                norm_data = stats_calc(batch_norm)
                k = math.floor(x / 500)
                for i in range(Dyn_epochs):
                    for j in range(k):
                        batch_dyn = dyn_buffer.sample_batch(500, j = j)
                        dyn_data = [batch_dyn]
                        update_dyn(dyn_data= dyn_data, i=i, norm_data=norm_data, t = t, j =j, k = k)
                eval_loss = eval_dyn(eval_env, norm_data)
                writer.add_scalar("Dynamics Model Evaluation Error", eval_loss, t)

            
            
            #if (t+1) % 4000 == 0:
            if (t+1) % 4000 == 0 :
                rewards, steps, _ = MPCPlannerOpt(eval_env, norm_data, \
                N_Trajectories=1500, H = 10,  count = 5, update = False, E = 10)
                writer_mpc.add_scalar("Reward", rewards, t)
                writer_mpc.add_scalar("Steps", steps, t)
            
            
            


            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('PolicyEpRet', average_only=True)
            logger.log_tabular('PolicyScore', average_only=True)
            logger.log_tabular('PolicyEpLen', average_only=True)
            logger.log_tabular('MPCEpRet', average_only=True)
            logger.log_tabular('MPCScore', average_only=True)
            logger.log_tabular('MPCEpLen', average_only=True)
            #logger.log_tabular('LossDyn')
            #logger.log_tabular('LossEns5')
            logger.dump_tabular()
    writer.close()
    writer_mpc.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='mpc-sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    mpc_sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
