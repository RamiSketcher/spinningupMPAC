## Added by Rami >> ##
## << Added by Rami ##

# Imports
## Basic Python Imports
from copy import deepcopy
import itertools
import time

## ML & RL Impoorts
import math
import numpy as np
import torch
from torch.optim import Adam # {Adanm, SGD, RMSprop, ...}
import gym

## Added by Rami >> ##
# PyBullet Imports
# import pybullet_envs
# import pybullet as p
# p.connect(p.DIRECT)

## SpinningUp Imports
import spinup.algos.pytorch.memb_pe.core as core
from spinup.utils.logx import EpochLogger
## << Added by Rami ##

from spinup.pddm_envs.gym_env import GymEnv


device = torch.device("cpu")



class ReplayBuffer: # No changes
    """
    The replay buffer used to uniformly sample the data
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
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}


"""
Model Embedding Model Based Algorithm (MEMB)
(with TD3 style Q value function update)
"""
# InvertedPen
def memb_pe(
        env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), model=core.MLPModel,
        seed=0, steps_per_epoch=4000, epochs=125, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, dyn_lr=3e-4, rew_lr=3e-4, value_lr=1e-3, pi_lr=3e-4, alphai=0.4,
        Gsteps=120, batch_size=100, start_steps=1000,
        max_ep_len=40, save_freq=1, update_every=40,
        train_model_epoch=5, test_freq=2, num_tests=5, save_epoch=100,
        exp_name='', env_name='',
        logger_kwargs=dict()
        ):


    ## Added by Rami >> ##
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    ## << Added by Rami ##

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape #@!
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]


    # Create actor-critic module and target networks
    # [pi, q1, q2, v or v')] = MLPActorCritic(obs_space, act_space)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # we need a separate target network; bc it's optmz differnetly
    # [_, _, _, v_targ] = MLPActorCritic(obs2_space, act_space)
    ac_targ = deepcopy(ac)
    # Create model module
    # [transiton, reward] = MLPModel(obs_space, act_space)
    md = model(env.observation_space, env.action_space)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for all Value-networks (save this for convenience)
    # val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())

    # List of parameters for all Model-networks (save this for convenience)
    # md_params = itertools.chain(md.delta.parameters(), md.reward.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    ## Added by Rami >> ##
    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [md, ac])
    logger.log('\nNumber of parameters: \t totalModel: %d, \t totalAC: %d\n'%var_counts)
    ## << Added by Rami ##


    # TD3 style Q function updates #

    ## Optimized costs\losses ##
    
    # Set up function for computing Rew/Dyn model-losses
    ### Model/Reward losses (supervised learning):
    #   loss = 0.5*(actual-prediction)^2 }
    #       Jp(omega) = 0.5 Expt_D[(f(s,a)-s')^2] --> eq#4.a
    #       Jr(ph) = Expt_D[(r(s,a)-r)^2] --> eq#4.b    
    #           min_omeg,ph{ Jp(omeg), Jr(ph) }

    def compute_loss_dyn(data): # Rami (Done)

        o, a, o2 = data['obs'], data['act'], data['obs2']
        
        delta_backup = o2 - o
        loss_delta1 = ((delta_backup - md.delta.delta1(o,a))**2).mean()
        loss_delta2 = ((delta_backup - md.delta.delta2(o,a))**2).mean()
        loss_delta3 = ((delta_backup - md.delta.delta3(o,a))**2).mean()
        loss_dyn = (loss_delta1 + loss_delta2 + loss_delta3)/3

        # Useful info for logging
        dyn_info = dict(
                        LossDyn=loss_dyn.item(),
                        # LossDelta1=loss_delta1.item(),
                        # LossDelta2=loss_delta2.item(),
                        # LossDelta3=loss_delta3.item(),
                        )
        
        return loss_dyn, dyn_info

    def compute_loss_rew(data): # Rami (Done)

        o, a, r = data['obs'], data['act'], data['rew']
        
        r_backup = r
        loss_r1 = ((r_backup - md.reward.reward1(o,a))**2).mean()
        # loss_r2 = ((r_backup - md.reward.reward2(o,a))**2).mean()
        # loss_r3 = ((r_backup - md.reward.reward3(o,a))**2).mean()
        # loss_rew = (loss_r1 + loss_r2 + loss_r3)/3
        loss_rew = loss_r1

        # Useful info for logging
        rew_info = dict(
                        LossRew=loss_rew.item(),
                        # LossR1=loss_r1.item(),
                        # LossR2=loss_r2.item(),
                        # LossR3=loss_r3.item()
                        )
        
        return loss_rew, rew_info


    # Set up function for computing pi loss
    ### Policy loss ###
    #   State value-function of st:
    #       V(st) = Expt_pi[Q(st,at) - log pi(at|st)] --> eq#3.b,
    #   Policy learning's Soft Bellman eq (Reparameterization):
    #       V(s) = Expt_pi[Expt_rm[r_hat(s,pi]
    #                       - alpha*log pi(a|s)
    #                       + gamma*Expt_f[V'(f(s,pi))]] --> eq#8
    #   Optz pi--> max_pi{ Expt_s~D[V(s)] }
    def compute_loss_pi(data, alpha): # Rami (Done)

        o = data['obs']

        pi, logp_pi = ac.pi(o)

        o_next_pi = md.delta(o, pi) + o
        r_rm_pi = md.reward(o,pi)
        v_prime = ac.v(o_next_pi)

        # Entropy-regularized policy loss
        loss_pi = -(r_rm_pi - alpha*logp_pi + gamma*(1-d)*v_prime).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().data.numpy())

        return loss_pi, pi_info

    # Set up function for computing Q,V value-losses
    ### Value functions losses ###
    #   Optz--> min_phi,psi{ Jq(phi),Jv(psi) }
    def compute_loss_val(data, alpha): # Rami (Done)
        
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # pi, logp_pi = ac.pi(o)
        
        # Optimizesd functions
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        v = ac.v(o)

        # Bellman backup for Value functions
        with torch.no_grad():
            # Target value function
            pi, logp_pi = ac.pi(o)
            v_targ = ac_targ.v(o2)

            q1_pi = ac.q1(o,pi)
            q2_pi = ac.q2(o,pi)
            min_q_pi = torch.min(q1_pi, q2_pi)

            q_backup = r + gamma * (1 - d) * v_targ # By Rami
            v_backup = min_q_pi - alpha * logp_pi # By Rami

        # MSE loss against Bellman backup
        loss_q1 = ((q_backup - q1)**2).mean()
        loss_q2 = ((q_backup - q2)**2).mean()
        loss_v = ((v_backup - v)**2).mean()
        loss_val = (loss_q1 + loss_q2 + loss_v)/3

        # Useful info for logging
        val_info = dict(Q1Vals=q1.detach().cpu().data.numpy(),
                        Q2Vals=q2.detach().cpu().data.numpy(),
                        V_Vals=v.detach().cpu().data.numpy())

        return loss_val, val_info
    
    # Set up optimizers for model, policy and value-functions
    dyn_optimizer = Adam(md.delta.parameters(), lr=dyn_lr) # Rami
    rew_optimizer = Adam(md.reward.parameters(), lr=rew_lr) # Rami

    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())
    val_optimizer = Adam(val_params, lr=value_lr) # Rami


    # Set up model saving
    logger.setup_pytorch_saver(ac)


    def updateAC(data, alpha): # Rami (Done)

        # First run one gradient descent step for Q1, Q2, and V
        val_optimizer.zero_grad()
        loss_val, val_info = compute_loss_val(data, alpha)
        loss_val.backward() # Descent
        val_optimizer.step()

        # Record things
        logger.store(LossVal=loss_val.item(), **val_info)

        # Freeze Value-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())
        for p in val_params:
            p.requires_grad = False
        # Freeze Model-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in md.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, alpha)
        loss_pi.backward() # Ascent of +ve Jpi or Descent of -ve Jpi
        pi_optimizer.step()

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        
        # Unfreeze Value-networks so you can optimize it at next Update step.
        val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())
        for p in val_params:
            p.requires_grad = True
        # Unfreeze Value-networks so you can optimize it at Model Update step.
        for p in md.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)



    def updateDyn(data): # Rami (Done)

        # print("Model updating..")
        # Run one gradient descent step for model
        dyn_optimizer.zero_grad()
        loss_dyn, dyn_info = compute_loss_dyn(data)
        loss_dyn.backward() # Descent
        dyn_optimizer.step()

        # Record things
        logger.store(**dyn_info)

    def updateRew(data): # Rami (Done)

        # print("Model updating..")
        # Run one gradient descent step for model
        rew_optimizer.zero_grad()
        loss_rew, rew_info = compute_loss_rew(data)
        loss_rew.backward() # Descent
        rew_optimizer.step()

        # Record things
        logger.store(**rew_info)




    def get_action(o, deterministic=False): # Rami (Done)
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), 
                    deterministic)


    def test_agent(): # (Done)
        total_reward = 0
        for j in range(num_tests): # repeat num_tests=5 times
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                a = get_action(o, True)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            total_reward += ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len) ## By Rami

        return total_reward/num_tests


    start_time = time.time() ## Rami
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    reward_recorder = []



    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        The algorithm would take total_steps totally in the training
        """

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        if t >= start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample() # Random for 1k (epoch 1)

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d # Don't let the env done if just reach max_ep_length

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical!, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling if [(env is done) or (max_ep_legth reached)]
        if d or (ep_len == max_ep_len):
            
            ## Added by Rami >> ##
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## << Added by Rami ##
            
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


        # Learning/Training
        #   Train pi, Q, and V after 5 epochs for 5 times,
        #   Train dyn/rew models from start:
        if t >= (start_steps) and t % update_every == 0:
            # Train "ACsteps" steps of Q, V, and pi,
            # then train 1 step of model.
            alpha = alphai*math.exp(-4*((t-start_steps)/total_steps))
            for j in range(Gsteps):
                batch = replay_buffer.sample_batch(batch_size)
                updateAC(data=batch, alpha=alpha)
            updateDyn(data=batch)
            updateRew(data=batch)

        else:
            # pretrain the model
            batch = replay_buffer.sample_batch(batch_size)
            updateDyn(data=batch)
            updateRew(data=batch)


        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            # epoch = t // steps_per_epoch
            epoch = (t+1) // steps_per_epoch

            ## Added by Rami >> ##
            # Save model after each epoch:
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
            ## << Added by Rami ##

            test_agent()

            ## Added by Rami >> ##
            logger.log_tabular('Epoch', epoch)

            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)

            logger.log_tabular('TestEpRet', with_min_and_max=True) # if n=1 no variance
            logger.log_tabular('TestEpLen', average_only=True)

            logger.log_tabular('TotalEnvInteracts', t)
            
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)

            # logger.log_tabular('Q1Vals', with_min_and_max=True) 
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LossQ1', average_only=True)
            # logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('V_Vals', with_min_and_max=True)
            # logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossVal', average_only=True)

            ## Added by Rami >> ##
            logger.log_tabular('LossDyn', average_only=True)
            logger.log_tabular('LossRew', average_only=True)
            ## << Added by Rami ##

            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            ## << Added by Rami ##




if __name__ == '__main__': # (Done)
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='memb')

    parser.add_argument('--train_model_epoch', type=int, default=5) # Train f(s,a) after 5 epochs
    parser.add_argument('--test_freq', type=int, default=1)
    # parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--save_epoch', type=int, default=500)

    args = parser.parse_args()

    ## Added by Rami >> ##
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    torch.set_num_threads(torch.get_num_threads())
    ## << Added by Rami ##

    # I = 1
    # for i in range(0,I):
    #     # repeat I times of experiment
    memb_pe(lambda : gym.make(args.env),

                actor_critic=core.mlp_actor_critic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                model=core.MLPModel,

                gamma=args.gamma,
                seed=args.seed,
                epochs=args.epochs, # 200

                # save_freq=i, # 0,1,2,3,4
                train_model_epoch=args.train_model_epoch, # 5
                test_freq=args.test_freq, # 10

                exp_name=args.exp_name,
                env_name=args.env_name,
                # save_epoch=args.save_epoch,
                
                logger_kwargs=logger_kwargs)

