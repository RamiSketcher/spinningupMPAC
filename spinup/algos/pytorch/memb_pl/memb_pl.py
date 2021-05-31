## Added by Rami >> ##
## << Added by Rami ##

# Imports
## Basic Python Imports
from copy import deepcopy
import itertools
import time

## ML & RL Impoorts
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
from spinup.utils.logx import colorize
## << Added by Rami ##

from spinup.pddm_envs.gym_env import GymEnv



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
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


"""
Model Embedding Model Based Algorithm (MEMB-PE)
(with TD3 style Q value function update)
"""
# InvertedPen
def memb_pl(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), Model=core.MLPModel,
        seed=0, steps_per_epoch=1000, epochs=400, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, model_lr=3e-4, value_lr=1e-3, pi_lr=3e-4, alpha=0.4,
        batch_size=100, start_steps=1000,
        max_ep_len=40, save_freq=1,# max_ep_len for locomotion = 1k, for DClaw = 40
        train_model_epoch=5, test_freq=2, save_epoch=100,
        exp_name='', env_name='',
        logger_kwargs=dict()):


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
    # act_limit = env.action_space.high[0]


    # Create actor-critic module and target networks
    # [pi, q1, q2, v or v')] = MLPActorCritic(obs_space, act_space)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # we need a separate target network; bc it's optmz differnetly
    # [_, _, _, v_targ] = MLPActorCritic(obs2_space, act_space)
    ac_targ = deepcopy(ac)
    # Create model module
    # [transiton, reward] = MLPModel(obs_space, act_space)
    # md = model(env.observation_space, env.action_space, model_lr)
    model = Model(env.observation_space, env.action_space, model_lr)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for all Value-networks (save this for convenience)
    val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())

    # List of parameters for all Model-networks (save this for convenience)
    # model_params = itertools.chain(model.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    ## Added by Rami >> ##
    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['main/dm', 'main/rm', 'main/pi', 'main/v', 'main/q1', 'main/q2', 'main'])
    var_counts = tuple(core.count_vars(module) for module in [model.delta, model.reward, ac.pi, ac.q1, ac.q2, ac.v, ac])
    # print('\nNumber of parameters: \t dm: %d, \t rm: %d, \t pi: %d, \t v: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)
    logger.log('\nNumber of parameters: \t dm: %d, \t rm: %d, \t pi: %d, \t q1: %d, \t q2: %d, \t v: %d, \t total_ac: %d\n'%var_counts)
    ## << Added by Rami ##


    # TD3 style Q function updates #

    ## Optimized costs\losses ##

    # o, a, r, o2, d = data['obs'],
    #                  data['act'],
    #                  data['rew'],
    #                  data['obs2'],
    #                  data['done']
    
    # Set up function for computing Rew/Dyn model-losses
    ### Model/Reward losses (supervised learning):
    #   loss = 0.5*(actual-prediction)^2 }
    #       Jp(omega) = 0.5 Expt_D[(f(s,a)-s')^2] --> eq#4.a
    #       Jr(ph) = Expt_D[(r(s,a)-r)^2] --> eq#4.b    
    # #           min_omeg,ph{ Jp(omeg), Jr(ph) }
    def compute_loss_model(data): # Rami (Done)

        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2']
        
        delta = model.delta(o,a)
        transition = o + delta
        r_rm = model.reward(o,a)

        delta_backup = o2 - o
        r_backup = r
        
        loss_delta = ((delta_backup - delta)**2).mean()
        loss_r = ((r_backup-r_rm)**2).mean()
        loss_model = loss_delta + loss_r

        # Useful info for logging
        model_info = dict(Dyn=transition.detach().numpy(),
                          Rew=r_rm.detach().numpy())
        
        return loss_model, model_info



    # Set up function for computing pi loss
    ### Policy loss ###
    #   State value-function of st:
    #       V(st) = Expt_pi[Q(st,at) - log pi(at|st)] --> eq#3.b,
    #   Policy learning's Soft Bellman eq (Reparameterization):
    #       V(s) = Expt_pi[Expt_rm[r_hat(s,pi]
    #                       - alpha*log pi(a|s)
    #                       + gamma*Expt_f[V'(f(s,pi))]] --> eq#8
    #   Optz pi--> max_pi{ Expt_s~D[V(s)] }
    def compute_loss_pi(data): # Rami (Done)

        o = data['obs']

        pi, logp_pi = ac.pi(o)

        # delta_pi = md.delta(o,pi)
        delta_pi = model.delta(o,pi)
        transition_pi = o + delta_pi
        # r_rm_pi = md.reward(o,pi)
        r_rm_pi = model.reward(o,pi)
        v_prime = ac.v(transition_pi)

        # Entropy-regularized policy loss
        loss_pi = -(r_rm_pi - alpha*logp_pi + gamma*(1-d)*v_prime).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up function for computing Q,V value-losses
    ### Value functions losses ###
    #   Optz--> min_phi,psi{ Jq(phi),Jv(psi) }
    def compute_loss_val(data): # Rami (Done)
        
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # pi, logp_pi = ac.pi(o)
        
        # Optimizesd functions
        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        v = ac.v(o)

        # q1_pi = ac.q1(o,pi)
        # q2_pi = ac.q2(o,pi)
        # min_q_pi = torch.min(q1_pi, q2_pi)

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
        loss_val = loss_q1 + loss_q2 + loss_v

        # Useful info for logging
        val_info = dict(Q1Vals=q1.detach().numpy(),
                        Q2Vals=q2.detach().numpy(),
                        V_Vals=v.detach().numpy())

        return loss_val, val_info
    
    # Set up optimizers for model, policy and value-functions
    model_optimizer = Adam(model.parameters(), lr=model_lr) # Rami
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    val_optimizer = Adam(val_params, lr=value_lr) # Rami


    # Set up model saving
    logger.setup_pytorch_saver(ac)


    def updateAC(data): # Rami (Done)
        # print("AC updating..")
        # First run one gradient descent step for Q1, Q2, and V
        val_optimizer.zero_grad()
        loss_val, val_info = compute_loss_val(data)
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
        for p in model.parameters():
            p.requires_grad = False


        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        (loss_pi).backward() # Ascent
        pi_optimizer.step()

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        
        # Unfreeze Value-networks so you can optimize it at next Update step.
        val_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters(), ac.v.parameters())
        for p in val_params:
            p.requires_grad = True
        # Unfreeze Value-networks so you can optimize it at Model Update step.
        for p in model.parameters():
            p.requires_grad = True

        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        # print("..AC updated")



    def updateModel(data): # Rami (Done)
        # print("Model updating..")
        # Run one gradient descent step for model
        model_optimizer.zero_grad()
        loss_model, model_info = compute_loss_model(data)
        loss_model.backward() # Descent
        model_optimizer.step()

        # Record things
        logger.store(LossModel=loss_model.item(), **model_info)
        # logger.store(LossRew=loss_model.item(), **model_info)
        # print("..Model updated")



    def get_action(o, deterministic=False): # Rami (Done)
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)


    def test_agent(epoch,n=5): # (Done)
        # global mu, pi, q1, q2, q1_pi, q2_pi
        total_reward = 0
        for _ in range(n): # repeat n=5 times
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            total_reward += ep_ret
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len) ## By Rami

        # print('The '+str(epoch)+' epoch is finished!')
        # print('The test reward is '+str(total_reward/n))
        return total_reward/n


    start_time = time.time() ## Rami
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    # reward_recorder = []



    # Main loop: collect experience in env and update/log each epoch
    e = 1
    for t in range(total_steps):
        """
        The algorithm would take total_steps totally in the training
        """

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        if t > start_steps:
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
        # update_after=1000, update_every=50
        #   Train pi, Q, and V after 5 epochs for 5 times,
        #   Train dyn/rew models from start:
        # if t // steps_per_epoch > train_model_epoch: # if epoch > 5
        # if t >= 2000  and t % 100 == 0:
        #     # Train 5 steps of Q, V, and pi,
        #     # then train 1 step of model.
        #     for _ in range(100):
        #         batch = replay_buffer.sample_batch(batch_size)
        #         updateAC(data=batch)
            
        #     # updateModel(data=batch) # Rami
        #     model.train_model(replay_buffer, t+1, batch_size)
        #     # Record things
        #     # logger.store(LossModel=loss_model.item())
        # elif t % 100 == 0:
        #     # pretrain the model
        #     # batch = replay_buffer.sample_batch(batch_size)
        #     # updateModel(data=batch) # Rami
        #     model.train_model(replay_buffer, t+1, batch_size)
        #     # Record things
        #     # logger.store(LossModel=loss_model.item())


        # Training Loops:
        # Start with random env steps (e.g 2000) and then use policy
        # if t >= 2000 and t % 50 == 0: # Update AC & model
        # if t > start_steps: # Update AC (every t for 5 itr) & model (every 1k t)
        if t // steps_per_epoch > train_model_epoch:
            print(colorize('AC training: time step = ', color='green'), t)
            for _ in range(5):
                batch = replay_buffer.sample_batch(batch_size)
                updateAC(data=batch)
            print(colorize('Model training while learning (Flexible)..', color='yellow'))
            updateModel(data=batch)

            # z = 1
            if ((e-1)*1e4) < t <= (e*1e4) and (t - ((e-1)*1e4)) % (e*1000) == 0:
                print(colorize('Model training while learning (Robust)..', color='yellow', bold=True))
                model.train_model(replay_buffer, e*1000, int(0.1*e*1000), 5)
                # z += 1
            elif e*1e4 < t:
                e += 1
            
            # logger.store(LossModel=loss_model.item())

            # if start_steps <= t-e <= 3000 and (t-start_steps) % 200 == 0: # start_steps = 2000
            #     print(colorize('Model training while learning (every 200 t, batch: 20, epochs: 5)..', color='yellow', bold=True))
            #     model.train_model(replay_buffer, 200, 20, 2)
            # elif 3000 < t-e <= 5000 and (t-3000) % 400 == 0:
            #     print(colorize('Model training while learning (every 400 t, batch: 40, epochs: 5)..', color='yellow', bold=True))
            #     model.train_model(replay_buffer, 400, 40, 2)
            # elif 5000 < t-e <= 8000 and (t-5000) % 600 == 0:
            #     print(colorize('Model training while learning (every 600 t, batch: 60, epochs: 5)..', color='yellow', bold=True))
            #     model.train_model(replay_buffer, 600, 60, 2)
            # elif 8000 < t-e <= 12000 and (t-8000) % 800 == 0:
            #     print(colorize('Model training while learning (every 800 t, batch: 80, epochs: 5)..', color='yellow', bold=True))
            #     model.train_model(replay_buffer, 800, 80, 2)
            # # elif 12000 < t-e <= 17000 and t % 1000 == 0:
            # #     print(colorize('Model training while learning (every 1000 t for 1 epochs)..', color='yellow', bold=True))
            # #     model.train_model(replay_buffer, t+1, 64, 1)
            # elif 12000 < t-e:
            #     e = t - 2000
            # else:
            #     print(colorize('Model training while learning (Flexible)..', color='yellow', bold=True))
            #     model.train_model(replay_buffer, batch_size, 32, 1)
        # # Update dynamics model after rand steps, then update every 100 steps
        # # Update AC every 50 steps
        # elif t % 500 == 0: # Only update model (every 500 t)
        # elif t % 1000 == 0: #t == start_steps:
        elif t == (steps_per_epoch*train_model_epoch):
            print(colorize('Model training before learning (Initial for 5 epochs)..', color='yellow', bold=True))
            model.train_model(replay_buffer, steps_per_epoch*train_model_epoch, 100, 5)

            # logger.store(LossModel=loss_model.item())

        # if t >= 1000 and t % 500 == 0:
        #     print(model_params)
        #     model.train_model(replay_buffer, t+1, batch_size)
        #     print(model_params)
        #     print('...Actor-Critic...')
        #     batch = replay_buffer.sample_batch(batch_size)
        #     print(model_params)
        #     updateAC(data=batch)
        #     model.train_model(replay_buffer, t+1, batch_size)



        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            ## Added by Rami >> ##
            # Save model after each epoch:
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)
            ## << Added by Rami ##

            # if epoch > 5 and epoch % 10 == 0 start to test the agent:
            if epoch > train_model_epoch and epoch % test_freq == 0:
            # if epoch > train_model_epoch and epoch % 1 == 0:
                # test the agent when we reach the test_freq:
                # reward_test = test_agent(epoch)
                test_agent(epoch)
                # save the experiment result:
                # reward_recorder.append(reward_test)
                # reward_nparray = np.asarray(reward_recorder)
                # np.save(str(exp_name)+'_'+str(env_name)+'_'+str(save_freq)+'.npy',reward_nparray)

                ## Added by Rami >> ##
                logger.log_tabular('Epoch', epoch)

                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)

                logger.log_tabular('TestEpRet', with_min_and_max=True) # if n=1 no variance
                logger.log_tabular('TestEpLen', average_only=True)

                logger.log_tabular('TotalEnvInteracts', t)
                
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)

                logger.log_tabular('Q1Vals', with_min_and_max=True) 
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                # logger.log_tabular('LossQ1', average_only=True)
                # logger.log_tabular('LossQ2', average_only=True)
                logger.log_tabular('V_Vals', with_min_and_max=True)
                # logger.log_tabular('LossV', average_only=True)
                logger.log_tabular('LossVal', average_only=True)


                ## Added by Rami >> ##
                # logger.log_tabular('DynM', with_min_and_max=True) 
                # logger.log_tabular('RewM', with_min_and_max=True)
                logger.log_tabular('LossModel', average_only=True)
                # logger.log_tabular('LossDyn', average_only=True)
                # logger.log_tabular('LossRew', average_only=True)
                ## << Added by Rami ##

                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
                ## << Added by Rami ##

            # if epoch % save_epoch == 0:
            #     # save the model after the final epoch:
            #     saver.save(sess, str(exp_name)+'_'+str(env_name),global_step=epoch)





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
    parser.add_argument('--exp_name', type=str, default='memb_pe')

    parser.add_argument('--train_model_epoch', type=int, default=5) # Train f(s,a) after 5 epochs
    parser.add_argument('--test_freq', type=int, default=1)
    # parser.add_argument('--env_name', type=str, default='HalfCheetah')
    # parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--save_epoch', type=int, default=500)

    args = parser.parse_args()

    ## Added by Rami >> ##
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    torch.set_num_threads(torch.get_num_threads())
    ## << Added by Rami ##

    I = 1
    for i in range(0,I):
        # repeat I times of experiment
        memb_pe(lambda : gym.make(args.env),

                    actor_critic=core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                    model=core.MLPModel,

                    gamma=args.gamma,
                    seed=args.seed,
                    epochs=args.epochs, # 200

                    save_freq=i, # 0,1,2,3,4
                    train_model_epoch=args.train_model_epoch, # 5
                    test_freq=args.test_freq, # 10

                    exp_name=args.exp_name,
                    env_name=args.env_name,
                    # save_epoch=args.save_epoch,
                    
                    logger_kwargs=logger_kwargs)



## Added by Rami >> ##
## << Added by Rami ##
