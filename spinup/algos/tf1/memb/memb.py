## Added by Rami >> ##
## << Added by Rami ##

import numpy as np
import tensorflow as tf
import gym
from gym.envs import mujoco
import time

## Added by Rami >> ##
# import pybullet_envs
# import pybullet as p
# p.connect(p.DIRECT)
## << Added by Rami ##

# import core
# from core import get_vars

## Added by Rami >> ##
from spinup.algos.tf1.memb import core
from spinup.algos.tf1.memb.core import get_vars
from spinup.utils.logx import EpochLogger
## << Added by Rami ##

#from spinup.pddm_envs.gym_env import GymEnv

class ReplayBuffer: # No changes
    """
    The replay buffer used to uniformly sample the data
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


"""
Model Embedding Model Based Algorithm (MEMB)
(with TD3 style Q value function update)
"""

def memb(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=200, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, model_lr=3e-4, value_lr=1e-3, pi_lr=3e-4, alpha=0.4,
        batch_size=100, start_steps=1000,
        max_ep_len=1000, save_freq=1,
        train_model_epoch=5, test_freq=5, save_epoch=100,
        exp_name='', env_name='',
        logger_kwargs=dict()):

    ## Added by Rami >> ##
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    ## << Added by Rami ##

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]



    ac_kwargs['action_space'] = env.action_space

    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    with tf.variable_scope('main'):
        # SAC: policy, value functions:
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)
        # Dyn/Rew models:
        transition , r_rm, transition_pi ,r_rm_pi, v_prime = core.reward_dynamic_model(x_ph, a_ph, pi, **ac_kwargs)

    # Target value network for updates
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _,v_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        # only needs st+1 --> Vpsi(st+1)




    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    ## Added by Rami >> ##
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/dm', 'main/rm', 'main/pi', 'main/v', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t dm: %d, \t rm: %d, \t pi: %d, \t v: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)
    ## << Added by Rami ##




    # TD3 style Q function updates #

    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Q_bp(st,at) = rt + gamma*Vtarg_psi-hat(st+1)
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    # V_bp(st) = Expt_at~pi[Q_phi(st,at) - alpha*logpi(at|st)]
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    r_backup = r_ph
    transition_backup = x2_ph


    ## Optimized costs\losses ##
    ### Model/Reward losses (supervised learning):
    #   loss = 0.5*(actual-prediction)^2 }
    #       Jp(omega) = 0.5 Expt_D[(f(s,a)-s')^2] --> eq#4.a
    #       Jr(ph) = Expt_D[(r(s,a)-r)^2] --> eq#4.b    
    #   Optz--> min_omeg,ph{ Jp(omeg), Jr(ph) }
    r_loss = 0.5 * tf.reduce_mean((r_backup-r_rm)**2)
    transition_loss = 0.5 * tf.reduce_mean((transition_backup - transition)**2)
    model_loss = r_loss+transition_loss

    ### Policy loss ###
    #   State value-function of st:
    #       V(st) = Expt_pi[Q(st,at) - log pi(at|st)] --> eq#3.b,
    #   Policy learning's Soft Bellman eq (Reparameterization):
    #       V(s) = Expt_pi[Expt_rm[r_hat(s,pi]
    #                       - alpha*log pi(a|s)
    #                       + gamma*Expt_f[V'(f(s,pi))]] --> eq#8
    #   Optz--> max_pi{ Expt_sD[V(s)] }
    pi_loss = r_rm_pi - alpha*logp_pi + gamma*(1-d_ph)*v_prime

    ### Value functions losses ###
    #   Optz--> min_phi,psi{ Jq(phi),Jv(psi) }
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    value_loss = q1_loss + q2_loss + v_loss


    ## Training/Optimization ##
    ### Model train op ###
    #       min_omeg,ph{ Jp(omeg), Jr(ph) }
    model_optimizer = tf.train.AdamOptimizer(learning_rate=model_lr)
    model_params = get_vars('main/dm') + get_vars('main/rm')
    train_model_op = model_optimizer.minimize(model_loss, var_list=model_params)

    ### Policy train op ###
    #       max_pi{ Expt[V(st)] }
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    policy_params = get_vars('main/pi')
    with tf.control_dependencies([train_model_op]): # policy pi depends on model f --> eq#8
        train_pi_op = pi_optimizer.minimize(-pi_loss, var_list=policy_params)

    ### Value train op ###
    #       min_phi,psi{ Jq(phi),Jv(psi) }
    value_optimizer = tf.train.AdamOptimizer(learning_rate=value_lr)
    value_params = get_vars('main/q') + get_vars('main/v')
    with tf.control_dependencies([train_pi_op]): # value V depends on policy pi --> eq#5
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    #### Value target update ####
    with tf.control_dependencies([train_value_op]): # V_target depends on V
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])



    step_ops = [logp_pi, q1, q2, v,
                pi_loss, q1_loss, q2_loss, v_loss,
                train_pi_op, train_value_op, target_update]

    model_ops = [transition , r_rm,
                transition_loss, r_loss,
                train_model_op]

    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    # saver = tf.compat.v1.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    ## Added by Rami from SpUp >> ##
    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={   'mu': mu,
                                            'pi': pi,
                                            'q1': q1,
                                            'q2': q2,
                                            'v' :  v,
                                            'dm': transition,
                                            'rm': r_rm})
    ## << Added by Rami ##

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    def test_agent(epoch,n=1):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        total_reward = 0
        for j in range(n): # repeat n=5 times
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

    start_time = time.time() ## By Rami
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs
    reward_recorder = []


    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        The algorithm would take total_steps totally in the training
        """

        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample() # Random for 1k (epoch 1)

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d # Don't let the env done if just reach max_ep_length

        replay_buffer.store(o, a, r, o2, d)

        o = o2 # Super critical!

        # End of trajectory handling if [(env is done) or (max_ep_legth reached)]
        if d or (ep_len == max_ep_len):
            
            ## Added by Rami >> ##
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ## << Added by Rami ##
            
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Learning/Training
        #   Train pi, Q, and V after 5 epochs for 5 times,
        #   Train dyn/rew models from start:
        if t // steps_per_epoch > train_model_epoch: # if epoch > 5
            # train 5 steps of Q, V, and pi.
            # train 1 step of model
            for j in range(5):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                            x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']}
                outs_step = sess.run(step_ops, feed_dict)
                ## Added by Rami >> ##
                # step_ops[0:7] = [logp_pi, q1, q2, v,
                #                   pi_loss, q1_loss, q2_loss, v_loss]
                logger.store(LogPi=outs_step[0],
                             Q1Vals=outs_step[1],
                             Q2Vals=outs_step[2],
                             V_Val=outs_step[3],
                             LossPi=outs_step[4],
                             LossQ1=outs_step[5],
                             LossQ2=outs_step[6],
                             LossV=outs_step[7])
                ## << Added by Rami ##
            # outs = sess.run(train_model_op, feed_dict)
            outs_model = sess.run(model_ops, feed_dict) # By Rami
            ## Added by Rami >> ##
            # model_ops[0:3] = [transition , r_rm,
            #                   transition_loss, r_loss,]
            # logger.store(DynM=outs_model[0],
            #              RewM=outs_model[1],
            #              LossDyn=outs_model[2],
            #              LossRew=outs_model[3])
            logger.store(LossDyn=outs_model[2],
                         LossRew=outs_model[3])
            ## Added by Rami >> ##
        else:
            # pretrain the model
            batch = replay_buffer.sample_batch(batch_size)
            feed_dict = {x_ph: batch['obs1'],
                         x2_ph: batch['obs2'],
                         a_ph: batch['acts'],
                         r_ph: batch['rews'],
                         d_ph: batch['done'],
                         }
            # outs = sess.run(train_model_op, feed_dict)
            outs_model = sess.run(model_ops, feed_dict) # By Rami
            ## Added by Rami >> ##
            # model_ops[0:3] = [transition , r_rm,
            #                   transition_loss, r_loss,]
            logger.store(DynM=outs_model[0],
                         RewM=outs_model[1],
                         LossDyn=outs_model[2],
                         LossRew=outs_model[3])
            ## Added by Rami >> ##

        # # End of trajectory handling if [(env is done) or (max_ep_legth reached)]
        # if d or (ep_len == max_ep_len):
        #     o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

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
                reward_test = test_agent(epoch)
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
                logger.log_tabular('LossQ1', average_only=True)
                logger.log_tabular('LossQ2', average_only=True)
                logger.log_tabular('V_Val', with_min_and_max=True)
                logger.log_tabular('LossV', average_only=True)


                ## Added by Rami >> ##
                # logger.log_tabular('DynM', with_min_and_max=True) 
                # logger.log_tabular('RewM', with_min_and_max=True)
                logger.log_tabular('LossDyn', average_only=True)
                logger.log_tabular('LossRew', average_only=True)
                ## << Added by Rami ##

                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()
                ## << Added by Rami ##

            # if epoch % save_epoch == 0:
            #     # save the model after the final epoch:
            #     saver.save(sess, str(exp_name)+'_'+str(env_name),global_step=epoch)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=250)
    # parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='memb')

    parser.add_argument('--train_model_epoch', type=int, default=5) # Train f(s,a) after 5 epochs
    parser.add_argument('--test_freq', type=int, default=1)
    # parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--save_epoch', type=int, default=5)

    args = parser.parse_args()

    ## Added by Rami >> ##
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    ## << Added by Rami ##

    I = 1
    for i in range(0,I):
        # repeat I times of experiment
        tf.reset_default_graph()
        memb(lambda : gym.make(args.env),

                    actor_critic=core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                    gamma=args.gamma,
                    seed=args.seed,
                    epochs=args.epochs, # 100

                    save_freq=i, # 0,1,2,3,4
                    train_model_epoch=args.train_model_epoch, # 5
                    test_freq=args.test_freq, # 10

                    exp_name=args.exp_name,
                    env_name=args.env_name,
                    # save_epoch=args.save_epoch,
                    
                    logger_kwargs=logger_kwargs)



## Added by Rami >> ##
## << Added by Rami ##
