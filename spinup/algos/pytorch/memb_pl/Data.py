from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


import spinup.algos.pytorch.memb_pe.core as core


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


class RLDataset(IterableDataset):

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        data = self.buffer.sample_batch(self.sample_size)

        obs = data['obs']
        obs2 = data['obs2']
        act = data['act']
        rew = data['rew']
        done = data['done']

        for i in range(len(done)):
            yield obs[i], obs2[i], act[i], rew[i], done[i]




replay_buffer = ReplayBuffer(3, 2, 50)
episode_length = 2
batch_size = 1


o = torch.Tensor([1,1,1])
a = torch.Tensor([1,1])
o2 = torch.Tensor([2,2,2])
r = 5
d = False
replay_buffer.store(o, a, r, o2, d)

o = torch.Tensor([3,3,3])
a = torch.Tensor([3,3])
o2 = torch.Tensor([6,6,6])
r = 10
d = True
replay_buffer.store(o, a, r, o2, d)


print(replay_buffer.size)

# dataset = RLDataset(replay_buffer, episode_length)
# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=batch_size,
#     sampler=None,
# )










# batch = replay_buffer.sample_batch(batch_size=10)
# print(batch)
