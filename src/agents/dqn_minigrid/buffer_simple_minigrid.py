import random
import numpy as np
import torch
from collections import namedtuple, deque


class SimpleBuffer:

  def __init__(self, max_size, device):
    self.device = device
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self,
           state_img,
           state_data,
           action,
           reward,
           next_state_img,
           next_state_data,
           done):
    experience = (state_img,
                  state_data,
                  action,
                  np.array(reward),
                  next_state_img,
                  next_state_data,
                  done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    batch = np.asarray(random.sample(self.buffer, batch_size))
    states_img = np.stack(batch[:, 0])
    states_data = np.vstack(batch[:, 1])
    actions = np.vstack(batch[:, 2])
    rewards = np.vstack(batch[:, 3])
    next_states_img = np.stack(batch[:, 4])
    next_states_data = np.stack(batch[:, 5])
    dones = np.vstack(batch[:, 6])
    states_img = torch.from_numpy(states_img).float().to(device=self.device)
    states_data = torch.from_numpy(states_data).float().to(device=self.device)
    actions = torch.from_numpy(actions).long().to(device=self.device)
    rewards = torch.from_numpy(rewards).float().to(device=self.device)
    next_states_img = torch.from_numpy(next_states_img).float().to(device=self.device)
    next_states_data = torch.from_numpy(next_states_data).float().to(device=self.device)
    dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device=self.device)
    return (states_img, states_data), actions, rewards, (next_states_img, next_states_data), dones

  def __len__(self):
    return len(self.buffer)
