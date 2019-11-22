import random
import numpy as np
import torch
from collections import deque

__device__ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleBuffer:

  def __init__(self, max_size):
    self.max_size = max_size
    self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
    experience = (state, action, np.array([reward]), next_state, done)
    self.buffer.append(experience)

  def sample(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    batch = random.sample(self.buffer, batch_size)

    for experience in batch:
      state, action, reward, next_state, done = experience
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    state_batch = torch.FloatTensor(state_batch).to(device=__device__)
    action_batch = torch.LongTensor(action_batch).to(device=__device__)
    reward_batch = torch.FloatTensor(reward_batch).to(device=__device__)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device=__device__)
    done_batch = torch.FloatTensor(done_batch).to(device=__device__)

    return (
      state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def sample_sequence(self, batch_size):
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    done_batch = []

    min_start = len(self.buffer) - batch_size
    start = np.random.randint(0, min_start)

    for sample in range(start, start + batch_size):
      state, action, reward, next_state, done = self.buffer[start]
      state_batch.append(state)
      action_batch.append(action)
      reward_batch.append(reward)
      next_state_batch.append(next_state)
      done_batch.append(done)

    state_batch = torch.FloatTensor(state_batch).to(device=__device__)
    action_batch = torch.FloatTensor(action_batch).to(device=__device__)
    reward_batch = torch.FloatTensor(reward_batch).to(device=__device__)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device=__device__)
    done_batch = torch.FloatTensor(done_batch).to(device=__device__)

    return (
    state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
    return len(self.buffer)