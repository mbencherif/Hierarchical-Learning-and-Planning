import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.dqn.qnet import QNet
from src.agents.dqn.buffer_simple import SimpleBuffer


class SRAgent(BaseAgent):
  """Agent based on Successor Representations."""

  def __init__(self, env, learning_rate, gamma):
    super(SRAgent, self).__init__()

    self.state_dim = env.observation_space.shape
    self.action_dim = env.action_space.n
    self.learning_rate = learning_rate
    self.gamma = gamma

    self.m_matrix = np.stack([np.identity(self.state_dim) for i in range(self.action_dim)])
    self.r_vect = np.zeros((self.state_dim))

  def plan(self, observation):
    # Calculate Q
    pass


  def update(self, state, reward):
    state_idx = self._get_state_idx(state)
    self.r_vect[state_idx] += self.learning_rate * (reward - self.r_vect)

  def _get_state_idx(self, state):
    raise NotImplementedError
