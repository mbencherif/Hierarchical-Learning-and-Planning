import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.dqn.qnet import QNet
from src.agents.dqn.buffer_simple import SimpleBuffer

# TODO: #3 Add decay of epsilon

class DQNAgent(BaseAgent):

  """Clipped Double Deep Q-Learning Agent."""

  def __init__(self,
               env,
               layer_param,
               learning_rate,
               gamma,
               tau,
               buffer_size,
               epsilon,
               epsilon_decay,
               epsilon_min,
               device):
    super(DQNAgent, self).__init__()
    self.device = device

    self.env = env
    self.gamma = gamma
    self.tau = tau
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min

    #self.replay_buffer = SimpleBuffer(max_size=buffer_size, device=self.device)
    self.replay_buffer = SimpleBuffer(action_size=0, buffer_size=buffer_size, device=self.device)

    self.qnet1 = QNet(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n,
                      layer_param=layer_param)
    self.qnet1 = self.qnet1.to(device)
    self.qnet2 = QNet(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n,
                      layer_param=layer_param)
    self.qnet2 = self.qnet2.to(device)
    self.optimizer1 = torch.optim.Adam(self.qnet1.parameters(), lr=learning_rate)
    #self.optimizer2 = torch.optim.Adam(self.qnet2.parameters())

    self.global_training_step = 0

  def plan_old(self, obs):
    """
    Epsilon-greedy action selection.

    :param obs: nd.array, torch.FloatTensor
      The observation of the state.

    :return action: nd.array
      The action which will be executed next.
    """
    if np.random.randn() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
      self.qnet1.eval()
      with torch.no_grad():
        q_vals = self.qnet1(state)
      self.qnet1.train()
      action = np.argmax(q_vals.cpu().numpy())

    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)
    return action

  def plan(self, obs, eps):
    """Returns actions for given state as per current policy.

    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
    self.qnet1.eval()
    with torch.no_grad():
      action_values = self.qnet1(state)
    self.qnet1.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.env.action_space.n))

  def _compute_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    :param batch: Tuple(torch.FloatTensors)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1, loss2:
      The MSE loss of Q-Net1 and Q-Net2.
    """
    states, actions, rewards, next_states, dones = batch
    #actions = actions.view(actions.size(0), 1)  # TODO: only needed for batch_size == 1?
    #dones = dones.view(dones.size(0), 1)  # TODO: only needed for batch_size == 1?

    # Get max predicted Q values (for next states) from target model
    #next_q1 = self.qnet1.forward(next_states).detach()
    #next_q2 = self.qnet2.forward(next_states).detach()
    #next_q = torch.min(torch.max(next_q1, 1)[0],
    #                   torch.max(next_q2, 1)[0])
    #next_q = next_q.view(next_q.size(0), 1)
    next_q = self.qnet2(next_states).detach().max(1)[0].unsqueeze(1)  # use Q_target

    # Compute Q targets for current states
    dones = dones.int()
    target_q = rewards + (self.gamma * next_q * (1 - dones))

    # Get expected Q values from local model
    #curr_q1 = self.qnet1(states).gather(1, actions)
    #curr_q2 = self.qnet2(states).gather(1, actions)
    curr_q = self.qnet1(states).gather(1, actions)
    # Return losses
    #loss1 = F.mse_loss(curr_q1, target_q.detach())
    #loss2 = F.mse_loss(curr_q2, target_q.detach())
    #return loss1, loss2
    loss = F.mse_loss(curr_q, target_q)
    return loss

  def update(self, batch_size):
    experiences = self.replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnet2(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnet1(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    self.optimizer1.zero_grad()
    loss.backward()
    self.optimizer1.step()

  def update_old(self, batch_size):
    """
    Train the Q-Networks with data from the replay buffer.

    :param batch_size: int
      Number of tuples to use for training the Q-Networks.
    :return:
    """
    batch = self.replay_buffer.sample(batch_size)
    #loss1, loss2 = self._compute_loss(batch)
    loss = self._compute_loss(batch)

    self.optimizer1.zero_grad()
    loss.backward()
    self.optimizer1.step()

    #self.optimizer1.zero_grad()
    #loss1.backward()
    #self.optimizer1.step()

    #self.optimizer2.zero_grad()
    #loss2.backward()
    #self.optimizer2.step()

    self._update_target_network()

    self.global_training_step += 1

  def _update_target_network(self):
    """Update target network: θ_target = τ*θ_local + (1 - τ)*θ_target."""
    for target_param, main_param in zip(self.qnet2.parameters(), self.qnet1.parameters()):
      target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)


