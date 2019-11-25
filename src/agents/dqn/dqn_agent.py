import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.dqn.qnet import QNet
from src.agents.dqn.buffer_simple import SimpleBuffer

# TODO: #3 Add decay of epsilon
# TODO: #4 Check output layer of q net: Sigm, tanh, ... ?

class DQNAgent(BaseAgent):

  """Clipped Double Deep Q-Learning Agent."""

  def __init__(self,
               env,
               layer_param,
               learning_rate=3e-4,
               gamma=0.99,
               tau=0.01,
               buffer_size=10000,
               epsilon=0.99,
               epsilon_decay=0.999,
               device="cpu"):
    super(DQNAgent, self).__init__()
    self.device = device

    self.env = env
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.tau = tau
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay

    self.replay_buffer = SimpleBuffer(max_size=buffer_size, device=self.device)

    self.qnet1 = QNet(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n,
                      layer_param=layer_param)
    self.qnet1 = self.qnet1.to(device)
    self.qnet2 = QNet(state_dim=env.observation_space.shape[0],
                      action_dim=env.action_space.n,
                      layer_param=layer_param)
    self.qnet2 = self.qnet2.to(device)
    self.optimizer1 = torch.optim.Adam(self.qnet1.parameters())
    self.optimizer2 = torch.optim.Adam(self.qnet2.parameters())

    self.global_training_step = 0

  def plan(self, obs):
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
      state = torch.FloatTensor(obs).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        qvals = self.qnet1(state)
      action = np.argmax(qvals.cpu().detach().numpy())
    self.epsilon *= self.epsilon_decay
    return action

  def _compute_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    :param batch: Tuple(torch.FloatTensors)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1, loss2:
      The MSE loss of Q-Net1 and Q-Net2.
    """
    states, actions, rewards, next_states, dones = batch
    actions = actions.view(actions.size(0), 1)
    dones = dones.view(dones.size(0), 1)

    # Get max predicted Q values (for next states) from target model
    next_q1 = self.qnet1.forward(next_states).detach()
    next_q2 = self.qnet2.forward(next_states).detach()
    next_q = torch.min(torch.max(next_q1, 1)[0],
                       torch.max(next_q2, 1)[0])
    next_q = next_q.view(next_q.size(0), 1)

    # Compute Q targets for current states
    dones = dones.int()
    target_q = rewards + (1 - dones) * self.gamma * next_q

    # Get expected Q values from local model
    curr_q1 = self.qnet1.forward(states).gather(1, actions)
    curr_q2 = self.qnet2.forward(states).gather(1, actions)
    # Return losses
    loss1 = F.mse_loss(curr_q1, target_q.detach())
    loss2 = F.mse_loss(curr_q2, target_q.detach())
    return loss1, loss2

  def update(self, batch_size):
    """
    Train the Q-Networks with data from the replay buffer.

    :param batch_size: int
      Number of tuples to use for training the Q-Networks.
    :return:
    """
    batch = self.replay_buffer.sample(batch_size)
    loss1, loss2 = self._compute_loss(batch)

    self.optimizer1.zero_grad()
    loss1.backward()
    self.optimizer1.step()

    self.optimizer2.zero_grad()
    loss2.backward()
    self.optimizer2.step()

    self.global_training_step += 1