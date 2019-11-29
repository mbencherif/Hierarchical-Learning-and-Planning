import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.sr.dsr_net import DSRNet
from src.agents.dqn.buffer_simple import SimpleBuffer


class DSRAgent(BaseAgent):
  """Agent based on Successor Representations."""

  def __init__(self, env, latent_state_dim, learning_rate, gamma, buffer_size, batch_size,
               epsilon, epsilon_decay, epsilon_min, device):
    super(DSRAgent, self).__init__()

    self.device = device
    self.env = env
    self.state_dim = env.observation_space.shape
    self.action_dim = env.action_space.n
    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.replay_buffer = SimpleBuffer(max_size=buffer_size, device=self.device)

    self.dsr_net = DSRNet(state_dim=env.observation_space["image"].shape,
                          action_dim=env.action_space.n,
                          latent_state_dim=latent_state_dim)
    self.dsr_net = self.dsr_net.to(device)
    self.optimizer = torch.optim.Adam(self.dsr_net.parameters(), lr=learning_rate)

  def plan(self, obs):
    action = self._exporation()
    if action is None:
      obs = torch.FloatTensor(obs).float().unsqueeze(0).to(self.device)
      self.dsr_net.eval()
      with torch.no_grad():
        #latent_state, w, pred_reward, reconstr_state, sr_a = self.dsr_net(obs).cup().numpy
        m_sr_a = self.dsr_net.forward_successor_rep(obs).cup().numpy
        w = self.dsr_net.get_w_vector()
      self.dsr_net.train()
      q_vals = np.stack(m_sa * w for m_sa in m_sr_a)  # TODO: check if scalar output
      action = np.argmax(q_vals)
    return action

  def _exporation(self):
    """Epsilon Greedy."""
    if np.random.randn() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      action = None
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    return action

  def update(self, n_iterations):
    flip_flop = True
    for i in range(n_iterations):
      self.optimizer.zero_grad()
      batch = self.replay_buffer.sample(batch_size=self.batch_size)

      if flip_flop:
        loss = self._loss_latent_rep(batch)
        self.dsr_net.deactivate_sr_path()
        loss.backward()

      else:
        loss = self._loss_successor_rep(batch)
        self.dsr_net.deactivate_features_path()
        loss.backward()

      self.optimizer.step()
      self.dsr_net.activate_network()
      flip_flop = 1 - self.flip_flop

  def _loss_latent_rep(self, batch):
    """
    Calculate the loss for the latent representation.

    This loss is decomposed in two parts:
      - J_r = MSE(reward, calculated_reward)
      - J_s = MSE(s, s_reconstructed)

    :param batch: tuple
      states, actions, rewards, next_states, dones

    :return loss:
      The combined loss of the reward and reconstruction heads of the neural network.
    """
    states, _, rewards, _, _ = batch  # TODO: Paper says reward_s_next instead of reward_s?
    z = self.dsr_net.forward_latent_state(states)
    reconstr_states = self.dsr_net.forward_reconstr_state(states)
    w = self.dsr_net.get_w_vector()

    rewards_target = z * w  # TODO: check if z[0] * w is scalar, total should be vector
    loss_r = F.mse_loss(rewards, rewards_target)  # TODO (!): use received reward or calc reward?
    loss_reconst = F.mse_loss(states, reconstr_states)
    loss = sum(loss_r, loss_reconst)
    return loss

  def _loss_successor_rep(self, batch):
    """
    Calculate the loss for the latent representation.

    The loss is the TD-Error: (z + gamma*m(s_next,a_next)) - m(s,a)

    :return loss:
      The loss of the neural network's head for successor features.
    """
    a_next = 0  # TODO: next action
    z = 0  # TODO: laten state
    m_sa = 0  # TODO: for argmax a from q_vals
    m_sa_next = 0  # TODO: for argmax a_next from q_vals_next

    target_m_sa_next = z + self.gamma * m_sa
    loss = F.mse_loss(target_m_sa_next, m_sa_next)
    return loss

  def store(self, s, a, r, s_, t):
    """
    Store the experience tuple in the buffer

    :param s: nd.array
      State
    :param a: int
      Action
    :param r: float
      Reward
    :param s_: nd.array
      Next state
    :param t: Bool
      Terminal
    """
    self.replay_buffer.push(state=s, action=a, reward=r, next_state=s_, done=t)