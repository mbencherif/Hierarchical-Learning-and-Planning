import numpy as np
import random
from absl import logging

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn_gym.qnet_dense import QNet
from src.agents.dqn_gym.qnet_conv import ConvQNet
from src.agents.dqn_gym.buffer_simple import SimpleBuffer
from src.utils.plotting import plot_train_progress


class DQNAgent:
  """Normal and Clipped Double Deep Q-Learning Agent."""

  # TODO: state space as tuple! (4,) or (28, 28, 3)

  def __init__(self,
               state_space,
               action_space,
               buffer_size,
               batch_size,
               gamma,
               tau,
               lr,
               training_interval,
               epsilon,
               epsilon_decay,
               epsilon_min,
               layer_param,
               plot_dir,
               device):
    """
    Initialize the Deep Q Learning agent.

    :param state_dim: Int or tuple(Ints)
      State dimension:  Int for dense network
                        Tuple for conv network
    :param action_size: Int
      Number of actions.
    :param buffer_size: Int
      Size of the replay buffer.
    :param batch_size: Int
      Number of samples to use for computing the loss at a time.
    :param gamma: Float
      Discount factor between 0 and 1.
    :param tau: Float
      Value between 0 and 1 used for updating the target network.
      Only used in the case of ordinary Double Deep Q Learning
    :param lr: Float
      The learning rate used for both Q networks.
    :param training_interval: Int
      Defining the interval on how often to update the network.
    :param epsilon: Float
      Start value for epsilon greedy. Between 0 and 1.
    :param epsilon_decay: Float
      Rate at which epsilon will decay during training. Between 0 and 1.
    :param epsilon_min: Float
      Min value epsilon can reach. Between 0 and 1.
    :param layer_param: Dict
      Desciption of the Q-Net architecture
    :param device: String
      Set 'cpu' or 'cuda' for either using the cpu or gpu the neural
      network calculations repectively.
    """
    self.device=device
    self.batch_size = batch_size
    self.gamma = gamma
    self.training_interval = training_interval  # TODO is 1?
    self.tau = tau
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.memory = SimpleBuffer(max_size=buffer_size, device=device)
    self.global_step_count = 0
    self.training_progress = []
    self.plot_dir = plot_dir
    self.action_dim = action_space
    if layer_param == "conv":
      #self.channel_dim = state_space.shape[-1]
      #self.action_dim = action_space.shape[0]
      self.qnet1 = ConvQNet(state_space[-1], self.action_dim).to(device)
      self.qnet2 = ConvQNet(state_space[-1], self.action_dim).to(device)
    else:
      #self.state_dim = state_space.shape[0]
      #self.action_dim = action_space.n
      self.qnet1 = QNet(state_space[0], self.action_dim, layer_param).to(device)
      self.qnet2 = QNet(state_space[0], self.action_dim, layer_param).to(device)
    self.optimizer1 = optim.Adam(self.qnet1.parameters(), lr=lr)
    self.optimizer2 = optim.Adam(self.qnet2.parameters(), lr=lr)

  def plan(self, obs):
    """
    Epsilon-greedy action selection.

    :param obs: nd.array
      The observation of the state.

    :return action: nd.array
      The action which will be executed next.
    """
    if random.random() > self.epsilon:
      obs = torch.from_numpy(obs.copy()).float().unsqueeze(0).to(self.device)
      self.qnet1.eval()
      with torch.no_grad():
        action_values = self.qnet1(obs)
      self.qnet1.train()
      action = np.argmax(action_values.cpu().data.numpy())
    else:
      action = random.choice(np.arange(self.action_dim))
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon, self.epsilon_min)
    return action

  def step(self, state, action, reward, next_state, done):
    self.memory.push(state, action, reward, next_state, done)
    if self.global_step_count % self.training_interval == 0:
      if len(self.memory) > self.batch_size:
        batch = self.memory.sample(self.batch_size)
        loss = self.optimize_regular(batch)
        self.training_progress.append([self.global_step_count, reward, loss, self.epsilon])
    if self.training_progress and self.global_step_count % 1000 == 0:
      plot_train_progress(self.training_progress,
                          save_dir=self.plot_dir + "/agent_training.png")
    self.global_step_count += 1

  def optimize_clipped(self, batch):
    """Optimize the Q networks corresponding to Clipped Double Q-Learning."""
    #batch = self.replay_buffer.sample(batch_size)
    loss1, loss2 = self._compute_clipped_loss(batch)
    self.optimizer1.zero_grad()
    loss1.backward()
    self.optimizer1.step()
    self.optimizer2.zero_grad()
    loss2.backward()
    self.optimizer2.step()

  def _compute_clipped_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    Calculating the loss for Clipped Double Q-Learning from:
    "Addressing Function Approximation Error in Actor-Critic Methods", Fujimoto et al. (2018)
    https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param batch: Tuple(torch.FloatTensor,
                        torch.LongTensor,
                        torch.FloatTensor,
                        torch.FloatTensor,
                        torch.FloatTensor)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1, loss2:
      The MSE loss of Q-Net1 and Q-Net2.
    """
    states, actions, rewards, next_states, dones = batch
    # Current Q1, Q2
    q1_current = self.qnet1(states).gather(1, actions)
    q2_current = self.qnet2(states).gather(1, actions)
    # Target Q
    q1_target_next = torch.max(self.qnet1.forward(next_states).detach(), 1)[0]
    q2_target_next = torch.max(self.qnet2.forward(next_states).detach(), 1)[0]
    q_target_next = torch.min(q1_target_next, q2_target_next)
    q_target_next = q_target_next.view(q_target_next.size(0), 1)
    target_q = rewards + ((1 - dones) * self.gamma * q_target_next)
    # Loss
    loss1 = F.mse_loss(q1_current, target_q.detach())
    loss2 = F.mse_loss(q2_current, target_q.detach())
    return loss1, loss2

  def optimize_regular(self, batch):
    """Optimize the Q networks corresponding to Double Q-Learning."""
    loss = self._compute_regular_loss(batch)
    self.optimizer1.zero_grad()
    loss.backward()
    self.optimizer1.step()
    self._update_target_network(self.qnet1, self.qnet2)
    return loss

  def _compute_regular_loss(self, batch):
    """
    Compute the loss given a batch of (s,a,s',r,t).

    Regular loss for Double Deep Q Learning where the next_a is computed
    using the target network.

    :param batch: Tuple(torch.FloatTensor,
                        torch.LongTensor,
                        torch.FloatTensor,
                        torch.FloatTensor,
                        torch.FloatTensor)
      Batch of (state, action , next_state, reward, terminal)-tuples.

    :return loss1:
      The MSE loss of Q-Net1.
    """
    states, actions, rewards, next_states, dones = batch
    q_targets_next = self.qnet2(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
    q_current = self.qnet1(states).gather(1, actions)
    loss = F.mse_loss(q_current, q_targets)
    return loss

  def _update_target_network(self, local_model, target_model):
    """Update target network: θ_target = τ*θ_local + (1 - τ)*θ_target."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

  def save(self, model_path):
    model_path.mkdir(parents=True, exist_ok=True)
    try:
      torch.save(self.qnet1.state_dict(),
                 (model_path / "qnet1").absolute().as_posix())
      torch.save(self.optimizer1.state_dict(),
                 (model_path / "optimizer1").absolute().as_posix())
      torch.save(self.qnet2.state_dict(),
                 (model_path / "qnet2").absolute().as_posix())
      torch.save(self.optimizer2.state_dict(),
                 (model_path / "optimizer2").absolute().as_posix())
      logging.debug(f"DQN model saved to '{model_path}'")
    except Exception as e:
      logging.info(f"ERROR: DQN model was NOT saved to '{model_path}'")

  def load(self, model_path):
    try:
      self.qnet1.load_state_dict(torch.load(
        (model_path / "qnet1").absolute().as_posix()))
      self.optimizer1.load_state_dict(torch.load(
        (model_path / "optimizer1").absolute().as_posix()))
      self.qnet2.load_state_dict(torch.load(
        (model_path / "qnet2").absolute().as_posix()))
      self.optimizer2.load_state_dict(torch.load(
        (model_path / "optimizer2").absolute().as_posix()))
      logging.info(f"DQN model loaded from '{model_path}'")
    except Exception as e:
      logging.info(f"No DQN model loaded from '{model_path}'")