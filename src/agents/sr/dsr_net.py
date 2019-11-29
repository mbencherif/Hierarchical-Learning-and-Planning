import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.agent_commons import create_nn_layer

# Future TODOs:
# - best hidden representation? Probabilistic layer with VAE?

class DSRNet(nn.Module):

  def __init__(self, state_dim, action_dim, latent_state_dim=64):
    """
    Init the Deep Successor Representation Network.

    :param state_dim:
      Dimension of input data in the form of (n_rows, n_cols, n_channel)

    :param action_dim:
      Number of action-values to output, one-to-one correspondence to actions in
      game.

    """
    super(DSRNet, self).__init__()

    # Check if latent_state_dim has a perfect sqrt
    root = np.sqrt(latent_state_dim)
    assert int(root + 0.5) ** 2 == latent_state_dim

    self.action_dim = action_dim
    self.latent_state_dim = latent_state_dim

    self.modules_encoder_conv = nn.Sequential(
      nn.Conv2d(state_dim[-1], 16, kernel_size=3, stride=1, padding=0),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      #nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      #nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      #nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.modules_encoder_dense = nn.Sequential(
      nn.Linear(64 * 2 * 2, latent_state_dim),  # Given an input data of shape (7,7,3)
      nn.Sigmoid
    )

    self.modules_r = nn.Sequential(nn.Linear(latent_state_dim, 1))

    self.modules_decoder = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.ConvTranspose2d(16, state_dim[-1], kernel_size=5, stride=1, padding=0),
      nn.Tanh()
    )

    self.modules_m = nn.Sequential(
      nn.Linear(latent_state_dim, 128),
      nn.ReLU,
      nn.Linear(128, 128),
      nn.ReLU,
      nn.Linear(128, latent_state_dim),
      nn.Sigmoid
    )

  def get_w_vector(self):
    return self.modules_r[0].weight

  def forward(self, x):
    """
    Forward pass through the network

    :param x: torch.tensor
      input state or observation.

    :return tuple(x, w, x_r, x_reconstr, x_ms):
      x: latent representation of the state
      w: weight vector foc cumputing the reward
      x_r: scalar predicting the reward for being in the given state
      x_reconstr: the reconstructed state
      x_ms: a list containing all successor features for each action at the current state
    """
    z = self.forward_latent_state(x)
    r = self.forward_reward(z.clone())
    s_reconstr = self.forward_reconstr_state(z.clone())
    m_sr_a = self.forward_successor_rep(z.clone())
    return x, r, s_reconstr, m_sr_a

  def forward_latent_state(self, x):
    """State -> Latent Representation"""
    for layer in self.modules_encoder_conv:
      x = layer(x)
    x = x.view(x.size(0), -1)
    for layer in self.modules_encoder_dense:
      x = layer(x)
    return x

  def forward_reward(self, x):
    """State -> Reward, Weight Vector"""
    x = self.forward_latent_state(x)
    x = self.modules_r[0](x)
    return x

  def forward_reconstr_state(self, x):
    """State -> Reconstructed State"""
    x = x.view(np.sqrt(x.size()), np.sqrt(x.size()))
    x = self.forward_latent_state(x)
    for encoder_layer in self.modules_decoder:
      x = encoder_layer(x)
    return x

  def forward_successor_rep(self, x):
    """State -> list(SR_a_i)"""
    x_ms = []
    for _ in range(self.action_dim):
      x_m = x.clone()
      for m_layer in self.modules_m:
        x_m = m_layer(x_m)
      x_ms.append(x_m)
    return x_ms

  def activate_network(self):
    for p in self.modules_encoder.parameters():
      p.requires_grad = True
    for p in self.modules_r.parameters():
      p.requires_grad = True
    for p in self.modules_decoder.parameters():
      p.requires_grad = True
    for p in self.modules_m.parameters():
      p.requires_grad = True

  def deactivate_features_path(self):
    for p in self.modules_encoder.parameters():
      p.requires_grad = False
    for p in self.modules_r.parameters():
      p.requires_grad = False
    for p in self.modules_decoder.parameters():
      p.requires_grad = False

  def deactivate_sr_path(self):
    for p in self.modules_m.parameters():
      p.requires_grad = False