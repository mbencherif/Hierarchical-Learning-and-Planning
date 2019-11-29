import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.agent_commons import create_nn_layer



class DSRNet(nn.Module):

  def __init__(self, state_dim, latent_rep_dim, action_dim):
    """
    Init the Deep Successor Representation Network.

    :param state_dim:
      The number of channel of input.
      i.e The number of most recent frames stacked together.

    :param action_dim:
      Number of action-values to output, one-to-one correspondence to actions in
      game.

    """
    super(DSRNet, self).__init__()

    description_encoder = [
      {"type": "linear", "n_neurons": [state_dim, 256]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [256, 128]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [128, latent_rep_dim]},
      {"type": "sigmoid"}  # TODO: best hidden representation? Probabilistic layer with VAE?
    ]
    description_r = [
      {"type": "linear", "n_neurons": [latent_rep_dim, 128]},
      {"type": "linear", "n_neurons": [128, 1]}
    ]
    description_decoder = [
      {"type": "linear", "n_neurons": [latent_rep_dim, 64]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [64, 128]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [128, state_dim]},
      {"type": "tanh"}
    ]
    description_m = [
      {"type": "linear", "n_neurons": [latent_rep_dim, 128]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [128, 128]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [128, 128]},
      {"type": "relu"},
      {"type": "linear", "n_neurons": [128, state_dim]},
      {"type": "sigmoid"}
    ]
    self.modules_encoder = self._create_networks(description_encoder)
    self.modules_r = self._create_networks(description_r)
    self.modules_decoder = self._create_networks(description_decoder)
    self.modules_m = self._create_networks(description_m)

  def _create_networks(self, description):
    module_list = nn.ModuleList()
    for layer_def in description:
      layer = create_nn_layer(layer_def)
      module_list.append(layer)
    return module_list

  def forward(self, x):
    for encoder_layer in self.modules_encoder:
      x = encoder_layer(x)

    x_r = x.clone()
    for r_layer in self.modules_r:
      x_r = r_layer(x_r)

    x_reconstr = x.clone()
    for r_layer in self.modules_r:
      x_reconstr = r_layer(x_reconstr)

    for
    x_ = x.clone()
    for r_layer in self.modules_r:
      x_reconstr = r_layer(x_reconstr)

    return x