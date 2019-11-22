import torch.nn as nn
import torch.nn.functional as F

from src.agents.base_agent import BaseAgent
from src.agents.agent_commons import create_nn_layer



class QNet(nn.Module):

  def __init__(self, state_dim, action_dim, layer_param):
    """
    Init the Q-Network: Q(s) = r_a.

    The Q-Net returns the expected reward for all actions at the current time
    step.

    :param state_dim:
    The number of channel of input.
    i.e The number of most recent frames stacked together.

    :param action_dim:
    Number of action-values to output, one-to-one correspondence to actions in
    game.

    """
    super(QNet, self).__init__()

    layer_param[0]["n_neurons"][0] = state_dim
    layer_param[-2]["n_neurons"][1] = action_dim
    self.layer_param = layer_param

    self.module_list = nn.ModuleList()
    for layer_def in self.layer_param:
      layer = create_nn_layer(layer_def)
      self.module_list.append(layer)

  def forward(self, x):
    for layer in self.module_list:
      x = layer(x)
    x = x.clone()
    return x