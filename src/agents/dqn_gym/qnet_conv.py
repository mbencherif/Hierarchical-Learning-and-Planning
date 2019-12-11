import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvQNet(nn.Module):

  def __init__(self, channel_dim, action_dim):
    """
    Init the Q-Network: Q(s) = r_a.

    The Q-Net returns the expected reward for all actions at the current time
    step.

    :param channel_dim:
    The number of channel of input.
    i.e The number of most recent frames stacked together.

    :param action_dim:
    Number of action-values to output, one-to-one correspondence to actions in
    game.

    """
    super(ConvQNet, self).__init__()
    conv_output_dim = 8 * 5 * 5

    self.conv_modules = nn.Sequential(
      nn.Conv2d(channel_dim, 32, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2),
      nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2),
      nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2)
    )
    #summary(self.conv_modules, (3, 96, 96) ,device="cpu")
    self.dense_modules = nn.Sequential(
      nn.Linear(conv_output_dim, 128),
      nn.ReLU(),
      nn.Linear(128, action_dim))

  def forward(self, x):

    """
    Forward pass of the network.

    :param x:
      Network input data of the form (N x C x H x W) with
        N: batch size
        C: number of channels
        H: hight of the input data
        W  width of the input data

    :return x:
      Network output.
    """
    x = x.permute(0, 3, 1, 2)
    for conv_layer in self.conv_modules:
      x = conv_layer(x)
    x = x.reshape(x.size(0), -1)
    for dense_layer in self.dense_modules:
      x = dense_layer(x)
    return x
