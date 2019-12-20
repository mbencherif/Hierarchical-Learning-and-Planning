import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from absl import logging


class ConvNet(nn.Module):

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
    super(ConvNet, self).__init__()

    input_shape = (3, 7, 7)
    conv_output_dim = 16 * 2 * 2

    self.conv_modules = nn.Sequential(
      nn.Conv2d(channel_dim, 32, kernel_size=4, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
      nn.ReLU(),
      nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=0),
      nn.ReLU()
    )
    print(f"Network Input Shape {input_shape}, Conv Output: {conv_output_dim}")
    #summary(self.conv_modules, input_shape, device="cpu")
    self.dense_modules = nn.Sequential(
      nn.Linear(conv_output_dim + 1, 128),  # TODO: add state[data] dim!
      nn.ReLU(),
      nn.Linear(128, action_dim))

    logging.info("Network initialized")

  def forward(self, x):
    """
    Forward pass of the network.

    :param x:
      Tuple of (Image, Data)
        Image input data of shape (N x C x H x W) with
            N: batch size
            C: number of channels
            H: hight of the input data
            W  width of the input data
        Data input data as a vector of shape (n,)

    :return x:
      Network output.
    """
    x_img, x_data = x
    x_img = x_img.permute(0, 3, 1, 2)
    for conv_layer in self.conv_modules:
      x_img = conv_layer(x_img)
    x = torch.cat((x_img.reshape(x_img.size(0), -1),
                   x_data.reshape(x_img.size(0), -1)), dim=1)
    for dense_layer in self.dense_modules:
      x = dense_layer(x)
    return x
