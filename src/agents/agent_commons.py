import torch.nn as nn


def create_nn_layer(conf):
  """
  Create a PyTorch layer given a definition file.

  :param conf: dict
    Dictionary containing the layer configuration.

  :return pytorch.nn.module:
    Return the corresponding PyTorch layer module.
  """
  if conf["type"] == "linear":
    return nn.Linear(conf["n_neurons"][0],
                     conf["n_neurons"][1])
  elif conf["type"] == "conv":
    return nn.Conv2d(in_channels=conf["in_channels"],
                     out_channels=conf["out_channels"],
                     kernel_size=conf["kernel_size"],
                     stride=conf["stride"],
                     padding=0)
  elif conf["type"] == "maxpool":
    return nn.MaxPool2d(kernel_size=conf["kernel_size"],
                        stride=conf["stride"],
                        padding=conf["padding"])
  elif conf["type"] == "batchnorm":
    return nn.BatchNorm2d(conf["channels"])
  elif conf["type"] == "relu":
    return nn.ReLU()
  elif conf["type"] == "tanh":
    return nn.Tanh()
  elif conf["type"] == "sigmoid":
    return nn.Sigmoid()
  elif conf["type"] == "softmax":
    return nn.Softmax()
  else:
    raise NotImplementedError(
      f"This type of layer is not supported: {conf['type']}")