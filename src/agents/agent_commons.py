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
    return nn.Conv2d(conf["filter"][0],
                     conf["filter"][1],
                     conf["kernel_size"],
                     conf["stride"])
  elif conf["type"] == "relu":
    return nn.ReLU()
  elif conf["type"] == "tanh":
    return nn.Tanh()
  elif conf["type"] == "sigm":
    return nn.Sigmoid()
  elif conf["type"] == "softmax":
    return nn.Softmax()
  else:
    raise NotImplementedError(
      f"This type of layer is not supported: {conf['type']}")