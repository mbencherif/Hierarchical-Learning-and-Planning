from collections import deque

from src.agents.sr_ddc.conv_nn import ConvNet


class SRDDCAgent:

    def __init__(self,
                 state_space,
                 action_space,
                 buffer_size,
                 batch_size,
                 lr,
                 training_interval,
                 epsilon,
                 epsilon_decay,
                 epsilon_min,
                 plot_dir,
                 device):
        self.buffer = deque(maxlen=buffer_size)

    def step(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self._train()

    def plan(self, state):
        return 0

    def _train(self):
        pass