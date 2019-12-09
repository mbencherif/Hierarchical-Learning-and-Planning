import gym
from src.agents.dqn.dqn_agent import DQNAgent


def create_dqn_agent(env,
                     device,
                     layer_param=None,
                     buffer_size=int(1e5),
                     batch_size=128,
                     gamma=0.99,
                     tau=1e-3,
                     lr=1e-4,
                     training_interval=2,
                     epsilon=0.999,
                     epsilon_decay=0.9999,
                     epsilon_min=0.01,
                     **unused_kwargs):

    layer_param = layer_param or [
        {"type": "linear", "n_neurons": [0, 128]},
        {"type": "relu"},
        {"type": "linear", "n_neurons": [128, 128]},
        {"type": "relu"},
        {"type": "linear", "n_neurons": [128, 0]}]

    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     gamma=gamma,
                     tau=tau,
                     lr=lr,
                     training_interval=training_interval,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     layer_param=layer_param,
                     device=device)
    return agent
