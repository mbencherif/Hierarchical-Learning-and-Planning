import gym
from src.agents.dqn_gym.dqn_agent import DQNAgent
from src.agents.dqn_minigrid.dqn_agent import DQNAgentMinigrid


def create_dqn_gym_agent(env,
                         device,
                         plot_dir,
                         custom_state_space=None,
                         custom_action_space=None,
                         q_net_layers=None,
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
    action_space = custom_action_space if custom_action_space is not None else env.action_space
    state_space = custom_state_space if custom_state_space is not None else env.observation_space
    agent = DQNAgent(state_space=state_space,
                     action_space=action_space,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     gamma=gamma,
                     tau=tau,
                     lr=lr,
                     training_interval=training_interval,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     layer_param=q_net_layers,
                     plot_dir=plot_dir,
                     device=device)
    return agent


def create_dqn_minigrid_agent(env,
                              device,
                              plot_dir,
                              custom_state_space=None,
                              custom_action_space=None,
                              q_net_layers=None,
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
    action_space = custom_action_space if custom_action_space is not None else env.action_space
    state_space = custom_state_space if custom_state_space is not None else env.observation_space
    agent = DQNAgentMinigrid(state_space=state_space,
                             action_space=action_space,
                             buffer_size=buffer_size,
                             batch_size=batch_size,
                             gamma=gamma,
                             tau=tau,
                             lr=lr,
                             training_interval=training_interval,
                             epsilon=epsilon,
                             epsilon_decay=epsilon_decay,
                             epsilon_min=epsilon_min,
                             layer_param=q_net_layers,
                             plot_dir=plot_dir,
                             device=device)
    return agent
