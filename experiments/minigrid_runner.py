import time
import gym
import gym_minigrid
import numpy as np
from absl import logging
from pathlib2 import Path

from experiments.create_agents import create_dqn_minigrid_agent
from src.utils.plotting import plot_rewards


def run_minigrid(agent_param,
                 save_dir,
                 env_id,
                 n_episodes=10000,
                 max_ep_steps=500,
                 logging_interval=100,
                 device="cuda"):
    env = gym.make(env_id)
    observation_dim = env.observation_space.spaces["image"].shape
    agent = create_dqn_minigrid_agent(env=env,
                                      custom_state_space=observation_dim,
                                      custom_action_space=env.action_space.n,
                                      device=device,
                                      plot_dir=save_dir,
                                      **agent_param)
    agent.load(model_path=Path(save_dir))

    t0 = time.time()
    rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(max_ep_steps):
            action = agent.plan(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state

            if done or step >= max_ep_steps - 1:
                if episode % logging_interval == 0:
                    rewards.append(ep_reward)
                    logging.info(f"Cum. Reward in Ep {episode}:"
                                 f"\t{ep_reward:.2f},"
                                 f"\tEpsilon: {agent.epsilon:.2f},"
                                 f"\ttook {time.time() - t0:.2f} sec")
                    agent.save(model_path=Path(save_dir))
                    plot_rewards(rewards, save_dir)
                    t0 = time.time()

                break