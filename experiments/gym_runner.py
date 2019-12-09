import time
import gym
import unittest
import numpy as np
from pathlib2 import Path
import torch

from experiments.create_agents import create_dqn_agent

# TODO: add saveing model and results


def run_gym(agent_param,
            save_dir,
            env_id,
            n_episodes=10000,
            max_ep_steps=500,
            logging_interval=100,
            device="cpu"):

    env = gym.make(env_id)
    agent = create_dqn_agent(env=env, device=device, **agent_param)
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
                    print(f"Cum. Reward in {episode}. Episode:\t{ep_reward:.2f},"
                          f"\tEpsilon: {agent.epsilon:.2f},\ttook {time.time() - t0:.2f} sec")
                    agent.save(model_path=Path(save_dir))
                    t0 = time.time()

                break
