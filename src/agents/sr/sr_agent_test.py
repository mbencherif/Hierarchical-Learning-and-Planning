import time
import gym
import gym_minigrid
from absl import logging

from src.utils.plotting import plot_rewards

from src.agents.sr_ddc.sr_ddc_agent import SRDDCAgent


def test(n_episodes = 10000,
         max_ep_steps = 10000,
         logging_interval = 100,
         env_id="MiniGrid-Empty-5x5-v0",
         save_dir="./",
         device = "cuda"):
    env = gym.make(env_id)
    agent = SRDDCAgent()

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
                                 f"\ttook {time.time() - t0:.2f} sec")
                    plot_rewards(rewards, save_dir)
                    t0 = time.time()

                break