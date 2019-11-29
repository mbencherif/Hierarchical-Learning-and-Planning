import gym
import gym_minigrid
import unittest
import numpy as np
import torch

from src.agents.sr.dsr_agent import DSRAgent


class SRAgentTest(unittest.TestCase):

  def train(self, env, agent, n_episodes, max_ep_steps):
    episode_rewards = []

    for episode in range(n_episodes):
      state = env.reset()
      episode_reward = 0

      for step in range(max_ep_steps):
        action = agent.plan(obs=state)
        next_state, reward, done, _ = env.step(action)

        agent.store(state, action, reward, next_state, done)

        episode_reward += reward

        #if step % 4 == 0:
        agent.update()

        if done or step == max_ep_steps - 1:
          avrg_reward = episode_reward / step
          episode_rewards.append(avrg_reward)
          if episode % 100 == 0:
            print(f"Ep {episode},\tAvrg Reward {avrg_reward:.3f}\t"
                  f"Current Epsilon {agent.epsilon:.4f}")
          break

        state = next_state

    return episode_rewards

  def test_dqn_on_cartpole(self):
    device = "cpu"
    env_id = "MiniGrid-Empty-5x5-v0"

    n_episodes = 100000
    max_ep_steps = 1000

    learning_rate = 2.5e-4
    momentum = 0.95
    gamma = 0.99
    epsilon = 1
    epsilon_decay = 0.9999
    epsilon_min = 0.1

    env = gym.make(env_id)
    agent = DSRAgent(env=env,
                     latent_state_dim=64,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     buffer_size=int(1e6),
                     batch_size=128,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     device=device)
    episode_rewards = self.train(env=env,
                                 agent=agent,
                                 n_episodes=n_episodes,
                                 max_ep_steps=max_ep_steps)

    avrg_r_first_half = np.mean(np.array(
      episode_rewards[:len(episode_rewards) / 2]))
    avrg_r_second_half = np.mean(np.array(
      episode_rewards[len(episode_rewards) / 2:]))
    # Check if avrg reward is at least double compared to the first half
    assert avrg_r_first_half * 2 >= avrg_r_second_half


if __name__ == '__main__':
  unittest.main()
