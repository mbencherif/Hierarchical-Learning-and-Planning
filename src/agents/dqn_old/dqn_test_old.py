import time
import gym
import unittest
import numpy as np
import torch

from src.agents.dqn.dqn_agent import DQNAgent


class DQNAgentTest(unittest.TestCase):

  def mini_batch_train(self, env, agent, n_episodes, max_ep_steps, batch_size,
                       epsilon, epsilon_decay, epsilon_min, logging_interval=100):
    t0 = time.time()
    for episode in range(n_episodes):
      state = env.reset()
      episode_rewards = []
      ep_steps = []

      for step in range(max_ep_steps):
        action = agent.plan(obs=state, eps=epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > batch_size and step % 2 == 0:
          agent.update(batch_size)

        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)
        state = next_state
        episode_rewards.append(reward)

        if done or step >= max_ep_steps - 1:
          ep_steps.append(step + 1.)
          if episode % logging_interval == 0:
            avrg_rewards = np.sum(episode_rewards)
            avrg_steps = np.mean(np.asarray(ep_steps))
            print(f"Ep {episode},\tAvrg Reward {avrg_rewards:.2f},"
                  f"\tAvrg Reward: {avrg_steps},\t"
                  f"\tEpsilon {epsilon:.2f},\ttook {time.time() - t0:.2f} sec")
            t0 = time.time()
          break


    return episode_rewards

  def test_dqn_on_cartpole(self):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    env_id = "CartPole-v1"
    n_episodes = 10000
    max_ep_steps = 1000
    batch_size = 128
    learning_rate = 5e-4
    tau = 1e-3
    gamma = 0.99
    buffer_size = int(1e5)
    epsilon = 0.999
    epsilon_decay = 0.99995
    epsilon_min = 0.01

    env = gym.make(env_id)

    layer_param = [
          {"type": "linear", "n_neurons": [0, 128]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [128, 128]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [128, 0]}
        ]
    agent = DQNAgent(env=env,
                     layer_param=layer_param,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     tau=tau,
                     buffer_size=buffer_size,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     device=device)

    episode_rewards = self.mini_batch_train(env=env,
                                            agent=agent,
                                            n_episodes=n_episodes,
                                            max_ep_steps=max_ep_steps,
                                            batch_size=batch_size,
                                            epsilon=epsilon,
                                            epsilon_decay=epsilon_decay,
                                            epsilon_min=epsilon_min)

    avrg_r_first_half = np.mean(np.array(
      episode_rewards[:len(episode_rewards) / 2]))
    avrg_r_second_half = np.mean(np.array(
      episode_rewards[len(episode_rewards) / 2:]))
    # Check if avrg reward is at least double compared to the first half
    assert avrg_r_first_half * 2 >= avrg_r_second_half


if __name__ == '__main__':
  unittest.main()
