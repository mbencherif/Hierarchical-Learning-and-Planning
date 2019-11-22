import gym
import unittest
import numpy as np

from src.agents.dqn.dqn_agent import DQNAgent


class DQNAgentTest(unittest.TestCase):

  def mini_batch_train(self, env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
      state = env.reset()
      episode_reward = 0

      for step in range(max_steps):
        action = agent.plan(obs=state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
          agent.update(batch_size)

        if done or step == max_steps - 1:
          episode_rewards.append(episode_reward)
          if episode % 100 == 0:
            print(f"Ep {episode}: Reward {episode_reward} "
                  f"Current Epsilon {agent.epsilon:.5f}")

          break

        state = next_state

    return episode_rewards

  def test_dqn_on_cartpole(self):
    env_id = "CartPole-v0"
    MAX_EPISODES = 1000
    MAX_STEPS = 500
    BATCH_SIZE = 1

    env = gym.make(env_id)

    layer_param = [
          {"type": "linear", "n_neurons": [0, 256]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [256, 256]},
          {"type": "relu"},
          {"type": "linear", "n_neurons": [256, 0]},
          {"type": "tanh"}  # TODO (#4): is tanh needed?
        ]

    agent = DQNAgent(env, layer_param)

    episode_rewards = self.mini_batch_train(
      env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

    avrg_r_first_half = np.mean(np.array(
      episode_rewards[:len(episode_rewards) / 2]))
    avrg_r_second_half = np.mean(np.array(
      episode_rewards[len(episode_rewards) / 2:]))
    # Check if avrg reward is at least double compared to the first half
    assert avrg_r_first_half * 2 >= avrg_r_second_half


if __name__ == '__main__':
  unittest.main()
