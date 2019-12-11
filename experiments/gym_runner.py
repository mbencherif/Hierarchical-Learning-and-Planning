import time
import gym
import numpy as np
from absl import logging
from pathlib2 import Path


from experiments.create_agents import create_dqn_gym_agent
from src.utils.plotting import plot_rewards


def run_gym(agent_param,
            save_dir,
            env_id,
            n_episodes=10000,
            max_ep_steps=500,
            logging_interval=100,
            device="cpu"):
    env = gym.make(env_id)
    agent = create_dqn_gym_agent(env=env, device=device, plot_dir=save_dir, **agent_param)
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


def run_gym_continuous(agent_param,
                       save_dir,
                       env_id,
                       n_episodes=10000,
                       max_ep_steps=1000,
                       logging_interval=1,
                       device="cuda"):
    action_map = [
        [0, 0, 0],
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
        [-0.1, 0, 0],
        [0, -0.1, 0],
        [0, 0, -0.1]]
    env = gym.make(env_id)

    agent = create_dqn_gym_agent(env=env,
                                 device=device,
                                 plot_dir=save_dir,
                                 custom_state_space=env.observation_space.shape[-1],
                                 custom_action_space=np.asarray(action_map),
                                 **agent_param)
    agent.load(model_path=Path(save_dir))

    t0 = time.time()
    rewards = []
    for episode in range(n_episodes):
        env = gym.make(env_id)
        state = env.reset()
        ep_reward = 0
        action_arr = np.zeros(env.action_space.shape[0])
        for step in range(max_ep_steps):
            action = agent.plan(state)
            action_arr += action_map[action]
            action_arr.clip(min=env.action_space.low,
                            max=env.action_space.high)
            next_state, reward, done, _ = env.step(action_arr)
            agent.step(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state

            if done or step >= max_ep_steps - 1:
                if episode % logging_interval == 0:
                    rewards.append(ep_reward)
                    logging.info(f"Cum. Reward in Ep {episode} (Step {step}):"
                                 f"\t{ep_reward:.2f},"
                                 f"\tEpsilon: {agent.epsilon:.2f},"
                                 f"\ttook {time.time() - t0:.2f} sec")
                    agent.save(model_path=Path(save_dir))
                    plot_rewards(rewards, save_dir)
                    t0 = time.time()

                break
