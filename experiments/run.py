import os
import time
import json
import argparse
from absl import logging
from pathlib2 import Path

import torch
import torch.optim as optim

from experiments.gym_runner import run_gym
from src.utils import read_config
from src.agents.dqn.dqn_agent import DQNAgent
from environments.stochastic_mdp.stochastic_mdp import StochasticMDP


def run(device, config, save_dir):
    env_id = "CartPole-v1"
    run_gym(agent_param=config, device=device, save_dir=save_dir, env_id=env_id)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_config(args.config_file)
    run(device=device, config=config, save_dir=args.save_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hierarchical Decision Making.")
    parser.add_argument("-l", "--logging",
                        help="Set logging verbosity: "
                             "'debug': print all; 'info': print info only",
                        default="info", type=str)
    parser.add_argument("-c", "--config_file",
                        help="Path or file to experiment config file(s). "
                             "E.g '/Robust-Robotic-Manipulation/experiments"
                             "/configs'",
                        default=f"{Path(os.getcwd()).parent / 'configs/dqn_dense.json'}")
    parser.add_argument("-d", "--save_directory",
                        help="The experiment output directory. "
                             "E.g.: ./experiment_results",
                        default=f"{Path(os.getcwd()) / 'results/dqn'}",
                        type=str)

    args = parser.parse_args()

    if args.logging is "debug":
        logging.set_verbosity(logging.DEBUG)
    elif args.logging is "info":
        logging.set_verbosity(logging.INFO)
    logging._warn_preinit_stderr = 0

    t0 = time.time()
    main(args)
    print(f"Execution took {time.time() - t0:.2f} seconds.")