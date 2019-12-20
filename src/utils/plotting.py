import numpy as np
import matplotlib.pyplot as plt


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_train_progress(data, save_dir):
    """
    data: [step_nr, reward, loss, epsilon]
    """
    data = np.asarray(data)
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(8, 8)
    ax0 = fig.add_subplot(gs[:3, :])
    ax0.set_title("Reward for each step")
    ax0.set_xlabel("Game Steps")
    ax0.set_ylabel("Reward")
    ax0.plot(data[:, 0][::10], data[:, 1][::10])

    losses = smooth(data[:, 2], 1)
    ax1 = fig.add_subplot(gs[3:6, :])
    ax1.set_title("Q-Network Training Loss (Smoothed)")
    ax1.set_xlabel("Updates")
    ax1.set_ylabel("TD Error Loss")
    # ax1.set_yscale("log")
    ax1.plot(data[:, 0][::100], losses[::100])

    ax2 = fig.add_subplot(gs[6:, :])
    ax2.set_title("Epsilon decay over time")
    ax2.set_xlabel("Game Steps")
    ax2.set_ylabel("Epsilon")
    ax2.plot(data[:, 0][::100], data[:, 3][::100], c="grey")
    fig.savefig(save_dir)
    plt.close(fig)

def plot_rewards(rewards, save_dir):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(rewards)), rewards)
    ax.set_title("Reward during training")
    ax.set_ylabel("Total Reward / Episode")
    ax.set_xlabel("Episodes [100]")
    fig.savefig(save_dir + "/reward.png")
    plt.close(fig)