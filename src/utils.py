"""
utils.py - Visualization, Evaluation, and GIF Recording Utilities
"""

import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def moving_average(data, window_size=100):
    """Compute simple moving average over a 1D sequence."""
    if len(data) < window_size:
        return data
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_training_curves(
    rewards,
    lengths,
    td_errors=None,
    window=100,
    save_path="training_curves.png",
):
    """
    Plot learning curves: episodic reward, rolling average, and episode steps.
    """
    fig, axes = plt.subplots(1, 2 if td_errors is None else 3, figsize=(15 if td_errors else 11, 4.5))

    # Reward Plot
    axes[0].plot(rewards, alpha=0.25, color="steelblue", label="Episodic Reward")
    avg_rewards = moving_average(rewards, window)
    axes[0].plot(
        range(window - 1, len(rewards)),
        avg_rewards,
        color="darkblue",
        linewidth=2,
        label=f"{window}-Ep Moving Avg",
    )
    axes[0].set_title("Training Reward Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Episode Lengths
    axes[1].plot(lengths, alpha=0.3, color="coral", label="Steps per Episode")
    avg_lengths = moving_average(lengths, window)
    axes[1].plot(
        range(window - 1, len(lengths)),
        avg_lengths,
        color="crimson",
        linewidth=2,
        label=f"{window}-Ep Moving Avg",
    )
    axes[1].set_title("Episode Length (Steps to Goal/Fail)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # TD Error (if provided)
    if td_errors is not None:
        axes[2].plot(td_errors, alpha=0.2, color="teal", label="Mean TD Error")
        avg_errors = moving_average(td_errors, window)
        axes[2].plot(
            range(window - 1, len(td_errors)),
            avg_errors,
            color="darkgreen",
            linewidth=2,
            label=f"{window}-Ep Moving Avg",
        )
        axes[2].set_title("TD Error Convergence", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("TD Error")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Training plot saved to: {save_path}")
    plt.close()


def plot_comparison(results_dict, window=100, save_path="comparison_curve.png"):
    """
    Plot comparative learning curves for multiple agents or hyperparameter setups.
    """
    plt.figure(figsize=(9, 5))
    for label, rewards in results_dict.items():
        avg = moving_average(rewards, window)
        plt.plot(range(window - 1, len(rewards)), avg, label=label, linewidth=2)

    plt.title("Algorithm / Hyperparameter Comparison (Moving Avg Reward)", fontsize=12, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel(f"{window}-Episode Average Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] Comparison plot saved to: {save_path}")
    plt.close()


def evaluate_agent(env, agent, num_episodes=100):
    """
    Evaluate trained agent using deterministic greedy policy.
    Returns success rate (mean reward) and average steps.
    """
    total_rewards = []
    total_steps = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0
        steps = 0

        while not (terminated or truncated):
            action = agent.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            ep_reward += reward
            steps += 1

        total_rewards.append(ep_reward)
        total_steps.append(steps)

    mean_reward = float(np.mean(total_rewards))
    success_rate = float(np.mean([1 if r > 0 else 0 for r in total_rewards])) * 100.0
    mean_steps = float(np.mean(total_steps))
    return mean_reward, success_rate, mean_steps


def render_grid_frame(state, total_states=16, step=0, total_reward=0, title="FrozenLake-v1 (Q-Learning Agent)"):
    """
    Generate a high-quality frame image representing the FrozenLake grid state.
    """
    grid_size = 4
    fig, ax = plt.subplots(figsize=(6, 7), dpi=100)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size + 0.8)
    ax.axis("off")

    lake_map = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]

    colors = {
        "S": "#81D4FA",  # Light Blue
        "F": "#E1F5FE",  # Frozen Ice
        "H": "#455A64",  # Danger Dark Hole
        "G": "#A5D6A7",  # Emerald Goal
    }

    agent_row = state // grid_size
    agent_col = state % grid_size

    for r in range(grid_size):
        for c in range(grid_size):
            tile_type = lake_map[r][c]
            plot_y = grid_size - 1 - r
            plot_x = c

            rect = plt.Rectangle(
                (plot_x, plot_y), 1, 1,
                facecolor=colors[tile_type],
                edgecolor="#90A4AE",
                linewidth=2,
            )
            ax.add_patch(rect)

            label_text = tile_type
            if tile_type == "S":
                label_text = "START\n(S)"
            elif tile_type == "H":
                label_text = "HOLE\n(H)"
            elif tile_type == "G":
                label_text = "GOAL\n(G)"
            elif tile_type == "F":
                label_text = "F"

            text_color = "white" if tile_type == "H" else "#263238"
            ax.text(
                plot_x + 0.5, plot_y + 0.5, label_text,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=text_color,
            )

            if r == agent_row and c == agent_col:
                circle = plt.Circle(
                    (plot_x + 0.5, plot_y + 0.5), 0.32,
                    facecolor="#FF7043", edgecolor="#D84315", linewidth=3, zorder=5
                )
                ax.add_patch(circle)
                ax.text(
                    plot_x + 0.5, plot_y + 0.5, "AGENT",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=6
                )

    ax.text(
        grid_size / 2, grid_size + 0.5, title,
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1A237E"
    )
    ax.text(
        grid_size / 2, grid_size + 0.15,
        f"Step: {step:02d} | Current State: {state:02d} (Row: {agent_row}, Col: {agent_col}) | Reward: {total_reward:0.1f}",
        ha="center", va="center", fontsize=10, color="#37474F",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1", edgecolor="#CFD8DC")
    )

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return Image.fromarray(img_array)


def record_policy_gif(env_name, agent, filename="trained_agent.gif", is_slippery=False, max_steps=100):
    """
    Render and record an evaluation episode of the trained agent and save as an animated GIF.
    """
    env = gym.make(env_name, is_slippery=is_slippery)
    state, _ = env.reset(seed=42)
    frames = []
    total_reward = 0

    frames.append(render_grid_frame(state, step=0, total_reward=total_reward))

    for step in range(1, max_steps + 1):
        action = agent.choose_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward

        frames.append(render_grid_frame(state, step=step, total_reward=total_reward))

        if terminated or truncated:
            for _ in range(3):
                frames.append(render_grid_frame(state, step=step, total_reward=total_reward))
            break

    env.close()

    if frames:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=400,
            loop=0,
        )
        print(f"[Saved] Agent rollout GIF saved to: {filename} ({len(frames)} frames)")
    else:
        print("[Warning] No frames captured during render.")
