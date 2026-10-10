"""
main.py - Entry Point for Training, Evaluation, Comparison, and GIF Rendering.
Farama Gymnasium Custom Reinforcement Learning Agent Implementation.
"""

import argparse
import os
import sys

# Add current directory to path so src modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import numpy as np

from src.agent import QLearningAgent, SARSAAgent
from src.utils import (
    evaluate_agent,
    plot_comparison,
    plot_training_curves,
    record_policy_gif,
)


def train_q_learning(
    env_name="FrozenLake-v1",
    is_slippery=False,
    episodes=2000,
    lr=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    seed=42,
):
    """Train a Q-Learning agent on the specified environment."""
    env = gym.make(env_name, is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    lengths_history = []
    td_errors_history = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + ep if seed else None)
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        ep_td_errors = []

        while not (terminated or truncated):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            td_err = agent.update(state, action, reward, next_state, terminated)
            ep_td_errors.append(td_err)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        lengths_history.append(steps)
        td_errors_history.append(np.mean(ep_td_errors) if ep_td_errors else 0.0)

        if ep % 200 == 0 or ep == episodes:
            avg_rew = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            print(f"[Q-Learning] Episode {ep:4d}/{episodes} | Avg Reward (last 100): {avg_rew:0.2f} | Epsilon: {agent.epsilon:0.4f}")

    env.close()
    return agent, rewards_history, lengths_history, td_errors_history


def train_sarsa(
    env_name="FrozenLake-v1",
    is_slippery=False,
    episodes=2000,
    lr=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    seed=42,
):
    """Train a SARSA agent for comparative analysis."""
    env = gym.make(env_name, is_slippery=is_slippery)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = SARSAAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    lengths_history = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + ep if seed else None)
        action = agent.choose_action(state, training=True)
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = agent.choose_action(next_state, training=True)

            agent.update(state, action, reward, next_state, next_action, terminated)

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        lengths_history.append(steps)

    env.close()
    return agent, rewards_history, lengths_history


def run_experiment(
    env_name="FrozenLake-v1",
    is_slippery=False,
    episodes=2000,
    save_dir="results",
):
    """Run full training, hyperparameter sensitivity, evaluation, and GIF generation."""
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 65)
    print(f"[+] Training Reinforcement Learning Agent on '{env_name}'")
    print(f"[+] Output Directory: '{save_dir}'")
    print("=" * 65)

    # 1. Main Training (Q-Learning)
    agent_ql, ql_rewards, ql_lengths, ql_td_errors = train_q_learning(
        env_name=env_name,
        is_slippery=is_slippery,
        episodes=episodes,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    # Save Q-Table
    q_table_path = os.path.join(save_dir, "q_table.npy")
    agent_ql.save(q_table_path)
    print(f"\n[Saved] Learned Q-Table saved to '{q_table_path}'")

    # 2. Plot Training Performance
    train_plot_path = os.path.join(save_dir, "training_curves.png")
    plot_training_curves(ql_rewards, ql_lengths, ql_td_errors, window=100, save_path=train_plot_path)

    # 3. Comparative Analysis (Q-Learning vs SARSA & Learning Rates)
    print("\n[+] Running Comparative Experiments (Q-Learning vs SARSA & Learning Rates)...")
    _, sarsa_rewards, _ = train_sarsa(env_name=env_name, is_slippery=is_slippery, episodes=episodes, lr=0.1)
    _, ql_lr05_rewards, _, _ = train_q_learning(env_name=env_name, is_slippery=is_slippery, episodes=episodes, lr=0.01)
    _, ql_lr50_rewards, _, _ = train_q_learning(env_name=env_name, is_slippery=is_slippery, episodes=episodes, lr=0.5)

    comparison_dict = {
        "Q-Learning (lr=0.10)": ql_rewards,
        "Q-Learning (lr=0.50)": ql_lr50_rewards,
        "Q-Learning (lr=0.01)": ql_lr05_rewards,
        "SARSA (lr=0.10)": sarsa_rewards,
    }
    comp_plot_path = os.path.join(save_dir, "comparison_curve.png")
    plot_comparison(comparison_dict, window=100, save_path=comp_plot_path)

    # 4. Evaluation
    print("\n[+] Evaluating Trained Q-Learning Agent (100 Test Episodes)...")
    eval_env = gym.make(env_name, is_slippery=is_slippery)
    mean_rew, success_rate, mean_steps = evaluate_agent(eval_env, agent_ql, num_episodes=100)
    eval_env.close()

    print("-" * 50)
    print(f"  Final Success Rate : {success_rate:.2f}%")
    print(f"  Mean Reward        : {mean_rew:.2f}")
    print(f"  Average Steps/Goal : {mean_steps:.2f}")
    print("-" * 50)

    # 5. Render & Save GIF
    gif_path = os.path.join(save_dir, "trained_agent.gif")
    print(f"\n[+] Rendering visual policy rollout to '{gif_path}'...")
    record_policy_gif(env_name, agent_ql, filename=gif_path, is_slippery=is_slippery)

    print("\n[SUCCESS] All deliverables successfully generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gymnasium RL Agent Implementation")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="Gymnasium environment name")
    parser.add_argument("--slippery", action="store_true", help="Enable stochastic transitions (slippery)")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save artifacts")
    args = parser.parse_args()

    run_experiment(
        env_name=args.env,
        is_slippery=args.slippery,
        episodes=args.episodes,
        save_dir=args.save_dir,
    )
