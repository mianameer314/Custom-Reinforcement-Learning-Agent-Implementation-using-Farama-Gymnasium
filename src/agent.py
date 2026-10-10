"""
agent.py - Reinforcement Learning Agents (Q-Learning and SARSA)
Implemented from scratch using NumPy.
"""

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning Agent (Off-Policy TD Control).
    Updates Q-values using Bellman Optimality Equation:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size), dtype=np.float64)

    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy during training,
        and purely greedy policy during evaluation.
        """
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))

        # Break ties randomly among actions with maximum Q-value
        max_q = np.max(self.q_table[state])
        best_actions = np.where(self.q_table[state] == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> float:
        """
        Apply Bellman Optimality update rule.
        Returns the TD error magnitude for monitoring.
        """
        target = reward if terminated else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(abs(td_error))

    def decay_epsilon(self) -> None:
        """Decay exploration rate down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table to a .npy file."""
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        """Load Q-table from a .npy file."""
        self.q_table = np.load(filepath)


class SARSAAgent:
    """
    Tabular SARSA Agent (On-Policy TD Control).
    Updates Q-values using the on-policy action a':
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((state_size, action_size), dtype=np.float64)

    def choose_action(self, state: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))

        max_q = np.max(self.q_table[state])
        best_actions = np.where(self.q_table[state] == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        terminated: bool,
    ) -> float:
        target = reward if terminated else reward + self.gamma * self.q_table[next_state, next_action]
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        return float(abs(td_error))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        np.save(filepath, self.q_table)

    def load(self, filepath: str) -> None:
        self.q_table = np.load(filepath)
