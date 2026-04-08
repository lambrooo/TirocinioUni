"""
Q-Learning Agent for IIoT Security
====================================

Implements a tabular Q-Learning agent as a baseline comparison
for the Active Inference agents in the thesis.

Q-Learning is a model-free reinforcement learning algorithm that
learns action-values directly, unlike Active Inference which uses
a generative model of the environment.

Key differences from Active Inference:
- Model-free: No explicit transition model (B matrix)
- Value-based: Learns Q(s,a) instead of beliefs over states
- No epistemic value: Pure reward maximization
- Simpler but less sample efficient
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent for IIoT motor control.

    Uses discretized state space similar to Active Inference agent
    for fair comparison.

    State Space:
    - Temperature: [Low, Optimal, High]
    - Motor Temperature: [Safe, Overheating]
    - Load: [Low, High]
    - Cyber Alert: [Safe, Alert]

    Action Space:
    - Thermal: [Cool, Maintain, Heat]
    - Load: [Decrease, Maintain, Increase]
    - Verify: [Wait, Verify]
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_decay=0.999,
        epsilon_min=0.01,
    ):
        """
        Initialize Q-Learning agent.

        Args:
            learning_rate: Alpha parameter for Q-value updates
            discount_factor: Gamma parameter for future reward discounting
            epsilon: Initial exploration rate
            epsilon_decay: Multiplicative decay for epsilon
            epsilon_min: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # State discretization (same as Active Inference for fair comparison)
        self.n_temp_states = 3  # Low, Optimal, High
        self.n_motor_states = 2  # Safe, Overheating
        self.n_load_states = 2  # Low, High
        self.n_cyber_states = 2  # Safe, Alert

        # Action space
        self.n_thermal_actions = 3  # Cool, Maintain, Heat
        self.n_load_actions = 3  # Decrease, Maintain, Increase
        self.n_verify_actions = 2  # Wait, Verify

        # Total state/action dimensions
        self.n_states = (
            self.n_temp_states
            * self.n_motor_states
            * self.n_load_states
            * self.n_cyber_states
        )
        self.n_actions = (
            self.n_thermal_actions * self.n_load_actions * self.n_verify_actions
        )

        # Q-table: state -> action -> value
        # Using defaultdict for sparse initialization
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

        # For tracking
        self.prev_state = None
        self.prev_action = None
        self.total_updates = 0
        self.current_step = 0

        # Learning history for analysis
        self.learning_history = {
            "step": [],
            "epsilon": [],
            "avg_q_value": [],
            "max_q_value": [],
            "td_error": [],
        }
        self.recent_td_errors = []

    def _discretize_state(self, temp, motor_temp, load, attack_detected):
        """Convert continuous observations to discrete state index."""
        # Temperature discretization
        if temp < 40:
            temp_state = 0  # Low
        elif temp <= 70:
            temp_state = 1  # Optimal
        else:
            temp_state = 2  # High

        # Motor temperature discretization
        motor_state = 0 if motor_temp < 80 else 1

        # Load discretization
        load_state = 0 if load < 30 else 1

        # Cyber alert
        cyber_state = 1 if attack_detected else 0

        # Combine into single state index
        state = (
            temp_state
            * (self.n_motor_states * self.n_load_states * self.n_cyber_states)
            + motor_state * (self.n_load_states * self.n_cyber_states)
            + load_state * self.n_cyber_states
            + cyber_state
        )

        return state, (temp_state, motor_state, load_state, cyber_state)

    def _action_to_indices(self, action_idx):
        """Convert flat action index to individual action components."""
        verify_action = action_idx % self.n_verify_actions
        action_idx //= self.n_verify_actions
        load_action = action_idx % self.n_load_actions
        action_idx //= self.n_load_actions
        thermal_action = action_idx

        return thermal_action, load_action, verify_action

    def _indices_to_action(self, thermal, load, verify):
        """Convert individual action components to flat action index."""
        return (
            thermal * self.n_load_actions * self.n_verify_actions
            + load * self.n_verify_actions
            + verify
        )

    def _compute_reward(
        self, temp_state, motor_state, load_state, cyber_state, action_tuple
    ):
        """
        Compute reward based on state and action.

        Reward structure mirrors Active Inference preferences:
        - Prefer optimal temperature
        - Strongly prefer safe motor temperature
        - Prefer high load (productivity)
        - Prefer no cyber alerts
        - Penalize unnecessary verification
        """
        thermal_action, load_action, verify_action = action_tuple

        reward = 0.0

        # Temperature reward (prefer optimal)
        if temp_state == 1:  # Optimal
            reward += 2.0
        elif temp_state == 2:  # High
            reward -= 1.0

        # Motor temperature reward (critical for safety)
        if motor_state == 0:  # Safe
            reward += 3.0
        else:  # Overheating
            reward -= 5.0  # Strong penalty for overheating

        # Load reward (productivity)
        if load_state == 1:  # High
            reward += 1.5

        # Cyber safety
        if cyber_state == 1:  # Alert
            reward -= 1.0

        # Verification cost
        if verify_action == 1:
            reward -= 2.0  # Opportunity cost

        return reward

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state index

        Returns:
            int: Action index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done=False):
        """
        Update Q-value using TD learning.

        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether episode is done
        """
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.Q[next_state])

        td_error = target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

        self.recent_td_errors.append(abs(td_error))
        self.total_updates += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def step(
        self, temp_reading, motor_temp_reading, load_reading, attack_detected=False
    ):
        """
        Main step function compatible with simulation interface.

        Returns: temp_cmd, load_cmd, verify_cmd
        """
        # Discretize current state
        state, state_tuple = self._discretize_state(
            temp_reading, motor_temp_reading, load_reading, attack_detected
        )

        # Update Q-values from previous transition
        if self.prev_state is not None:
            prev_action_tuple = self._action_to_indices(self.prev_action)
            reward = self._compute_reward(
                state_tuple[0],
                state_tuple[1],
                state_tuple[2],
                state_tuple[3],
                prev_action_tuple,
            )
            self.update(self.prev_state, self.prev_action, reward, state)

        # Select action
        action = self.select_action(state)
        thermal_action, load_action, verify_action = self._action_to_indices(action)

        # Store for next update
        self.prev_state = state
        self.prev_action = action
        self.current_step += 1

        # Record history periodically
        if self.current_step % 10 == 0:
            self._record_history()

        # Map actions to control commands
        thermal_vals = [-5.0, 0.0, 5.0]
        load_vals = [-5.0, 0.0, 5.0]

        temp_cmd = thermal_vals[thermal_action]
        load_cmd = load_vals[load_action]
        verify_cmd = verify_action == 1

        return temp_cmd, load_cmd, verify_cmd

    def _record_history(self):
        """Record learning metrics."""
        self.learning_history["step"].append(self.current_step)
        self.learning_history["epsilon"].append(self.epsilon)

        # Average Q-value across visited states
        if self.Q:
            all_q = [np.max(q) for q in self.Q.values()]
            self.learning_history["avg_q_value"].append(np.mean(all_q))
            self.learning_history["max_q_value"].append(np.max(all_q))
        else:
            self.learning_history["avg_q_value"].append(0)
            self.learning_history["max_q_value"].append(0)

        # Average TD error
        if self.recent_td_errors:
            self.learning_history["td_error"].append(
                np.mean(self.recent_td_errors[-100:])
            )
        else:
            self.learning_history["td_error"].append(0)

    def get_learning_stats(self):
        """Return learning statistics for compatibility with simulation."""
        return {
            "total_updates": self.total_updates,
            "current_step": self.current_step,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "n_visited_states": len(self.Q),
            "avg_td_error": np.mean(self.recent_td_errors[-100:])
            if self.recent_td_errors
            else 0,
            "model_divergence": 0.0,  # N/A for Q-learning (model-free)
            "avg_prediction_error": np.mean(self.recent_td_errors[-100:])
            if self.recent_td_errors
            else 0,
        }

    def get_learning_history_df(self):
        """Return learning history as DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.learning_history)

    def reset_learning(self):
        """Reset learning state."""
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.prev_state = None
        self.prev_action = None
        self.total_updates = 0
        self.current_step = 0
        self.epsilon = 0.1
        self.recent_td_errors = []
        self.learning_history = {k: [] for k in self.learning_history.keys()}

    def save_model(self, filepath):
        """Save Q-table to disk."""
        import os

        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        # Convert Q-table to arrays
        states = list(self.Q.keys())
        values = np.array([self.Q[s] for s in states])

        np.savez(
            filepath,
            states=np.array(states),
            values=values,
            epsilon=np.array([self.epsilon]),
            total_updates=np.array([self.total_updates]),
            current_step=np.array([self.current_step]),
        )

        print(f"Q-Learning model saved to {filepath}")
        return filepath

    def load_model(self, filepath, continue_learning=True):
        """Load Q-table from disk."""
        if not filepath.endswith(".npz"):
            filepath = filepath + ".npz"

        data = np.load(filepath)

        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        for s, v in zip(data["states"], data["values"]):
            self.Q[int(s)] = v

        if continue_learning:
            self.total_updates = int(data["total_updates"][0])
            self.current_step = int(data["current_step"][0])
            self.epsilon = float(data["epsilon"][0])
        else:
            self.epsilon = 0.0  # Pure exploitation

        print(f"Q-Learning model loaded from {filepath}")
        return {"n_states": len(self.Q), "continue_learning": continue_learning}


class DoubleQLearningAgent(QLearningAgent):
    """
    Double Q-Learning agent to reduce overestimation bias.

    Uses two Q-tables and alternates updates to reduce the
    maximization bias inherent in standard Q-learning.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Second Q-table
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))

    def select_action(self, state):
        """Select action using combined Q-values."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Use average of both Q-tables for action selection
            combined_q = self.Q[state] + self.Q2[state]
            return np.argmax(combined_q)

    def update(self, state, action, reward, next_state, done=False):
        """Update Q-value using Double Q-learning."""
        # Randomly choose which Q-table to update
        if np.random.random() < 0.5:
            # Update Q1 using Q2's value estimate
            if done:
                target = reward
            else:
                best_action = np.argmax(self.Q[next_state])
                target = (
                    reward + self.discount_factor * self.Q2[next_state][best_action]
                )

            td_error = target - self.Q[state][action]
            self.Q[state][action] += self.learning_rate * td_error
        else:
            # Update Q2 using Q1's value estimate
            if done:
                target = reward
            else:
                best_action = np.argmax(self.Q2[next_state])
                target = reward + self.discount_factor * self.Q[next_state][best_action]

            td_error = target - self.Q2[state][action]
            self.Q2[state][action] += self.learning_rate * td_error

        self.recent_td_errors.append(abs(td_error))
        self.total_updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_learning(self):
        super().reset_learning()
        self.Q2 = defaultdict(lambda: np.zeros(self.n_actions))


if __name__ == "__main__":
    print("Testing Q-Learning agent...")

    agent = QLearningAgent(learning_rate=0.1, epsilon=0.1)

    for step in range(500):
        temp = 50 + np.random.randn() * 15
        motor = 60 + np.random.randn() * 20
        load = 30 + np.random.randn() * 10
        attack = np.random.random() < 0.05

        temp_cmd, load_cmd, verify_cmd = agent.step(temp, motor, load, attack)

    stats = agent.get_learning_stats()
    print(f"\nLearning Stats:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  States visited: {stats['n_visited_states']}")
    print(f"  Final epsilon: {stats['epsilon']:.4f}")
    print(f"  Avg TD error: {stats['avg_td_error']:.4f}")

    agent.save_model("test_qlearning_model")

    agent2 = QLearningAgent()
    agent2.load_model("test_qlearning_model")
    print(f"\nLoaded model has {len(agent2.Q)} states")

    print("\nQ-Learning agent test completed.")
