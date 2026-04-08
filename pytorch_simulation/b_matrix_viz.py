"""
B Matrix Visualization Module
==============================

Provides functions for visualizing the B matrix (transition dynamics)
of the Active Inference agent, including:
- Static heatmaps for thesis figures
- Animation of B matrix evolution during learning
- Comparison between initial and learned matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns


def plot_b_matrix_heatmap(agent, factor_idx=0, action_idx=None, figsize=(12, 4)):
    """
    Plot heatmap of B matrix for a specific factor.

    Args:
        agent: AdaptiveActiveInferenceAgent instance
        factor_idx: Which factor to visualize (0-4)
        action_idx: Specific action to show (None = all actions)
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure
    """
    snapshot = agent.get_b_matrix_snapshot()
    factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]
    factor_name = factor_names[factor_idx]

    data = snapshot[factor_name]
    B = data["matrix"]
    B_init = data["initial_matrix"]
    actions = data["actions"]
    states = data["states"]

    n_actions = B.shape[2]

    if action_idx is not None:
        # Single action
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Initial
        sns.heatmap(
            B_init[:, :, action_idx],
            ax=axes[0],
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=states,
            yticklabels=states,
        )
        axes[0].set_title(f"Initial B[{factor_name}][{actions[action_idx]}]")
        axes[0].set_xlabel("Current State")
        axes[0].set_ylabel("Next State")

        # Learned
        sns.heatmap(
            B[:, :, action_idx],
            ax=axes[1],
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=states,
            yticklabels=states,
        )
        axes[1].set_title(f"Learned B[{factor_name}][{actions[action_idx]}]")
        axes[1].set_xlabel("Current State")
        axes[1].set_ylabel("Next State")

        # Difference
        diff = B[:, :, action_idx] - B_init[:, :, action_idx]
        norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
        sns.heatmap(
            diff,
            ax=axes[2],
            annot=True,
            fmt="+.2f",
            cmap="RdBu_r",
            center=0,
            norm=norm,
            xticklabels=states,
            yticklabels=states,
        )
        axes[2].set_title(f"Change (Learned - Initial)")
        axes[2].set_xlabel("Current State")
        axes[2].set_ylabel("Next State")

    else:
        # All actions
        fig, axes = plt.subplots(2, n_actions, figsize=(figsize[0], figsize[1] * 2))

        for a in range(n_actions):
            # Initial (top row)
            sns.heatmap(
                B_init[:, :, a],
                ax=axes[0, a],
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                xticklabels=states,
                yticklabels=states,
            )
            axes[0, a].set_title(f"Initial: {actions[a]}")
            if a == 0:
                axes[0, a].set_ylabel("Next State")

            # Learned (bottom row)
            sns.heatmap(
                B[:, :, a],
                ax=axes[1, a],
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                xticklabels=states,
                yticklabels=states,
            )
            axes[1, a].set_title(f"Learned: {actions[a]}")
            axes[1, a].set_xlabel("Current State")
            if a == 0:
                axes[1, a].set_ylabel("Next State")

    fig.suptitle(f"B Matrix: {factor_name} Factor", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_all_factors_summary(agent, figsize=(16, 12)):
    """
    Plot summary of B matrix changes for all factors.

    Args:
        agent: AdaptiveActiveInferenceAgent instance
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure
    """
    snapshot = agent.get_b_matrix_snapshot()
    factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for f, factor_name in enumerate(factor_names):
        ax = axes[f]
        data = snapshot[factor_name]

        # Calculate total change per action
        change = np.abs(data["change"])
        total_change_per_action = change.sum(axis=(0, 1))

        actions = data["actions"]
        colors = ["#3498db", "#2ecc71", "#e74c3c"][: len(actions)]

        bars = ax.bar(actions, total_change_per_action, color=colors)
        ax.set_title(f"{factor_name}")
        ax.set_ylabel("Total |Change|")
        ax.set_ylim(0, max(total_change_per_action.max() * 1.2, 0.1))

        # Add value labels
        for bar, val in zip(bars, total_change_per_action):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Use last subplot for legend/info
    axes[5].axis("off")
    stats = agent.get_learning_stats()
    info_text = f"""Learning Statistics
    
Total Updates: {stats["total_updates"]}
Current Step: {stats["current_step"]}
Learning Rate: {stats["learning_rate"]:.4f}
Schedule: {stats["lr_schedule"]}
Model Divergence: {stats["model_divergence"]:.4f}
Avg Pred Error: {stats["avg_prediction_error"]:.4f}
"""
    axes[5].text(
        0.1,
        0.5,
        info_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=axes[5].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle("B Matrix Learning Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_b_matrix_evolution(
    history_snapshots, factor_idx=0, action_idx=0, figsize=(12, 6)
):
    """
    Plot how a specific B matrix element evolved over time.

    Args:
        history_snapshots: List of (step, B_matrix) tuples
        factor_idx: Which factor to visualize
        action_idx: Which action to visualize
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure
    """
    if not history_snapshots:
        print("No history data available")
        return None

    steps = [s[0] for s in history_snapshots]
    matrices = [s[1][factor_idx][:, :, action_idx] for s in history_snapshots]

    n_next, n_curr = matrices[0].shape

    fig, axes = plt.subplots(n_next, n_curr, figsize=figsize, sharex=True, sharey=True)

    factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]

    for i in range(n_next):
        for j in range(n_curr):
            ax = axes[i, j] if n_next > 1 and n_curr > 1 else axes
            values = [m[i, j] for m in matrices]
            ax.plot(steps, values, "b-", linewidth=1.5)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i == n_next - 1:
                ax.set_xlabel("Step")
            if j == 0:
                ax.set_ylabel(f"P(next={i})")
            ax.set_title(f"curr={j}", fontsize=9)

    fig.suptitle(
        f"B Matrix Evolution: {factor_names[factor_idx]}, Action {action_idx}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


class BMatrixRecorder:
    """
    Records B matrix snapshots during simulation for later visualization.

    Usage:
        recorder = BMatrixRecorder(agent, interval=100)
        for step in range(n_steps):
            agent.step(...)
            recorder.record(step)

        # After simulation
        fig = recorder.plot_evolution(factor_idx=0)
    """

    def __init__(self, agent, interval=100):
        """
        Args:
            agent: AdaptiveActiveInferenceAgent instance
            interval: Record every N steps
        """
        self.agent = agent
        self.interval = interval
        self.history = []

    def record(self, step):
        """Record current B matrix state."""
        if step % self.interval == 0:
            B_copy = [b.copy() for b in self.agent.B]
            self.history.append((step, B_copy))

    def get_history(self):
        """Get recorded history."""
        return self.history

    def plot_evolution(self, factor_idx=0, action_idx=0, figsize=(12, 6)):
        """Plot evolution of specific B matrix elements."""
        return plot_b_matrix_evolution(self.history, factor_idx, action_idx, figsize)

    def create_animation(
        self, factor_idx=0, action_idx=0, interval_ms=200, figsize=(8, 6)
    ):
        """
        Create an animation of B matrix evolution.

        Args:
            factor_idx: Which factor to animate
            action_idx: Which action to animate
            interval_ms: Milliseconds between frames
            figsize: Figure size

        Returns:
            matplotlib.animation.FuncAnimation
        """
        if not self.history:
            print("No history data available")
            return None

        factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]
        state_names = [
            ["Low", "Optimal", "High"],
            ["Safe", "Overheating"],
            ["Low", "High"],
            ["Reliable", "Spoofed"],
            ["Reliable", "Spoofed"],
        ]

        fig, ax = plt.subplots(figsize=figsize)

        initial_B = self.history[0][1][factor_idx][:, :, action_idx]
        states = state_names[factor_idx]

        # Initialize heatmap
        im = ax.imshow(initial_B, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Probability")

        ax.set_xticks(range(len(states)))
        ax.set_yticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.set_xlabel("Current State")
        ax.set_ylabel("Next State")

        title = ax.set_title(f"{factor_names[factor_idx]} - Step 0")

        # Add text annotations
        texts = []
        for i in range(initial_B.shape[0]):
            row = []
            for j in range(initial_B.shape[1]):
                text = ax.text(
                    j,
                    i,
                    f"{initial_B[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )
                row.append(text)
            texts.append(row)

        def update(frame):
            step, B_list = self.history[frame]
            B = B_list[factor_idx][:, :, action_idx]
            im.set_array(B)
            title.set_text(f"{factor_names[factor_idx]} - Step {step}")

            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    texts[i][j].set_text(f"{B[i, j]:.2f}")
                    # Change color for visibility
                    texts[i][j].set_color("white" if B[i, j] > 0.5 else "black")

            return [im, title] + [t for row in texts for t in row]

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.history), interval=interval_ms, blit=True
        )

        return anim

    def save_animation(self, filepath, factor_idx=0, action_idx=0, fps=5):
        """
        Save animation to file (requires ffmpeg for mp4, pillow for gif).

        Args:
            filepath: Output path (e.g., 'animation.gif' or 'animation.mp4')
            factor_idx: Which factor to animate
            action_idx: Which action to animate
            fps: Frames per second
        """
        anim = self.create_animation(factor_idx, action_idx, interval_ms=1000 // fps)
        if anim is None:
            return

        if filepath.endswith(".gif"):
            anim.save(filepath, writer="pillow", fps=fps)
        else:
            anim.save(filepath, writer="ffmpeg", fps=fps)

        print(f"Animation saved to {filepath}")
        plt.close()


def generate_thesis_b_matrix_figures(agent, output_dir=".", prefix="b_matrix"):
    """
    Generate publication-quality B matrix figures for thesis.

    Args:
        agent: AdaptiveActiveInferenceAgent instance (after training)
        output_dir: Directory to save figures
        prefix: Filename prefix

    Returns:
        list: Paths to generated figures
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    paths = []

    # 1. Summary figure
    fig = plot_all_factors_summary(agent, figsize=(14, 10))
    path = os.path.join(output_dir, f"{prefix}_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 2. Detailed Temperature factor (most important for thesis)
    fig = plot_b_matrix_heatmap(agent, factor_idx=0, figsize=(14, 8))
    path = os.path.join(output_dir, f"{prefix}_temperature.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # 3. Health factors (important for cybersecurity aspect)
    for f_idx, name in [(3, "temp_health"), (4, "motor_health")]:
        fig = plot_b_matrix_heatmap(agent, factor_idx=f_idx, figsize=(14, 8))
        path = os.path.join(output_dir, f"{prefix}_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    print(f"Generated {len(paths)} B matrix figures in {output_dir}")
    return paths


if __name__ == "__main__":
    from active_inference_agent import AdaptiveActiveInferenceAgent

    print("Creating agent and running a visualization test...")
    agent = AdaptiveActiveInferenceAgent(learning_rate=0.05, lr_schedule="constant")

    # Simulate some learning
    recorder = BMatrixRecorder(agent, interval=50)

    for step in range(500):
        temp = 50 + np.random.randn() * 10
        motor = 60 + np.random.randn() * 10
        load = 30 + np.random.randn() * 5
        attack = np.random.random() < 0.05

        agent.step(temp, motor, load, attack)
        recorder.record(step)

    print(f"Recorded {len(recorder.history)} snapshots")

    fig1 = plot_all_factors_summary(agent)
    fig1.savefig("test_b_matrix_summary.png", dpi=100)
    plt.close(fig1)

    fig2 = plot_b_matrix_heatmap(agent, factor_idx=0)
    fig2.savefig("test_b_matrix_temperature.png", dpi=100)
    plt.close(fig2)

    print("Visualization test completed.")
