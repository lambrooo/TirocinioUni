"""
Curriculum Learning Module for IIoT Security
==============================================

Implements curriculum learning strategies that gradually increase
attack difficulty as the agent learns.

Curriculum learning is particularly relevant for the thesis because:
1. It mimics real-world scenarios where threats evolve over time
2. It helps the learning agent build robust internal models
3. It can improve sample efficiency and final performance

Curriculum Stages:
1. Easy: Low attack probability, simple attack types
2. Medium: Moderate attacks, some sophisticated types
3. Hard: High attack probability, all attack types
4. Adversarial: Maximum difficulty, coordinated attacks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Definition of a curriculum stage."""

    name: str
    attack_prob: float
    attack_types: List[str]
    attack_duration_range: tuple
    description: str


class CurriculumScheduler:
    """
    Manages curriculum progression during training.

    Supports multiple progression strategies:
    - step_based: Progress after fixed number of steps
    - performance_based: Progress when performance threshold is met
    - loss_based: Progress when prediction error is low enough
    - time_based: Progress after fixed episodes/runs
    """

    # Predefined curriculum stages
    STAGES = {
        "easy": CurriculumStage(
            name="easy",
            attack_prob=0.005,
            attack_types=["bias"],
            attack_duration_range=(5, 20),
            description="Low probability bias attacks only",
        ),
        "medium": CurriculumStage(
            name="medium",
            attack_prob=0.015,
            attack_types=["bias", "outlier"],
            attack_duration_range=(10, 50),
            description="Moderate attacks with outliers",
        ),
        "hard": CurriculumStage(
            name="hard",
            attack_prob=0.025,
            attack_types=["bias", "outlier", "spoofing"],
            attack_duration_range=(20, 80),
            description="High probability with spoofing",
        ),
        "adversarial": CurriculumStage(
            name="adversarial",
            attack_prob=0.04,
            attack_types=["bias", "outlier", "spoofing", "dos"],
            attack_duration_range=(30, 100),
            description="Maximum difficulty with DoS",
        ),
    }

    def __init__(
        self,
        strategy: str = "step_based",
        stages: List[str] = None,
        progression_thresholds: Dict = None,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            strategy: 'step_based', 'performance_based', 'loss_based'
            stages: List of stage names in order (default: all stages)
            progression_thresholds: Dict with thresholds for progression
        """
        self.strategy = strategy
        self.stages = stages or ["easy", "medium", "hard", "adversarial"]
        self.current_stage_idx = 0

        # Default thresholds
        self.thresholds = {
            "steps_per_stage": 1000,
            "min_efficiency": 0.6,
            "max_prediction_error": 0.1,
            "min_survival_rate": 0.8,
        }
        if progression_thresholds:
            self.thresholds.update(progression_thresholds)

        # Tracking
        self.steps_in_stage = 0
        self.stage_history = []
        self.performance_history = []

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        stage_name = self.stages[self.current_stage_idx]
        return self.STAGES[stage_name]

    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete (at final stage)."""
        return self.current_stage_idx >= len(self.stages) - 1

    def should_progress(
        self,
        step: int,
        efficiency: float = None,
        prediction_error: float = None,
        survival_rate: float = None,
    ) -> bool:
        """
        Check if we should progress to next stage.

        Args:
            step: Current step number
            efficiency: Current efficiency (for performance_based)
            prediction_error: Current prediction error (for loss_based)
            survival_rate: Current survival rate

        Returns:
            True if should progress to next stage
        """
        if self.is_complete:
            return False

        if self.strategy == "step_based":
            return self.steps_in_stage >= self.thresholds["steps_per_stage"]

        elif self.strategy == "performance_based":
            if efficiency is None:
                return False
            self.performance_history.append(efficiency)

            # Use rolling average
            if len(self.performance_history) >= 100:
                recent_avg = np.mean(self.performance_history[-100:])
                return recent_avg >= self.thresholds["min_efficiency"]
            return False

        elif self.strategy == "loss_based":
            if prediction_error is None:
                return False
            return prediction_error <= self.thresholds["max_prediction_error"]

        elif self.strategy == "survival_based":
            if survival_rate is None:
                return False
            return survival_rate >= self.thresholds["min_survival_rate"]

        return False

    def step(
        self,
        current_step: int,
        efficiency: float = None,
        prediction_error: float = None,
        survival_rate: float = None,
    ) -> bool:
        """
        Update curriculum state and check for progression.

        Args:
            current_step: Current simulation step
            efficiency: Current efficiency metric
            prediction_error: Current prediction error
            survival_rate: Current survival rate

        Returns:
            True if stage changed
        """
        self.steps_in_stage += 1

        if self.should_progress(
            current_step, efficiency, prediction_error, survival_rate
        ):
            return self.advance_stage()

        return False

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns:
            True if successfully advanced
        """
        if self.is_complete:
            return False

        old_stage = self.current_stage.name
        self.current_stage_idx += 1
        new_stage = self.current_stage.name

        self.stage_history.append(
            {
                "from_stage": old_stage,
                "to_stage": new_stage,
                "steps_in_previous": self.steps_in_stage,
            }
        )

        self.steps_in_stage = 0
        self.performance_history = []

        print(f"[Curriculum] Advanced: {old_stage} -> {new_stage}")
        return True

    def reset(self):
        """Reset curriculum to beginning."""
        self.current_stage_idx = 0
        self.steps_in_stage = 0
        self.stage_history = []
        self.performance_history = []

    def get_attack_config(self) -> Dict:
        """
        Get current attack configuration for simulation.

        Returns:
            Dict with attack parameters
        """
        stage = self.current_stage
        return {
            "attack_prob": stage.attack_prob,
            "attack_types": stage.attack_types,
            "duration_range": stage.attack_duration_range,
            "stage_name": stage.name,
        }

    def get_status(self) -> Dict:
        """Get current curriculum status."""
        return {
            "current_stage": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
            "total_stages": len(self.stages),
            "steps_in_stage": self.steps_in_stage,
            "is_complete": self.is_complete,
            "attack_prob": self.current_stage.attack_prob,
        }


class CurriculumEnvironment:
    """
    Wrapper for simulation environment with curriculum-controlled attacks.

    This class modifies attack generation based on curriculum stage.
    """

    def __init__(self, curriculum: CurriculumScheduler, sensors: Dict):
        """
        Args:
            curriculum: CurriculumScheduler instance
            sensors: Dictionary of Sensor objects
        """
        self.curriculum = curriculum
        self.sensors = sensors

    def maybe_inject_attack(self, step: int) -> tuple:
        """
        Potentially inject an attack based on curriculum.

        Args:
            step: Current simulation step

        Returns:
            Tuple of (is_attack, attack_info)
        """
        config = self.curriculum.get_attack_config()

        if np.random.random() < config["attack_prob"]:
            # Select attack type from allowed types
            attack_type = np.random.choice(config["attack_types"])

            # Select sensor to attack
            sensor_name = np.random.choice(list(self.sensors.keys()))
            sensor = self.sensors[sensor_name]

            # Determine attack value
            if attack_type == "bias":
                value = np.random.uniform(10, 30)
            elif attack_type == "outlier":
                value = np.random.uniform(100, 200)
            elif attack_type == "spoofing":
                value = 40.0  # Fixed spoofed value
            else:  # dos
                value = 0.0

            # Determine duration
            duration = np.random.randint(*config["duration_range"])

            # Inject attack
            sensor.introduce_anomaly(attack_type, value, duration)

            return True, {
                "type": attack_type,
                "sensor": sensor_name,
                "value": value,
                "duration": duration,
                "stage": config["stage_name"],
            }

        return False, None


def run_curriculum_simulation(
    agent,
    n_steps: int = 5000,
    curriculum_strategy: str = "step_based",
    steps_per_stage: int = 1000,
    natural_anomaly_prob: float = 0.005,
    cyber_defense_active: bool = True,
):
    """
    Run simulation with curriculum learning.

    Args:
        agent: Agent instance (Active Inference or Q-Learning)
        n_steps: Total simulation steps
        curriculum_strategy: Curriculum progression strategy
        steps_per_stage: Steps per stage (for step_based)
        natural_anomaly_prob: Probability of natural anomalies
        cyber_defense_active: Whether cyber defense is active

    Returns:
        Tuple of (log_df, curriculum_history)
    """
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from simulation import Environment, Sensor, Actuator
    import torch
    import random

    # Initialize curriculum
    curriculum = CurriculumScheduler(
        strategy=curriculum_strategy,
        progression_thresholds={"steps_per_stage": steps_per_stage},
    )

    # Initialize environment
    env = Environment(
        initial_budget=1000.0,
        cyber_defense_active=cyber_defense_active,
    )

    sensors = {
        "temperature": Sensor("temperature", noise_std_dev=0.5, sampling_interval=5),
        "motor_temperature": Sensor(
            "motor_temperature", noise_std_dev=0.5, sampling_interval=5
        ),
        "load": Sensor("load", noise_std_dev=0.2, sampling_interval=5),
    }

    actuators = {
        "temperature": Actuator("temp_actuator"),
        "load": Actuator("load_actuator"),
    }

    curriculum_env = CurriculumEnvironment(curriculum, sensors)

    log_data = []
    curriculum_log = []

    for step in range(n_steps):
        is_under_attack = 0
        attack_info = None

        # Curriculum-controlled attack injection
        is_attack, attack_info = curriculum_env.maybe_inject_attack(step)
        if is_attack:
            is_under_attack = 1

        # Natural anomalies (independent of curriculum)
        if random.random() < natural_anomaly_prob:
            sensor_name = random.choice(list(sensors.keys()))
            sensors[sensor_name].introduce_anomaly("bias", 5.0, random.randint(1, 5))

        # Attack detection (simplified)
        attack_detected = 0
        if cyber_defense_active:
            for sensor in sensors.values():
                if sensor.anomaly and sensor.anomaly["type"] in ["dos", "spoofing"]:
                    attack_detected = 1
                    sensor.anomaly = None

        # Read sensors
        temp_reading = sensors["temperature"].read(env.temperature)
        motor_temp_reading = sensors["motor_temperature"].read(env.motor_temperature)
        load_reading = sensors["load"].read(env.load)

        # Agent action
        if env.verification_paused == 0:
            t_val = temp_reading.item()
            m_val = motor_temp_reading.item()
            l_val = load_reading.item()

            temp_cmd, load_cmd, verify_cmd = agent.step(
                t_val, m_val, l_val, attack_detected == 1
            )

            if verify_cmd:
                env.trigger_verification("motor_temperature")

            temp_command = torch.tensor([temp_cmd])
            load_command = torch.tensor([load_cmd])
        else:
            temp_command = torch.tensor([0.0])
            load_command = torch.tensor([0.0])

        # Actuators
        temp_action = actuators["temperature"].apply(temp_command)
        load_action = actuators["load"].apply(load_command)

        # Environment step
        env.step(temp_action, load_action)

        # Calculate efficiency
        efficiency = env.calculate_performance()
        revenue = efficiency * 1.5
        env.budget += revenue

        # Get prediction error if available
        pred_error = None
        if hasattr(agent, "recent_errors") and agent.recent_errors:
            pred_error = np.mean(agent.recent_errors[-100:])

        # Update curriculum
        stage_changed = curriculum.step(
            step, efficiency=efficiency, prediction_error=pred_error
        )

        if stage_changed:
            curriculum_log.append({"step": step, **curriculum.get_status()})

        # Logging
        log_entry = {
            "step": step,
            "true_temp": env.temperature.item(),
            "true_motor_temp": env.motor_temperature.item(),
            "true_load": env.load.item(),
            "performance": efficiency,
            "budget": env.budget,
            "is_under_attack": is_under_attack,
            "attack_detected": attack_detected,
            "curriculum_stage": curriculum.current_stage.name,
            "curriculum_attack_prob": curriculum.current_stage.attack_prob,
        }

        if attack_info:
            log_entry["attack_type"] = attack_info["type"]
            log_entry["attack_sensor"] = attack_info["sensor"]

        log_data.append(log_entry)

    df = pd.DataFrame(log_data)
    curriculum_history = (
        pd.DataFrame(curriculum_log) if curriculum_log else pd.DataFrame()
    )

    return df, curriculum_history


def run_curriculum_experiment(
    n_runs: int = 5, n_steps: int = 4000, output_dir: str = "curriculum_experiments"
):
    """
    Run curriculum learning experiment comparing agents.

    Args:
        n_runs: Number of runs per configuration
        n_steps: Steps per run
        output_dir: Output directory
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    from active_inference_agent import (
        AdaptiveActiveInferenceAgent,
        ActiveInferenceAgent,
    )
    from qlearning_agent import QLearningAgent

    configurations = [
        {"name": "Static AI", "agent_class": ActiveInferenceAgent, "kwargs": {}},
        {
            "name": "Learning AI",
            "agent_class": AdaptiveActiveInferenceAgent,
            "kwargs": {"learning_rate": 0.01, "lr_schedule": "decay"},
        },
        {
            "name": "Q-Learning",
            "agent_class": QLearningAgent,
            "kwargs": {"learning_rate": 0.1, "epsilon": 0.1},
        },
    ]

    all_results = []

    for config in configurations:
        print(f"\n{'=' * 50}")
        print(f"Running curriculum experiment: {config['name']}")
        print("=" * 50)

        for run_id in range(n_runs):
            print(f"  Run {run_id + 1}/{n_runs}...", end=" ")

            agent = config["agent_class"](**config["kwargs"])
            df, curriculum_hist = run_curriculum_simulation(
                agent,
                n_steps=n_steps,
                curriculum_strategy="step_based",
                steps_per_stage=n_steps // 4,
            )

            # Compute metrics per stage
            for stage in ["easy", "medium", "hard", "adversarial"]:
                stage_data = df[df["curriculum_stage"] == stage]
                if len(stage_data) > 0:
                    result = {
                        "agent": config["name"],
                        "run_id": run_id,
                        "stage": stage,
                        "avg_efficiency": stage_data["performance"].mean(),
                        "final_budget": stage_data["budget"].iloc[-1]
                        if len(stage_data) > 0
                        else 0,
                        "survival": df["budget"].iloc[-1] > 0,
                        "n_attacks": stage_data["is_under_attack"].sum(),
                    }
                    all_results.append(result)

            print(f"Final budget: {df['budget'].iloc[-1]:.0f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "curriculum_results.csv"), index=False)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Curriculum Learning Experiment Results", fontsize=14, fontweight="bold"
    )

    # 1. Efficiency by stage
    ax1 = axes[0, 0]
    stages = ["easy", "medium", "hard", "adversarial"]
    agents = results_df["agent"].unique()
    x = np.arange(len(stages))
    width = 0.25

    for i, agent in enumerate(agents):
        means = [
            results_df[(results_df["agent"] == agent) & (results_df["stage"] == s)][
                "avg_efficiency"
            ].mean()
            for s in stages
        ]
        ax1.bar(x + i * width, means, width, label=agent)

    ax1.set_xlabel("Curriculum Stage")
    ax1.set_ylabel("Average Efficiency")
    ax1.set_title("Efficiency by Curriculum Stage")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(stages)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. Survival rates
    ax2 = axes[0, 1]
    survival_rates = [
        results_df[results_df["agent"] == a]["survival"].mean() for a in agents
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c"][: len(agents)]
    ax2.bar(agents, survival_rates, color=colors)
    ax2.set_ylabel("Survival Rate")
    ax2.set_title("Overall Survival Rate")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Efficiency progression
    ax3 = axes[1, 0]
    stage_order = {"easy": 0, "medium": 1, "hard": 2, "adversarial": 3}
    for agent in agents:
        agent_data = results_df[results_df["agent"] == agent].copy()
        agent_data["stage_num"] = agent_data["stage"].map(stage_order)
        grouped = agent_data.groupby("stage_num")["avg_efficiency"].mean()
        ax3.plot(
            grouped.index, grouped.values, "o-", label=agent, linewidth=2, markersize=8
        )

    ax3.set_xlabel("Curriculum Stage")
    ax3.set_ylabel("Average Efficiency")
    ax3.set_title("Efficiency Progression Through Curriculum")
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(stages)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Attack survival
    ax4 = axes[1, 1]
    ax4.text(
        0.5,
        0.5,
        "Curriculum Learning Summary\n\n"
        "Key insight: Learning agents should\n"
        "maintain efficiency as difficulty\n"
        "increases, while static agents\n"
        "may struggle in harder stages.",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax4.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_experiment.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}/")

    return results_df


if __name__ == "__main__":
    # Test curriculum scheduler
    print("Testing Curriculum Scheduler...")

    scheduler = CurriculumScheduler(
        strategy="step_based", progression_thresholds={"steps_per_stage": 100}
    )

    print(f"Initial stage: {scheduler.current_stage.name}")
    print(f"Attack prob: {scheduler.current_stage.attack_prob}")

    for step in range(500):
        changed = scheduler.step(step)
        if changed:
            print(f"Step {step}: Advanced to {scheduler.current_stage.name}")

    print(f"\nFinal stage: {scheduler.current_stage.name}")
    print(f"Stage history: {len(scheduler.stage_history)} transitions")

    # Short curriculum simulation for local verification
    print("\n" + "=" * 50)
    print("Running short curriculum simulation...")

    from active_inference_agent import AdaptiveActiveInferenceAgent

    agent = AdaptiveActiveInferenceAgent(learning_rate=0.02)
    df, hist = run_curriculum_simulation(agent, n_steps=1000, steps_per_stage=250)

    print(f"Simulation complete: {len(df)} steps")
    print(f"Stage transitions: {len(hist)}")
    print(f"Final budget: {df['budget'].iloc[-1]:.0f}")
    print(f"Stages visited: {df['curriculum_stage'].unique()}")

    print("\nCurriculum learning test completed.")
