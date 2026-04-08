#!/usr/bin/env python
"""
COMPREHENSIVE TEST SCRIPT FOR ALL NEW FEATURES
================================================

This script tests all 5 new features implemented for the thesis:
A. Save/Load Model (persist learned B matrix)
B. B Matrix Visualization
C. Q-Learning Agent comparison
D. Statistical Analysis (t-tests, p-values)
E. Curriculum Learning

Run with: python test_all_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_simulation.active_inference_agent import (
    ActiveInferenceAgent,
    AdaptiveActiveInferenceAgent,
)
from pytorch_simulation.qlearning_agent import QLearningAgent, DoubleQLearningAgent
from pytorch_simulation.statistical_analysis import (
    independent_ttest,
    one_way_anova,
    analyze_experiment_results,
    print_analysis_report,
    bootstrap_ci,
)
from pytorch_simulation.b_matrix_viz import (
    plot_b_matrix_heatmap,
    plot_all_factors_summary,
    BMatrixRecorder,
    generate_thesis_b_matrix_figures,
)
from pytorch_simulation.curriculum_learning import (
    CurriculumScheduler,
    run_curriculum_simulation,
)
from pytorch_simulation.simulation import run_simulation


def test_feature_a_save_load():
    """Test Feature A: Save/Load Model"""
    print("\n" + "=" * 60)
    print("FEATURE A: Save/Load Model")
    print("=" * 60)

    # Create and train agent
    agent = AdaptiveActiveInferenceAgent(learning_rate=0.05)
    print("Training agent for 300 steps...")

    for step in range(300):
        temp = 50 + np.random.randn() * 15
        motor = 60 + np.random.randn() * 20
        load = 30 + np.random.randn() * 10
        attack = np.random.random() < 0.05
        agent.step(temp, motor, load, attack)

    original_stats = agent.get_learning_stats()
    print(f"Original agent - Updates: {original_stats['total_updates']}")
    print(f"Original agent - Divergence: {original_stats['model_divergence']:.4f}")

    # Save model
    os.makedirs("test_outputs", exist_ok=True)
    save_path = agent.save_model("test_outputs/test_model")

    # Load into new agent
    agent2 = AdaptiveActiveInferenceAgent()
    metadata = agent2.load_model(save_path, continue_learning=True)

    loaded_stats = agent2.get_learning_stats()
    print(f"Loaded agent - Updates: {loaded_stats['total_updates']}")
    print(f"Loaded agent - Divergence: {loaded_stats['model_divergence']:.4f}")

    # Verify B matrices match
    for f in range(5):
        if not np.allclose(agent.B[f], agent2.B[f]):
            print(f"ERROR: B matrix {f} mismatch!")
            return False

    print("SUCCESS: B matrices match after save/load!")

    # Test frozen model
    agent3 = AdaptiveActiveInferenceAgent()
    agent3.load_model(save_path, continue_learning=False)
    print(f"Frozen model loaded - ready for deployment")

    return True


def test_feature_b_visualization():
    """Test Feature B: B Matrix Visualization"""
    print("\n" + "=" * 60)
    print("FEATURE B: B Matrix Visualization")
    print("=" * 60)

    # Create and train agent
    agent = AdaptiveActiveInferenceAgent(learning_rate=0.05)
    recorder = BMatrixRecorder(agent, interval=50)

    print("Training agent for 500 steps with recording...")
    for step in range(500):
        temp = 50 + np.random.randn() * 15
        motor = 60 + np.random.randn() * 20
        load = 30 + np.random.randn() * 10
        attack = np.random.random() < 0.05
        agent.step(temp, motor, load, attack)
        recorder.record(step)

    print(f"Recorded {len(recorder.history)} snapshots")

    # Generate figures
    os.makedirs("test_outputs/b_matrix_viz", exist_ok=True)

    # 1. Summary plot
    fig1 = plot_all_factors_summary(agent)
    fig1.savefig("test_outputs/b_matrix_viz/summary.png", dpi=100)
    plt.close(fig1)
    print("Generated: summary.png")

    # 2. Temperature factor detail
    fig2 = plot_b_matrix_heatmap(agent, factor_idx=0)
    fig2.savefig("test_outputs/b_matrix_viz/temperature.png", dpi=100)
    plt.close(fig2)
    print("Generated: temperature.png")

    # 3. Evolution plot
    fig3 = recorder.plot_evolution(factor_idx=0, action_idx=1)
    if fig3:
        fig3.savefig("test_outputs/b_matrix_viz/evolution.png", dpi=100)
        plt.close(fig3)
        print("Generated: evolution.png")

    # 4. Test thesis figure generation
    paths = generate_thesis_b_matrix_figures(agent, "test_outputs/b_matrix_viz")
    print(f"Generated {len(paths)} thesis figures")

    return True


def test_feature_c_qlearning():
    """Test Feature C: Q-Learning Agent"""
    print("\n" + "=" * 60)
    print("FEATURE C: Q-Learning Agent")
    print("=" * 60)

    # Test basic Q-Learning agent
    agent = QLearningAgent(learning_rate=0.1, epsilon=0.1)

    print("Running Q-Learning agent for 500 steps...")
    for step in range(500):
        temp = 50 + np.random.randn() * 15
        motor = 60 + np.random.randn() * 20
        load = 30 + np.random.randn() * 10
        attack = np.random.random() < 0.05

        temp_cmd, load_cmd, verify_cmd = agent.step(temp, motor, load, attack)

    stats = agent.get_learning_stats()
    print(f"Q-Learning Stats:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  States visited: {stats['n_visited_states']}")
    print(f"  Final epsilon: {stats['epsilon']:.4f}")
    print(f"  Avg TD error: {stats['avg_td_error']:.4f}")

    # Test Double Q-Learning
    agent2 = DoubleQLearningAgent(learning_rate=0.1)
    for step in range(500):
        temp = 50 + np.random.randn() * 15
        motor = 60 + np.random.randn() * 20
        load = 30 + np.random.randn() * 10
        attack = np.random.random() < 0.05
        agent2.step(temp, motor, load, attack)

    print(f"Double Q-Learning - States visited: {len(agent2.Q)}")

    # Test save/load
    agent.save_model("test_outputs/qlearning_model")
    agent3 = QLearningAgent()
    agent3.load_model("test_outputs/qlearning_model")
    print(f"Q-Learning model saved and loaded successfully")

    return True


def test_feature_d_statistics():
    """Test Feature D: Statistical Analysis"""
    print("\n" + "=" * 60)
    print("FEATURE D: Statistical Analysis")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    static_results = np.random.normal(5000, 1000, 20)
    learning_results = np.random.normal(7000, 1200, 20)
    qlearning_results = np.random.normal(4500, 1500, 20)

    # T-test
    print("\n1. Independent T-Test (Static vs Learning):")
    result = independent_ttest(static_results, learning_results)
    print(f"   t = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
    print(
        f"   Cohen's d = {result['cohens_d']:.3f} ({result['effect_size_interpretation']})"
    )
    print(f"   Significant at p<0.05: {result['significant_at_05']}")

    # ANOVA
    print("\n2. One-way ANOVA (3 groups):")
    groups = {
        "Static": static_results,
        "Learning": learning_results,
        "Q-Learning": qlearning_results,
    }
    anova_result = one_way_anova(groups)
    print(
        f"   F = {anova_result['f_statistic']:.3f}, p = {anova_result['p_value']:.4f}"
    )
    print(f"   eta^2 = {anova_result['eta_squared']:.4f}")

    if "pairwise_comparisons" in anova_result:
        print("   Post-hoc comparisons:")
        for comp_name, comp in anova_result["pairwise_comparisons"].items():
            print(
                f"     {comp_name}: p = {comp['p_value']:.4f}, d = {comp['cohens_d']:.3f}"
            )

    # Bootstrap CI
    print("\n3. Bootstrap CI:")
    obs, low, high = bootstrap_ci(learning_results, n_bootstrap=5000)
    print(f"   Learning budget: {obs:.0f} (95% CI: [{low:.0f}, {high:.0f}])")

    # Full analysis
    print("\n4. Full DataFrame Analysis:")
    data = []
    for label, results in [
        ("Static", static_results),
        ("Learning", learning_results),
        ("Q-Learning", qlearning_results),
    ]:
        for i, b in enumerate(results):
            data.append(
                {
                    "config_label": label,
                    "run_id": i,
                    "final_budget": b,
                    "avg_efficiency": np.random.uniform(0.5, 1.0),
                }
            )

    df = pd.DataFrame(data)
    analysis = analyze_experiment_results(df)
    print_analysis_report(analysis, "Test Statistical Report")

    return True


def test_feature_e_curriculum():
    """Test Feature E: Curriculum Learning"""
    print("\n" + "=" * 60)
    print("FEATURE E: Curriculum Learning")
    print("=" * 60)

    # Test curriculum scheduler
    print("\n1. Testing Curriculum Scheduler:")
    scheduler = CurriculumScheduler(
        strategy="step_based", progression_thresholds={"steps_per_stage": 100}
    )

    print(f"   Initial stage: {scheduler.current_stage.name}")
    transitions = 0
    for step in range(500):
        if scheduler.step(step):
            transitions += 1
            print(f"   Step {step}: Advanced to {scheduler.current_stage.name}")

    print(f"   Total transitions: {transitions}")

    # Test curriculum simulation
    print("\n2. Running Curriculum Simulation (quick test):")
    agent = AdaptiveActiveInferenceAgent(learning_rate=0.02)

    df, hist = run_curriculum_simulation(
        agent,
        n_steps=400,  # Quick test
        steps_per_stage=100,
    )

    print(f"   Simulation complete: {len(df)} steps")
    print(f"   Stage transitions: {len(hist)}")
    print(f"   Final budget: {df['budget'].iloc[-1]:.0f}")
    print(f"   Stages visited: {df['curriculum_stage'].unique().tolist()}")

    # Metrics by stage
    print("\n   Efficiency by stage:")
    for stage in df["curriculum_stage"].unique():
        stage_eff = df[df["curriculum_stage"] == stage]["performance"].mean()
        print(f"     {stage}: {stage_eff:.3f}")

    return True


def run_comparative_experiment():
    """Run a quick comparative experiment with all agent types."""
    print("\n" + "=" * 60)
    print("COMPARATIVE EXPERIMENT: All Agent Types")
    print("=" * 60)

    agents = {
        "Static AI": ActiveInferenceAgent(efe_mode="full"),
        "Learning AI": AdaptiveActiveInferenceAgent(learning_rate=0.02),
        "Q-Learning": QLearningAgent(learning_rate=0.1, epsilon=0.1),
    }

    results = []
    n_steps = 500

    for agent_name, agent in agents.items():
        print(f"\nRunning {agent_name}...")

        # Run simulation manually
        from pytorch_simulation.simulation import Environment, Sensor, Actuator
        import torch
        import random

        env = Environment(initial_budget=1000.0, cyber_defense_active=True)
        sensors = {
            "temperature": Sensor("temperature", noise_std_dev=0.5),
            "motor_temperature": Sensor("motor_temperature", noise_std_dev=0.5),
            "load": Sensor("load", noise_std_dev=0.2),
        }
        actuators = {
            "temperature": Actuator("temp_actuator"),
            "load": Actuator("load_actuator"),
        }

        for step in range(n_steps):
            # Random attacks
            if random.random() < 0.02:
                sensor = random.choice(list(sensors.values()))
                sensor.introduce_anomaly("bias", 20.0, 30)

            # Read sensors
            temp_reading = sensors["temperature"].read(env.temperature)
            motor_temp_reading = sensors["motor_temperature"].read(
                env.motor_temperature
            )
            load_reading = sensors["load"].read(env.load)

            # Agent action
            temp_cmd, load_cmd, verify_cmd = agent.step(
                temp_reading.item(),
                motor_temp_reading.item(),
                load_reading.item(),
                False,
            )

            # Apply actions
            temp_action = actuators["temperature"].apply(torch.tensor([temp_cmd]))
            load_action = actuators["load"].apply(torch.tensor([load_cmd]))
            env.step(temp_action, load_action)

            # Update budget
            efficiency = env.calculate_performance()
            env.budget += efficiency * 1.5

        results.append(
            {
                "agent": agent_name,
                "final_budget": env.budget,
                "survival": env.budget > 0,
            }
        )
        print(f"  Final budget: {env.budget:.0f}")

    print("\n" + "-" * 40)
    print("SUMMARY:")
    for r in results:
        status = "SURVIVED" if r["survival"] else "FAILED"
        print(f"  {r['agent']}: {r['final_budget']:.0f} ({status})")

    return results


def main():
    """Run all feature tests."""
    print("=" * 60)
    print("TESTING ALL NEW FEATURES")
    print("=" * 60)

    os.makedirs("test_outputs", exist_ok=True)

    results = {}

    # Feature A: Save/Load
    try:
        results["A_SaveLoad"] = test_feature_a_save_load()
    except Exception as e:
        print(f"ERROR in Feature A: {e}")
        results["A_SaveLoad"] = False

    # Feature B: Visualization
    try:
        results["B_Visualization"] = test_feature_b_visualization()
    except Exception as e:
        print(f"ERROR in Feature B: {e}")
        results["B_Visualization"] = False

    # Feature C: Q-Learning
    try:
        results["C_QLearning"] = test_feature_c_qlearning()
    except Exception as e:
        print(f"ERROR in Feature C: {e}")
        results["C_QLearning"] = False

    # Feature D: Statistics
    try:
        results["D_Statistics"] = test_feature_d_statistics()
    except Exception as e:
        print(f"ERROR in Feature D: {e}")
        results["D_Statistics"] = False

    # Feature E: Curriculum
    try:
        results["E_Curriculum"] = test_feature_e_curriculum()
    except Exception as e:
        print(f"ERROR in Feature E: {e}")
        results["E_Curriculum"] = False

    # Comparative experiment
    try:
        run_comparative_experiment()
        results["Comparative"] = True
    except Exception as e:
        print(f"ERROR in Comparative: {e}")
        results["Comparative"] = False

    # Final summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for feature, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {feature}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 60)

    print("\nTest outputs saved in: test_outputs/")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
