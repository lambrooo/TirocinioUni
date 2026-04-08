#!/usr/bin/env python
"""
THESIS EXPERIMENT: STATIC vs LEARNING AGENT
============================================

Questo script esegue gli esperimenti principali per la tesi:
"Confronto tra setting statico e learning in Active Inference per IIoT Security"

ESPERIMENTI PRINCIPALI:
1. Static vs Learning (short-term): Confronto su simulazioni brevi
2. Static vs Learning (long-term): Confronto su simulazioni lunghe (learning advantage)
3. Learning Rate Comparison: Confronto tra diversi valori di LR
4. LR Schedule Comparison: Confronto tra diversi schedule di LR
5. Learning Curves: Analisi della convergenza del modello

Output: CSV con dati, grafici PNG, e report summary
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_simulation.simulation import run_simulation, run_batch_simulation


def create_output_dir(base_dir="learning_experiments"):
    """Crea directory per i risultati con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def resolve_run_seed(base_seed, run_id):
    return None if base_seed is None else int(base_seed + run_id)


def experiment_1_static_vs_learning_short(
    n_steps=500, n_runs=10, attack_prob=0.02, output_dir=".", base_seed=None
):
    """
    ESPERIMENTO 1: Static vs Learning (Short-term)
    ------------------------------------------------
    Su simulazioni brevi, l'agente statico potrebbe avere un vantaggio
    perche' il modello pre-configurato e' gia' ragionevolmente buono.
    L'agente che apprende ha bisogno di tempo per convergere.
    """
    print("\n" + "=" * 60)
    print("ESPERIMENTO 1: Static vs Learning (Short-term)")
    print(f"n_steps={n_steps}, n_runs={n_runs}")
    print("=" * 60)

    configurations = [
        {"agent": "intelligent", "label": "Static (Fixed Model)", "color": "#3498db"},
        {
            "agent": "intelligent_adaptive",
            "lr": 0.01,
            "schedule": "constant",
            "label": "Learning (LR=0.01)",
            "color": "#2ecc71",
        },
        {
            "agent": "intelligent_adaptive",
            "lr": 0.05,
            "schedule": "constant",
            "label": "Learning (LR=0.05)",
            "color": "#e74c3c",
        },
    ]

    all_results = []
    sample_runs = {}

    for cfg in configurations:
        print(f"\n  Running {n_runs} simulations: {cfg['label']}...")

        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            if cfg["agent"] == "intelligent_adaptive":
                df, _ = run_simulation(
                    n_steps=n_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type="intelligent_adaptive",
                    efe_mode="full",
                    learning_rate=cfg["lr"],
                    lr_schedule=cfg["schedule"],
                    seed=run_seed,
                )
            else:
                df, _ = run_simulation(
                    n_steps=n_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type="intelligent",
                    efe_mode="full",
                    seed=run_seed,
                )
            plt.close("all")

            if run_id == 0:
                sample_runs[cfg["label"]] = df

            result = {
                "experiment": "Short_Term",
                "config_label": cfg["label"],
                "agent_type": cfg["agent"],
                "run_id": run_id,
                "seed": run_seed,
                "n_steps": n_steps,
                "attack_prob": attack_prob,
                "learning_rate": cfg.get("lr", np.nan),
                "lr_schedule": cfg.get("schedule", "fixed"),
                "final_budget": df["budget"].iloc[-1],
                "avg_efficiency": df["performance"].mean(),
                "second_half_efficiency": df["performance"].iloc[len(df) // 2 :].mean(),
                "survival": df["budget"].iloc[-1] > 0,
                "max_motor_temp": df["true_motor_temp"].max(),
                "overheating_steps": (df["true_motor_temp"] > 80).sum(),
            }

            # Add learning metrics if available
            if "learning_rate" in df.columns:
                result["final_lr"] = df["learning_rate"].iloc[-1]
                result["final_divergence"] = df["model_divergence"].iloc[-1]

            all_results.append(result)
        print(f"    Done.")

    results_df = pd.DataFrame(all_results)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Esperimento 1: Static vs Learning (Short-term, {n_steps} steps)",
        fontsize=14,
        fontweight="bold",
    )

    labels = [cfg["label"] for cfg in configurations]
    colors_list = [cfg["color"] for cfg in configurations]

    # 1. Final Budget Comparison
    ax1 = axes[0, 0]
    for i, label in enumerate(labels):
        data = results_df[results_df["config_label"] == label]["final_budget"]
        ax1.boxplot(
            [data],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors_list[i]),
        )
    ax1.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax1.set_ylabel("Budget Finale")
    ax1.set_title("Confronto Budget Finale")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Average Efficiency
    ax2 = axes[0, 1]
    for i, label in enumerate(labels):
        data = results_df[results_df["config_label"] == label]["avg_efficiency"]
        ax2.boxplot(
            [data],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors_list[i]),
        )
    ax2.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax2.set_ylabel("Efficienza Media")
    ax2.set_title("Confronto Efficienza")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Budget Over Time (sample runs)
    ax3 = axes[1, 0]
    for i, cfg in enumerate(configurations):
        label = cfg["label"]
        if label in sample_runs:
            ax3.plot(
                sample_runs[label]["step"],
                sample_runs[label]["budget"],
                label=label,
                color=cfg["color"],
                alpha=0.8,
            )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Budget")
    ax3.set_title("Evoluzione Budget (sample run)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Motor Temperature Over Time
    ax4 = axes[1, 1]
    for i, cfg in enumerate(configurations):
        label = cfg["label"]
        if label in sample_runs:
            ax4.plot(
                sample_runs[label]["step"],
                sample_runs[label]["true_motor_temp"],
                label=label,
                color=cfg["color"],
                alpha=0.8,
            )
    ax4.axhline(y=80, color="red", linestyle="--", label="T_safe", alpha=0.5)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Temperatura Motore (C)")
    ax4.set_title("Dinamica Temperatura (sample run)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_static_vs_learning_short.png"), dpi=150)
    plt.close()

    # Summary statistics
    summary = (
        results_df.groupby("config_label")
        .agg(
            {
                "final_budget": ["mean", "std"],
                "avg_efficiency": ["mean", "std"],
                "overheating_steps": ["mean", "std"],
            }
        )
        .round(2)
    )

    summary.to_csv(os.path.join(output_dir, "exp1_summary.csv"))
    results_df.to_csv(os.path.join(output_dir, "exp1_raw_data.csv"), index=False)

    print("\n  Summary:")
    print(summary)

    return results_df


def experiment_2_static_vs_learning_long(
    n_steps=5000, n_runs=5, attack_prob=0.02, output_dir=".", base_seed=None
):
    """
    ESPERIMENTO 2: Static vs Learning (Long-term)
    ------------------------------------------------
    Su simulazioni lunghe, l'agente che apprende dovrebbe avere un vantaggio
    perche' ha tempo di adattare il suo modello alla vera dinamica.
    """
    print("\n" + "=" * 60)
    print("ESPERIMENTO 2: Static vs Learning (Long-term)")
    print(f"n_steps={n_steps}, n_runs={n_runs}")
    print("=" * 60)

    configurations = [
        {"agent": "intelligent", "label": "Static (Fixed Model)", "color": "#3498db"},
        {
            "agent": "intelligent_adaptive",
            "lr": 0.01,
            "schedule": "constant",
            "label": "Learning (constant)",
            "color": "#2ecc71",
        },
        {
            "agent": "intelligent_adaptive",
            "lr": 0.01,
            "schedule": "decay",
            "label": "Learning (decay)",
            "color": "#e74c3c",
        },
    ]

    all_results = []
    sample_runs = {}

    for cfg in configurations:
        print(f"\n  Running {n_runs} simulations: {cfg['label']}...")

        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            if cfg["agent"] == "intelligent_adaptive":
                df, _ = run_simulation(
                    n_steps=n_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type="intelligent_adaptive",
                    efe_mode="full",
                    learning_rate=cfg["lr"],
                    lr_schedule=cfg["schedule"],
                    seed=run_seed,
                )
            else:
                df, _ = run_simulation(
                    n_steps=n_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type="intelligent",
                    efe_mode="full",
                    seed=run_seed,
                )
            plt.close("all")

            if run_id == 0:
                sample_runs[cfg["label"]] = df

            result = {
                "experiment": "Long_Term",
                "config_label": cfg["label"],
                "agent_type": cfg["agent"],
                "run_id": run_id,
                "seed": run_seed,
                "n_steps": n_steps,
                "attack_prob": attack_prob,
                "learning_rate": cfg.get("lr", np.nan),
                "lr_schedule": cfg.get("schedule", "fixed"),
                "final_budget": df["budget"].iloc[-1],
                "avg_efficiency": df["performance"].mean(),
                "avg_efficiency_first_half": df["performance"]
                .iloc[: n_steps // 2]
                .mean(),
                "avg_efficiency_second_half": df["performance"]
                .iloc[n_steps // 2 :]
                .mean(),
            }

            if "learning_rate" in df.columns:
                result["final_lr"] = df["learning_rate"].iloc[-1]
                result["final_divergence"] = df["model_divergence"].iloc[-1]

            all_results.append(result)
        print(f"    Done.")

    results_df = pd.DataFrame(all_results)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Esperimento 2: Static vs Learning (Long-term, {n_steps} steps)",
        fontsize=14,
        fontweight="bold",
    )

    labels = [cfg["label"] for cfg in configurations]
    colors_list = [cfg["color"] for cfg in configurations]

    # 1. Final Budget Comparison
    ax1 = axes[0, 0]
    for i, label in enumerate(labels):
        data = results_df[results_df["config_label"] == label]["final_budget"]
        ax1.boxplot(
            [data],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors_list[i]),
        )
    ax1.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax1.set_ylabel("Budget Finale")
    ax1.set_title("Confronto Budget Finale")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Efficiency Improvement (First Half vs Second Half)
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(labels))

    first_half = [
        results_df[results_df["config_label"] == l]["avg_efficiency_first_half"].mean()
        for l in labels
    ]
    second_half = [
        results_df[results_df["config_label"] == l]["avg_efficiency_second_half"].mean()
        for l in labels
    ]

    ax2.bar(x - width / 2, first_half, width, label="Prima meta", color="#95a5a6")
    ax2.bar(x + width / 2, second_half, width, label="Seconda meta", color="#27ae60")
    ax2.set_ylabel("Efficienza Media")
    ax2.set_title("Miglioramento nel Tempo")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3. Budget Over Time (sample runs)
    ax3 = axes[1, 0]
    for i, cfg in enumerate(configurations):
        label = cfg["label"]
        if label in sample_runs:
            ax3.plot(
                sample_runs[label]["step"],
                sample_runs[label]["budget"],
                label=label,
                color=cfg["color"],
                alpha=0.8,
            )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Budget")
    ax3.set_title("Evoluzione Budget (sample run)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Performance Over Time (smoothed)
    ax4 = axes[1, 1]
    window = 100
    for i, cfg in enumerate(configurations):
        label = cfg["label"]
        if label in sample_runs:
            perf = (
                sample_runs[label]["performance"]
                .rolling(window=window, min_periods=1)
                .mean()
            )
            ax4.plot(
                sample_runs[label]["step"],
                perf,
                label=label,
                color=cfg["color"],
                alpha=0.8,
            )
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Efficienza (rolling avg)")
    ax4.set_title(f"Performance nel Tempo (window={window})")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_static_vs_learning_long.png"), dpi=150)
    plt.close()

    results_df.to_csv(os.path.join(output_dir, "exp2_raw_data.csv"), index=False)

    # Summary
    summary = (
        results_df.groupby("config_label")
        .agg(
            {
                "final_budget": ["mean", "std"],
                "avg_efficiency_first_half": ["mean"],
                "avg_efficiency_second_half": ["mean"],
            }
        )
        .round(2)
    )

    summary.to_csv(os.path.join(output_dir, "exp2_summary.csv"))

    print("\n  Summary:")
    print(summary)

    return results_df


def experiment_3_learning_rate_comparison(
    n_steps=2000, n_runs=5, attack_prob=0.02, output_dir=".", base_seed=None
):
    """
    ESPERIMENTO 3: Confronto Learning Rate
    ----------------------------------------
    Valuta come diversi valori di learning rate influenzano le performance.
    """
    print("\n" + "=" * 60)
    print("ESPERIMENTO 3: Learning Rate Comparison")
    print(f"n_steps={n_steps}, n_runs={n_runs}")
    print("=" * 60)

    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    all_results = []

    for lr in learning_rates:
        print(f"\n  Running {n_runs} simulations with LR={lr}...")

        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type="intelligent_adaptive",
                efe_mode="full",
                learning_rate=lr,
                lr_schedule="constant",
                seed=run_seed,
            )
            plt.close("all")

            all_results.append(
                {
                    "experiment": "LR_Comparison",
                    "learning_rate": lr,
                    "run_id": run_id,
                    "seed": run_seed,
                    "n_steps": n_steps,
                    "attack_prob": attack_prob,
                    "lr_schedule": "constant",
                    "final_budget": df["budget"].iloc[-1],
                    "avg_efficiency": df["performance"].mean(),
                    "second_half_efficiency": df["performance"].iloc[len(df) // 2 :].mean(),
                    "survival": df["budget"].iloc[-1] > 0,
                    "overheating_steps": (df["true_motor_temp"] > 80).sum(),
                }
            )
        print(f"    Done.")

    results_df = pd.DataFrame(all_results)

    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Esperimento 3: Impatto del Learning Rate", fontsize=14, fontweight="bold"
    )

    # Group by learning_rate
    summary = results_df.groupby("learning_rate").agg(
        {
            "final_budget": ["mean", "std"],
            "avg_efficiency": ["mean", "std"],
            "overheating_steps": ["mean", "std"],
        }
    )

    # 1. Budget vs Learning Rate
    ax1 = axes[0]
    means = summary["final_budget"]["mean"]
    stds = summary["final_budget"]["std"]
    ax1.errorbar(
        learning_rates, means, yerr=stds, marker="o", capsize=5, color="#3498db"
    )
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Budget Finale (media +/- std)")
    ax1.set_title("Budget vs Learning Rate")
    ax1.set_xscale("log")
    ax1.grid(alpha=0.3)

    # 2. Efficiency vs Learning Rate
    ax2 = axes[1]
    means = summary["avg_efficiency"]["mean"]
    stds = summary["avg_efficiency"]["std"]
    ax2.errorbar(
        learning_rates, means, yerr=stds, marker="o", capsize=5, color="#2ecc71"
    )
    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("Efficienza Media (media +/- std)")
    ax2.set_title("Efficienza vs Learning Rate")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)

    # 3. Overheating vs Learning Rate
    ax3 = axes[2]
    means = summary["overheating_steps"]["mean"]
    stds = summary["overheating_steps"]["std"]
    ax3.errorbar(
        learning_rates, means, yerr=stds, marker="o", capsize=5, color="#e74c3c"
    )
    ax3.set_xlabel("Learning Rate")
    ax3.set_ylabel("Steps in Overheating (media +/- std)")
    ax3.set_title("Overheating vs Learning Rate")
    ax3.set_xscale("log")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_learning_rate.png"), dpi=150)
    plt.close()

    results_df.to_csv(os.path.join(output_dir, "exp3_raw_data.csv"), index=False)

    print("\n  Summary:")
    print(summary.round(2))

    return results_df


def experiment_4_lr_schedule_comparison(
    n_steps=3000, n_runs=5, attack_prob=0.02, output_dir=".", base_seed=None
):
    """
    ESPERIMENTO 4: Confronto LR Schedule
    --------------------------------------
    Confronta diversi approcci di scheduling del learning rate.
    """
    print("\n" + "=" * 60)
    print("ESPERIMENTO 4: LR Schedule Comparison")
    print(f"n_steps={n_steps}, n_runs={n_runs}")
    print("=" * 60)

    schedules = [
        {"schedule": "constant", "label": "Constant", "color": "#3498db"},
        {"schedule": "decay", "label": "Exponential Decay", "color": "#2ecc71"},
        {"schedule": "adaptive", "label": "Adaptive", "color": "#f39c12"},
    ]

    all_results = []
    sample_runs = {}

    for sched in schedules:
        print(f"\n  Running {n_runs} simulations with schedule: {sched['label']}...")

        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type="intelligent_adaptive",
                efe_mode="full",
                learning_rate=0.01,
                lr_schedule=sched["schedule"],
                seed=run_seed,
            )
            plt.close("all")

            if run_id == 0:
                sample_runs[sched["label"]] = df

            result = {
                "experiment": "Schedule_Comparison",
                "schedule": sched["label"],
                "run_id": run_id,
                "seed": run_seed,
                "n_steps": n_steps,
                "attack_prob": attack_prob,
                "learning_rate": 0.01,
                "lr_schedule": sched["schedule"],
                "final_budget": df["budget"].iloc[-1],
                "avg_efficiency": df["performance"].mean(),
                "avg_efficiency_second_half": df["performance"]
                .iloc[n_steps // 2 :]
                .mean(),
                "survival": df["budget"].iloc[-1] > 0,
            }

            if "learning_rate" in df.columns:
                result["final_lr"] = df["learning_rate"].iloc[-1]

            all_results.append(result)
        print(f"    Done.")

    results_df = pd.DataFrame(all_results)

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Esperimento 4: Confronto LR Schedule", fontsize=14, fontweight="bold")

    labels = [s["label"] for s in schedules]
    colors_list = [s["color"] for s in schedules]

    # 1. Final Budget
    ax1 = axes[0, 0]
    for i, label in enumerate(labels):
        data = results_df[results_df["schedule"] == label]["final_budget"]
        ax1.boxplot(
            [data],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors_list[i]),
        )
    ax1.set_xticklabels(labels, fontsize=8, rotation=15)
    ax1.set_ylabel("Budget Finale")
    ax1.set_title("Confronto Budget Finale")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Efficiency Second Half (to see learning effect)
    ax2 = axes[0, 1]
    for i, label in enumerate(labels):
        data = results_df[results_df["schedule"] == label]["avg_efficiency_second_half"]
        ax2.boxplot(
            [data],
            positions=[i],
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor=colors_list[i]),
        )
    ax2.set_xticklabels(labels, fontsize=8, rotation=15)
    ax2.set_ylabel("Efficienza Media (2a meta)")
    ax2.set_title("Efficienza dopo Convergenza")
    ax2.grid(axis="y", alpha=0.3)

    # 3. Learning Rate Over Time (sample runs)
    ax3 = axes[1, 0]
    for sched in schedules:
        label = sched["label"]
        if label in sample_runs and "learning_rate" in sample_runs[label].columns:
            df = sample_runs[label]
            ax3.plot(
                df["step"],
                df["learning_rate"],
                label=label,
                color=sched["color"],
                alpha=0.8,
            )
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Evoluzione Learning Rate")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Performance Over Time
    ax4 = axes[1, 1]
    window = 100
    for sched in schedules:
        label = sched["label"]
        if label in sample_runs:
            perf = (
                sample_runs[label]["performance"]
                .rolling(window=window, min_periods=1)
                .mean()
            )
            ax4.plot(
                sample_runs[label]["step"],
                perf,
                label=label,
                color=sched["color"],
                alpha=0.8,
            )
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Efficienza (rolling avg)")
    ax4.set_title(f"Performance nel Tempo (window={window})")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp4_lr_schedules.png"), dpi=150)
    plt.close()

    results_df.to_csv(os.path.join(output_dir, "exp4_raw_data.csv"), index=False)

    summary = (
        results_df.groupby("schedule")
        .agg(
            {
                "final_budget": ["mean", "std"],
                "avg_efficiency_second_half": ["mean", "std"],
            }
        )
        .round(2)
    )

    summary.to_csv(os.path.join(output_dir, "exp4_summary.csv"))

    print("\n  Summary:")
    print(summary)

    return results_df


def experiment_5_learning_curves(
    n_steps=5000, n_runs=3, attack_prob=0.02, output_dir=".", base_seed=None
):
    """
    ESPERIMENTO 5: Learning Curves Analysis
    -----------------------------------------
    Analisi dettagliata della convergenza del modello.
    """
    print("\n" + "=" * 60)
    print("ESPERIMENTO 5: Learning Curves Analysis")
    print(f"n_steps={n_steps}, n_runs={n_runs}")
    print("=" * 60)

    print(f"\n  Running {n_runs} long simulations for learning curve analysis...")

    all_learning_data = []

    for run_id in range(n_runs):
        run_seed = resolve_run_seed(base_seed, run_id)
        df, _ = run_simulation(
            n_steps=n_steps,
            attack_prob=attack_prob,
            cyber_defense_active=True,
            agent_type="intelligent_adaptive",
            efe_mode="full",
            learning_rate=0.01,
            lr_schedule="decay",
            seed=run_seed,
        )
        plt.close("all")

        # Collect learning metrics
        if "learning_rate" in df.columns:
            df["run_id"] = run_id
            df["seed"] = run_seed
            all_learning_data.append(
                df[
                    [
                        "step",
                        "run_id",
                        "seed",
                        "learning_rate",
                        "model_divergence",
                        "avg_prediction_error",
                        "performance",
                        "budget",
                    ]
                ]
            )

    if all_learning_data:
        learning_df = pd.concat(all_learning_data, ignore_index=True)

        # Generate learning curves plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Esperimento 5: Learning Curves Analysis", fontsize=14, fontweight="bold"
        )

        # Average across runs
        avg_data = learning_df.groupby("step").mean().reset_index()

        # 1. Learning Rate Decay
        ax1 = axes[0, 0]
        ax1.plot(avg_data["step"], avg_data["learning_rate"], color="#3498db")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Learning Rate")
        ax1.set_title("Learning Rate Decay")
        ax1.grid(alpha=0.3)

        # 2. Model Divergence (how much model changed from initial)
        ax2 = axes[0, 1]
        ax2.plot(avg_data["step"], avg_data["model_divergence"], color="#e74c3c")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Model Divergence (KL)")
        ax2.set_title("Cumulative Model Change")
        ax2.grid(alpha=0.3)

        # 3. Prediction Error
        ax3 = axes[1, 0]
        ax3.plot(
            avg_data["step"],
            avg_data["avg_prediction_error"],
            color="#2ecc71",
            alpha=0.7,
        )
        # Smoothed
        window = 100
        smoothed = (
            avg_data["avg_prediction_error"]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        ax3.plot(
            avg_data["step"], smoothed, color="#27ae60", linewidth=2, label="Smoothed"
        )
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Prediction Error")
        ax3.set_title("Prediction Error (Convergence)")
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Performance Over Time
        ax4 = axes[1, 1]
        perf_smoothed = (
            avg_data["performance"].rolling(window=100, min_periods=1).mean()
        )
        ax4.plot(avg_data["step"], perf_smoothed, color="#9b59b6", linewidth=2)
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Efficienza (smoothed)")
        ax4.set_title("Performance Improvement")
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp5_learning_curves.png"), dpi=150)
        plt.close()

        learning_df.to_csv(
            os.path.join(output_dir, "exp5_learning_data.csv"), index=False
        )

        print(f"    Done. Saved learning data for {n_runs} runs.")
    else:
        print("    Warning: No learning data available.")

    return learning_df if all_learning_data else pd.DataFrame()


def generate_thesis_summary(output_dir):
    """Genera un documento di sintesi per la tesi."""

    summary_text = """
# THESIS EXPERIMENTS: STATIC vs LEARNING AGENT
=============================================

## Obiettivo della Tesi
Confronto tra un agente Active Inference con modello statico (B matrix fissa)
e un agente che apprende le dinamiche di transizione online.

## Esperimenti Eseguiti

### Esperimento 1: Static vs Learning (Short-term)
- Confronta performance su simulazioni brevi (500 steps)
- Ipotesi: L'agente statico potrebbe avere un vantaggio iniziale
- File: exp1_static_vs_learning_short.png

### Esperimento 2: Static vs Learning (Long-term)
- Confronta performance su simulazioni lunghe (5000 steps)
- Ipotesi: L'agente che apprende dovrebbe migliorare nel tempo
- Metriche chiave: miglioramento efficienza tra prima e seconda meta
- File: exp2_static_vs_learning_long.png

### Esperimento 3: Learning Rate Comparison
- Valuta l'impatto di diversi valori di learning rate
- Range testato: 0.001 - 0.1
- File: exp3_learning_rate.png

### Esperimento 4: LR Schedule Comparison
- Confronta diversi approcci di scheduling:
  * Constant: LR fisso
  * Exponential Decay: LR decresce esponenzialmente
  * Linear Warmup: LR parte da 0 e sale gradualmente
  * Cosine Annealing: LR segue curva coseno
  * Adaptive: LR si adatta in base all'errore di predizione
- File: exp4_lr_schedules.png

### Esperimento 5: Learning Curves Analysis
- Analisi dettagliata della convergenza del modello
- Metriche: divergenza del modello, errore di predizione, performance
- File: exp5_learning_curves.png

## Conclusioni Attese

1. **Short-term**: L'agente statico e' competitivo perche' non ha
   il "cold start" del learning.

2. **Long-term**: L'agente che apprende dovrebbe superare lo statico
   una volta che il modello converge.

3. **Learning Rate**: Valori troppo alti causano instabilita',
   valori troppo bassi rallentano l'apprendimento.

4. **LR Schedule**: Schedule con decay dovrebbero bilanciare
   esplorazione iniziale e sfruttamento successivo.
"""

    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description="Thesis experiments: Static vs Learning"
    )
    parser.add_argument("--n_runs", type=int, default=5, help="Runs per configuration")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="learning_experiments",
        help="Output directory",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (fewer runs/steps)"
    )
    parser.add_argument(
        "--exp",
        type=int,
        nargs="+",
        default=None,
        help="Run specific experiments (1-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducible thesis experiments",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"\nOutput directory: {output_dir}\n")

    # Determine which experiments to run
    if args.exp:
        experiments_to_run = args.exp
    else:
        experiments_to_run = [1, 2, 3, 4, 5]

    # Adjust for quick mode
    if args.quick:
        n_runs = 2
        short_steps = 200
        long_steps = 1000
        print("Quick mode: reduced runs and steps")
    else:
        n_runs = args.n_runs
        short_steps = 500
        long_steps = 5000

    print("=" * 60)
    print("THESIS EXPERIMENTS: STATIC vs LEARNING AGENT")
    print("=" * 60)

    if 1 in experiments_to_run:
        experiment_1_static_vs_learning_short(
            n_steps=short_steps,
            n_runs=n_runs,
            output_dir=output_dir,
            base_seed=args.seed,
        )

    if 2 in experiments_to_run:
        experiment_2_static_vs_learning_long(
            n_steps=long_steps,
            n_runs=n_runs,
            output_dir=output_dir,
            base_seed=args.seed,
        )

    if 3 in experiments_to_run:
        experiment_3_learning_rate_comparison(
            n_steps=2000 if not args.quick else 500,
            n_runs=n_runs,
            output_dir=output_dir,
            base_seed=args.seed,
        )

    if 4 in experiments_to_run:
        experiment_4_lr_schedule_comparison(
            n_steps=3000 if not args.quick else 800,
            n_runs=n_runs,
            output_dir=output_dir,
            base_seed=args.seed,
        )

    if 5 in experiments_to_run:
        experiment_5_learning_curves(
            n_steps=long_steps,
            n_runs=min(n_runs, 3),
            output_dir=output_dir,
            base_seed=args.seed,
        )

    # Generate summary
    generate_thesis_summary(output_dir)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)
    print(f"\nResults saved in: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
