#!/usr/bin/env python
"""
THESIS FIGURE GENERATOR
=======================

Genera tutte le figure per la tesi con stile professionale usando seaborn.
Include anche export delle tabelle in formato LaTeX.

Output:
- thesis_figures/  : Directory con tutte le figure PNG/PDF
- thesis_tables/   : Directory con tabelle LaTeX

Uso:
    python generate_thesis_figures.py --quick     # Test veloce
    python generate_thesis_figures.py --full      # Esperimenti completi
"""

import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import seaborn, fall back to basic matplotlib if not available
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Using basic matplotlib style.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_thesis_style():
    """Configura lo stile per figure da tesi."""
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        sns.set_palette("colorblind")

    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def create_output_dirs():
    """Crea le directory di output."""
    figures_dir = "thesis_figures"
    tables_dir = "thesis_tables"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return figures_dir, tables_dir


def export_latex_table(df, filename, caption, label):
    """Esporta un DataFrame come tabella LaTeX."""
    latex = df.to_latex(
        index=True,
        caption=caption,
        label=label,
        float_format="%.2f",
        bold_rows=True,
        column_format="l" + "r" * len(df.columns),
    )

    with open(filename, "w") as f:
        f.write(latex)

    print(f"  Saved: {filename}")


def resolve_run_seed(base_seed, run_id):
    return None if base_seed is None else int(base_seed + run_id)


def crossover_step(static_series, learning_series, steps):
    gap = learning_series - static_series
    positive_steps = steps[gap > 0]
    if len(positive_steps) == 0:
        return None
    return int(positive_steps.iloc[0])


def run_experiments(n_runs=5, n_steps_short=500, n_steps_long=2000, base_seed=None):
    """Esegue gli esperimenti e raccoglie i dati."""
    # Import here to avoid issues if dependencies not installed
    os.environ["WANDB_MODE"] = "disabled"
    from pytorch_simulation.simulation import run_simulation

    results = {
        "short_term": [],
        "long_term": [],
        "lr_comparison": [],
        "schedule_comparison": [],
        "crossover": {},
    }

    attack_prob = 0.02

    # ===== SHORT-TERM COMPARISON =====
    print("\n[1/4] Running short-term experiments...")
    configs = [
        ("intelligent", "Static AI", None, None),
        ("intelligent_adaptive", "Learning (constant)", 0.01, "constant"),
        ("intelligent_adaptive", "Learning (decay)", 0.01, "decay"),
    ]

    for agent_type, label, lr, schedule in configs:
        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            if agent_type == "intelligent":
                df, fig = run_simulation(
                    n_steps=n_steps_short,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type=agent_type,
                    efe_mode="full",
                    precision=5.0,
                    seed=run_seed,
                )
            else:
                df, fig = run_simulation(
                    n_steps=n_steps_short,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type=agent_type,
                    efe_mode="full",
                    precision=5.0,
                    learning_rate=lr,
                    lr_schedule=schedule,
                    seed=run_seed,
                )
            plt.close(fig)

            results["short_term"].append(
                {
                    "agent": label,
                    "run_id": run_id,
                    "seed": run_seed,
                    "final_budget": df["budget"].iloc[-1],
                    "avg_efficiency": df["performance"].mean(),
                    "second_half_efficiency": df["performance"].iloc[len(df) // 2 :].mean(),
                    "survival": 1 if df["budget"].iloc[-1] > 0 else 0,
                }
            )

    # ===== LONG-TERM COMPARISON =====
    print("[2/4] Running long-term experiments...")
    for agent_type, label, lr, schedule in configs:
        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            if agent_type == "intelligent":
                df, fig = run_simulation(
                    n_steps=n_steps_long,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type=agent_type,
                    efe_mode="full",
                    precision=5.0,
                    seed=run_seed,
                )
            else:
                df, fig = run_simulation(
                    n_steps=n_steps_long,
                    attack_prob=attack_prob,
                    cyber_defense_active=True,
                    agent_type=agent_type,
                    efe_mode="full",
                    precision=5.0,
                    learning_rate=lr,
                    lr_schedule=schedule,
                    seed=run_seed,
                )
            plt.close(fig)

            n = len(df)
            results["long_term"].append(
                {
                    "agent": label,
                    "run_id": run_id,
                    "seed": run_seed,
                    "final_budget": df["budget"].iloc[-1],
                    "avg_efficiency": df["performance"].mean(),
                    "eff_first_half": df["performance"].iloc[: n // 2].mean(),
                    "eff_second_half": df["performance"].iloc[n // 2 :].mean(),
                    "survival": 1 if df["budget"].iloc[-1] > 0 else 0,
                }
            )

            if run_id == 0 and label in ["Static AI", "Learning (decay)"]:
                results["crossover"][label] = df[
                    ["step", "budget", "performance"]
                ].copy()

    # ===== LEARNING RATE COMPARISON =====
    print("[3/4] Running learning rate comparison...")
    for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            df, fig = run_simulation(
                n_steps=n_steps_short,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type="intelligent_adaptive",
                efe_mode="full",
                precision=5.0,
                learning_rate=lr,
                lr_schedule="constant",
                seed=run_seed,
            )
            plt.close(fig)

            results["lr_comparison"].append(
                {
                    "learning_rate": lr,
                    "run_id": run_id,
                    "seed": run_seed,
                    "final_budget": df["budget"].iloc[-1],
                    "avg_efficiency": df["performance"].mean(),
                    "second_half_efficiency": df["performance"].iloc[len(df) // 2 :].mean(),
                    "survival": 1 if df["budget"].iloc[-1] > 0 else 0,
                }
            )

    # ===== SCHEDULE COMPARISON =====
    print("[4/4] Running schedule comparison...")
    for schedule in ["constant", "decay", "adaptive"]:
        for run_id in range(n_runs):
            run_seed = resolve_run_seed(base_seed, run_id)
            df, fig = run_simulation(
                n_steps=n_steps_long,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type="intelligent_adaptive",
                efe_mode="full",
                precision=5.0,
                learning_rate=0.01,
                lr_schedule=schedule,
                seed=run_seed,
            )
            plt.close(fig)

            results["schedule_comparison"].append(
                {
                    "schedule": schedule,
                    "run_id": run_id,
                    "seed": run_seed,
                    "final_budget": df["budget"].iloc[-1],
                    "avg_efficiency": df["performance"].mean(),
                    "second_half_efficiency": df["performance"].iloc[len(df) // 2 :].mean(),
                    "survival": 1 if df["budget"].iloc[-1] > 0 else 0,
                }
            )

    return results


def generate_figures(results, figures_dir, tables_dir):
    """Genera tutte le figure per la tesi."""

    # ===== FIGURA 1: Short-term comparison =====
    print("\nGenerating Figure 1: Short-term comparison...")
    df = pd.DataFrame(results["short_term"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Budget boxplot
    if HAS_SEABORN:
        sns.boxplot(
            data=df,
            x="agent",
            y="final_budget",
            hue="agent",
            ax=axes[0],
            palette="colorblind",
            legend=False,
        )
    else:
        df.boxplot(column="final_budget", by="agent", ax=axes[0])
    axes[0].set_title("Final Budget by Agent Type")
    axes[0].set_xlabel("Agent")
    axes[0].set_ylabel("Final Budget")
    axes[0].tick_params(axis="x", rotation=15)

    # Survival rate bar
    survival = df.groupby("agent")["survival"].mean() * 100
    survival.plot(kind="bar", ax=axes[1], color=["#1f77b4", "#2ca02c", "#d62728"])
    axes[1].set_title("Survival Rate by Agent Type")
    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Short-term Comparison (500 steps)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_short_term_comparison.png"))
    plt.savefig(os.path.join(figures_dir, "fig1_short_term_comparison.pdf"))
    plt.close()

    # Export table
    summary = (
        df.groupby("agent")
        .agg(
            {
                "final_budget": ["mean", "std"],
                "avg_efficiency": ["mean", "std"],
                "survival": "mean",
            }
        )
        .round(2)
    )
    summary.columns = [
        "Budget Mean",
        "Budget Std",
        "Efficiency Mean",
        "Efficiency Std",
        "Survival Rate",
    ]
    export_latex_table(
        summary,
        os.path.join(tables_dir, "tab1_short_term.tex"),
        "Short-term Performance Comparison",
        "tab:short_term",
    )

    # ===== FIGURA 2: Long-term comparison =====
    print("Generating Figure 2: Long-term comparison...")
    df = pd.DataFrame(results["long_term"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Budget comparison
    if HAS_SEABORN:
        sns.boxplot(
            data=df,
            x="agent",
            y="final_budget",
            hue="agent",
            ax=axes[0],
            palette="colorblind",
            legend=False,
        )
    else:
        df.boxplot(column="final_budget", by="agent", ax=axes[0])
    axes[0].set_title("Final Budget by Agent Type")
    axes[0].set_xlabel("Agent")
    axes[0].set_ylabel("Final Budget")
    axes[0].tick_params(axis="x", rotation=15)

    # Efficiency improvement (first half vs second half)
    eff_data = df.groupby("agent")[["eff_first_half", "eff_second_half"]].mean()
    x = np.arange(len(eff_data))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        eff_data["eff_first_half"],
        width,
        label="First Half",
        color="#7f7f7f",
    )
    axes[1].bar(
        x + width / 2,
        eff_data["eff_second_half"],
        width,
        label="Second Half",
        color="#2ca02c",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(eff_data.index, rotation=15)
    axes[1].set_title("Efficiency Improvement Over Time")
    axes[1].set_xlabel("Agent")
    axes[1].set_ylabel("Average Efficiency")
    axes[1].legend()

    plt.suptitle("Long-term Comparison (2000 steps)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_long_term_comparison.png"))
    plt.savefig(os.path.join(figures_dir, "fig2_long_term_comparison.pdf"))
    plt.close()

    # Export table
    summary = (
        df.groupby("agent")
        .agg(
            {
                "final_budget": ["mean", "std"],
                "eff_first_half": "mean",
                "eff_second_half": "mean",
                "survival": "mean",
            }
        )
        .round(2)
    )
    summary.columns = [
        "Budget Mean",
        "Budget Std",
        "Eff 1st Half",
        "Eff 2nd Half",
        "Survival",
    ]
    export_latex_table(
        summary,
        os.path.join(tables_dir, "tab2_long_term.tex"),
        "Long-term Performance Comparison",
        "tab:long_term",
    )

    # ===== FIGURA 3: Learning Rate Impact =====
    print("Generating Figure 3: Learning rate impact...")
    df = pd.DataFrame(results["lr_comparison"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Budget vs LR
    summary = df.groupby("learning_rate").agg({"final_budget": ["mean", "std"]})
    summary.columns = ["mean", "std"]
    axes[0].errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Final Budget vs Learning Rate")
    axes[0].set_xlabel("Learning Rate (log scale)")
    axes[0].set_ylabel("Final Budget")

    # Efficiency vs LR
    summary = df.groupby("learning_rate").agg({"avg_efficiency": ["mean", "std"]})
    summary.columns = ["mean", "std"]
    axes[1].errorbar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        marker="s",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="#2ca02c",
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Average Efficiency vs Learning Rate")
    axes[1].set_xlabel("Learning Rate (log scale)")
    axes[1].set_ylabel("Average Efficiency")

    plt.suptitle("Impact of Learning Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_learning_rate_impact.png"))
    plt.savefig(os.path.join(figures_dir, "fig3_learning_rate_impact.pdf"))
    plt.close()

    # Export table
    summary = (
        df.groupby("learning_rate")
        .agg({"final_budget": ["mean", "std"], "avg_efficiency": ["mean", "std"]})
        .round(2)
    )
    summary.columns = ["Budget Mean", "Budget Std", "Efficiency Mean", "Efficiency Std"]
    export_latex_table(
        summary,
        os.path.join(tables_dir, "tab3_learning_rate.tex"),
        "Impact of Learning Rate",
        "tab:learning_rate",
    )

    # ===== FIGURA 4: Schedule Comparison =====
    print("Generating Figure 4: LR schedule comparison...")
    df = pd.DataFrame(results["schedule_comparison"])

    fig, ax = plt.subplots(figsize=(10, 6))

    if HAS_SEABORN:
        sns.boxplot(
            data=df,
            x="schedule",
            y="final_budget",
            hue="schedule",
            ax=ax,
            palette="colorblind",
            legend=False,
        )
    else:
        df.boxplot(column="final_budget", by="schedule", ax=ax)

    ax.set_title("Final Budget by LR Schedule", fontsize=14, fontweight="bold")
    ax.set_xlabel("Learning Rate Schedule")
    ax.set_ylabel("Final Budget")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_schedule_comparison.png"))
    plt.savefig(os.path.join(figures_dir, "fig4_schedule_comparison.pdf"))
    plt.close()

    # Export table
    summary = (
        df.groupby("schedule")
        .agg({"final_budget": ["mean", "std"], "avg_efficiency": ["mean", "std"]})
        .round(2)
    )
    summary.columns = ["Budget Mean", "Budget Std", "Efficiency Mean", "Efficiency Std"]
    export_latex_table(
        summary,
        os.path.join(tables_dir, "tab4_schedule.tex"),
        "Impact of LR Schedule",
        "tab:schedule",
    )

    # ===== FIGURA 5: Static vs Learning crossover =====
    print("Generating Figure 5: crossover view...")
    static_df = results["crossover"].get("Static AI")
    learning_df = results["crossover"].get("Learning (decay)")

    if static_df is not None and learning_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        budget_gap = learning_df["budget"] - static_df["budget"]
        rolling_static = static_df["performance"].rolling(50, min_periods=1).mean()
        rolling_learning = learning_df["performance"].rolling(50, min_periods=1).mean()
        efficiency_gap = rolling_learning - rolling_static

        budget_cross = crossover_step(
            static_df["budget"], learning_df["budget"], static_df["step"]
        )
        efficiency_cross = crossover_step(
            rolling_static, rolling_learning, static_df["step"]
        )

        axes[0].plot(static_df["step"], budget_gap, color="#2ca02c", linewidth=2)
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.7)
        if budget_cross is not None:
            axes[0].axvline(budget_cross, color="#1f77b4", linestyle=":", alpha=0.8)
        axes[0].set_title("Budget Gap (Learning - Static)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Budget Gap")

        axes[1].plot(static_df["step"], efficiency_gap, color="#9467bd", linewidth=2)
        axes[1].axhline(0, color="red", linestyle="--", alpha=0.7)
        if efficiency_cross is not None:
            axes[1].axvline(
                efficiency_cross, color="#1f77b4", linestyle=":", alpha=0.8
            )
        axes[1].set_title("Efficiency Gap (Rolling, Learning - Static)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Efficiency Gap")

        fig.suptitle("Learning Crossover vs Static Baseline", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig5_crossover_comparison.png"))
        plt.savefig(os.path.join(figures_dir, "fig5_crossover_comparison.pdf"))
        plt.close()

        crossover_summary = pd.DataFrame(
            {
                "Metric": ["Budget Crossover Step", "Efficiency Crossover Step"],
                "Value": [
                    budget_cross if budget_cross is not None else -1,
                    efficiency_cross if efficiency_cross is not None else -1,
                ],
            }
        ).set_index("Metric")
        export_latex_table(
            crossover_summary,
            os.path.join(tables_dir, "tab5_crossover.tex"),
            "Crossover points between static and learning agents",
            "tab:crossover",
        )

    print(f"\nAll figures saved to: {figures_dir}/")
    print(f"All tables saved to: {tables_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures")
    parser.add_argument("--quick", action="store_true", help="Quick test (few runs)")
    parser.add_argument("--full", action="store_true", help="Full experiments")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducible thesis figures",
    )
    args = parser.parse_args()

    if args.quick:
        n_runs = 2
        n_steps_short = 200
        n_steps_long = 500
        print("Quick mode: 2 runs, 200/500 steps")
    elif args.full:
        n_runs = 10
        n_steps_short = 500
        n_steps_long = 2000
        print("Full mode: 10 runs, 500/2000 steps")
    else:
        n_runs = 5
        n_steps_short = 500
        n_steps_long = 1000
        print("Default mode: 5 runs, 500/1000 steps")

    print("=" * 60)
    print("THESIS FIGURE GENERATOR")
    print("=" * 60)

    setup_thesis_style()
    figures_dir, tables_dir = create_output_dirs()

    results = run_experiments(n_runs, n_steps_short, n_steps_long, base_seed=args.seed)
    generate_figures(results, figures_dir, tables_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
