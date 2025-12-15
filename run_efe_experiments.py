#!/usr/bin/env python
"""
EFE Experiments Script

This script runs comparative simulations to study how agent behavior changes
based on different EFE (Expected Free Energy) calculation modes:
- full: Both epistemic (entropy) and pragmatic (utility) terms
- epistemic_only: Only epistemic term (minimize entropy / maximize information gain)
- pragmatic_only: Only pragmatic term (maximize utility / goal-directed behavior)
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

from pytorch_simulation.simulation import run_simulation


def run_efe_experiments(n_steps=500, n_runs=5, attack_prob=0.02, save_dir='efe_experiments'):
    """
    Run comparative simulations for each EFE mode and collect metrics.
    
    Args:
        n_steps: Number of simulation steps per run
        n_runs: Number of runs per mode for statistical averaging
        attack_prob: Probability of cyber attack per step
        save_dir: Directory to save results
        
    Returns:
        results_df: DataFrame with aggregated results
    """
    efe_modes = ['full', 'epistemic_only', 'pragmatic_only']
    all_results = []
    all_logs = {}
    
    print("="*60)
    print("EFE EXPERIMENTS - Comparing Agent Behavior")
    print("="*60)
    print(f"Configuration: {n_steps} steps, {n_runs} runs per mode, attack_prob={attack_prob}")
    print()
    
    for mode in efe_modes:
        print(f"\n{'='*40}")
        print(f"Running experiments with EFE mode: {mode}")
        print(f"{'='*40}")
        
        mode_logs = []
        
        for run_id in range(n_runs):
            print(f"  Run {run_id + 1}/{n_runs}...", end=" ")
            
            # Run simulation with intelligent agent and specified EFE mode
            df, fig = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                natural_anomaly_prob=0.005,
                cyber_defense_active=True,
                agent_type='intelligent',
                efe_mode=mode
            )
            plt.close(fig)  # Close figure to save memory
            
            # Collect metrics
            metrics = {
                'efe_mode': mode,
                'run_id': run_id,
                'final_budget': df['budget'].iloc[-1],
                'avg_efficiency': df['performance'].mean(),
                'max_efficiency': df['performance'].max(),
                'min_efficiency': df['performance'].min(),
                'total_attacks': df['is_under_attack'].sum(),
                'attacks_detected': df['attack_detected'].sum(),
                'verification_count': df['is_verifying_sensor'].sum(),
                'avg_motor_temp': df['true_motor_temp'].mean(),
                'max_motor_temp': df['true_motor_temp'].max(),
                'overheating_steps': (df['true_motor_temp'] > 80).sum(),
                'avg_load': df['true_load'].mean(),
            }
            
            all_results.append(metrics)
            mode_logs.append(df)
            print(f"Done. Budget: {metrics['final_budget']:.2f}, Verifications: {metrics['verification_count']}")
        
        # Save one representative log per mode
        all_logs[mode] = mode_logs[0]
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    results_path = os.path.join(save_dir, f'efe_results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary = results_df.groupby('efe_mode').agg({
        'final_budget': ['mean', 'std'],
        'avg_efficiency': ['mean', 'std'],
        'verification_count': ['mean', 'std'],
        'overheating_steps': ['mean', 'std'],
        'avg_load': ['mean', 'std'],
    }).round(2)
    
    print(summary)
    
    # Save summary
    summary_path = os.path.join(save_dir, f'efe_summary_{timestamp}.csv')
    summary.to_csv(summary_path)
    
    # Generate comparative plots
    generate_comparison_plots(results_df, all_logs, save_dir, timestamp)
    
    return results_df, all_logs


def generate_comparison_plots(results_df, logs, save_dir, timestamp):
    """Generate comparative visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('EFE Mode Comparison', fontsize=14, fontweight='bold')
    
    modes = ['full', 'epistemic_only', 'pragmatic_only']
    colors = {'full': 'green', 'epistemic_only': 'blue', 'pragmatic_only': 'orange'}
    
    # 1. Final Budget Comparison (Box Plot)
    ax1 = axes[0, 0]
    data_budget = [results_df[results_df['efe_mode'] == m]['final_budget'] for m in modes]
    bp1 = ax1.boxplot(data_budget, labels=modes, patch_artist=True)
    for patch, mode in zip(bp1['boxes'], modes):
        patch.set_facecolor(colors[mode])
    ax1.set_title('Final Budget')
    ax1.set_ylabel('Budget')
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Average Efficiency Comparison
    ax2 = axes[0, 1]
    data_eff = [results_df[results_df['efe_mode'] == m]['avg_efficiency'] for m in modes]
    bp2 = ax2.boxplot(data_eff, labels=modes, patch_artist=True)
    for patch, mode in zip(bp2['boxes'], modes):
        patch.set_facecolor(colors[mode])
    ax2.set_title('Average Efficiency')
    ax2.set_ylabel('Efficiency')
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Verification Count Comparison
    ax3 = axes[0, 2]
    data_verify = [results_df[results_df['efe_mode'] == m]['verification_count'] for m in modes]
    bp3 = ax3.boxplot(data_verify, labels=modes, patch_artist=True)
    for patch, mode in zip(bp3['boxes'], modes):
        patch.set_facecolor(colors[mode])
    ax3.set_title('Verification Count (Epistemic Actions)')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=15)
    
    # 4. Motor Temperature Over Time (Sample Run)
    ax4 = axes[1, 0]
    for mode in modes:
        if mode in logs:
            ax4.plot(logs[mode]['step'], logs[mode]['true_motor_temp'], 
                    label=mode, color=colors[mode], alpha=0.8)
    ax4.axhline(y=80, color='red', linestyle='--', label='T_safe', alpha=0.5)
    ax4.set_title('Motor Temperature (Sample Run)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Temperature')
    ax4.legend(loc='upper right')
    
    # 5. Budget Over Time (Sample Run)
    ax5 = axes[1, 1]
    for mode in modes:
        if mode in logs:
            ax5.plot(logs[mode]['step'], logs[mode]['budget'], 
                    label=mode, color=colors[mode], alpha=0.8)
    ax5.set_title('Budget Over Time (Sample Run)')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Budget')
    ax5.legend(loc='upper right')
    
    # 6. Overheating Steps Comparison
    ax6 = axes[1, 2]
    data_overheat = [results_df[results_df['efe_mode'] == m]['overheating_steps'] for m in modes]
    bp6 = ax6.boxplot(data_overheat, labels=modes, patch_artist=True)
    for patch, mode in zip(bp6['boxes'], modes):
        patch.set_facecolor(colors[mode])
    ax6.set_title('Overheating Steps (Motor > 80°C)')
    ax6.set_ylabel('Count')
    ax6.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'efe_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run EFE mode comparison experiments')
    parser.add_argument('--n_steps', type=int, default=500, help='Steps per simulation')
    parser.add_argument('--n_runs', type=int, default=5, help='Runs per EFE mode')
    parser.add_argument('--attack_prob', type=float, default=0.02, help='Attack probability')
    parser.add_argument('--save_dir', type=str, default='efe_experiments', help='Output directory')
    
    args = parser.parse_args()
    
    results, logs = run_efe_experiments(
        n_steps=args.n_steps,
        n_runs=args.n_runs,
        attack_prob=args.attack_prob,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print("\nKey Observations:")
    print("-" * 40)
    
    # Compute and display key insights
    summary = results.groupby('efe_mode').mean()
    
    # Epistemic behavior indicator (verification count)
    epistemic_verifications = summary.loc['epistemic_only', 'verification_count']
    pragmatic_verifications = summary.loc['pragmatic_only', 'verification_count']
    full_verifications = summary.loc['full', 'verification_count']
    
    print(f"Average Verifications:")
    print(f"  - Epistemic Only: {epistemic_verifications:.1f}")
    print(f"  - Pragmatic Only: {pragmatic_verifications:.1f}")
    print(f"  - Full EFE:       {full_verifications:.1f}")
    
    print(f"\nAverage Final Budget:")
    print(f"  - Epistemic Only: {summary.loc['epistemic_only', 'final_budget']:.1f}")
    print(f"  - Pragmatic Only: {summary.loc['pragmatic_only', 'final_budget']:.1f}")
    print(f"  - Full EFE:       {summary.loc['full', 'final_budget']:.1f}")
    
    print(f"\nAverage Overheating Steps:")
    print(f"  - Epistemic Only: {summary.loc['epistemic_only', 'overheating_steps']:.1f}")
    print(f"  - Pragmatic Only: {summary.loc['pragmatic_only', 'overheating_steps']:.1f}")
    print(f"  - Full EFE:       {summary.loc['full', 'overheating_steps']:.1f}")


if __name__ == "__main__":
    main()
