#!/usr/bin/env python
"""
THESIS EXPERIMENT SUITE
=======================

Questo script genera tutti gli esperimenti e grafici necessari per la tesi
sul comportamento dell'agente Active Inference.

ESPERIMENTI PRINCIPALI:
1. Confronto EFE modes (full vs epistemic_only vs pragmatic_only)
2. Variazione del peso epistemico vs pragmatico
3. Impatto degli attacchi informatici
4. Confronto agente statico vs intelligente

Output: CSV con dati e grafici PNG per ogni esperimento
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


def create_output_dir(base_dir='thesis_experiments'):
    """Crea directory per i risultati con timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def experiment_1_efe_modes(n_steps=500, n_runs=10, attack_prob=0.02, output_dir='.'):
    """
    ESPERIMENTO 1: Confronto EFE Modes
    -----------------------------------
    Confronta il comportamento dell'agente con diverse modalità EFE:
    - full: entrambi i termini (epistemico + pragmatico)
    - epistemic_only: solo curiosità/esplorazione
    - pragmatic_only: solo raggiungimento obiettivi
    """
    print("\n" + "="*60)
    print("ESPERIMENTO 1: Confronto EFE Modes")
    print("="*60)
    
    efe_modes = ['full', 'epistemic_only', 'pragmatic_only']
    all_results = []
    sample_runs = {}
    
    for mode in efe_modes:
        print(f"\n  Running {n_runs} simulations with EFE mode: {mode}...")
        
        for run_id in range(n_runs):
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type='intelligent',
                efe_mode=mode
            )
            plt.close('all')
            
            # Save first run as sample
            if run_id == 0:
                sample_runs[mode] = df
            
            all_results.append({
                'experiment': 'EFE_Modes',
                'efe_mode': mode,
                'run_id': run_id,
                'final_budget': df['budget'].iloc[-1],
                'avg_efficiency': df['performance'].mean(),
                'max_motor_temp': df['true_motor_temp'].max(),
                'overheating_steps': (df['true_motor_temp'] > 80).sum(),
                'verification_count': df['is_verifying_sensor'].sum(),
                'avg_load': df['true_load'].mean(),
            })
        print(f"    Done.")
    
    results_df = pd.DataFrame(all_results)
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Esperimento 1: Confronto EFE Modes', fontsize=14, fontweight='bold')
    
    colors = {'full': '#2ecc71', 'epistemic_only': '#3498db', 'pragmatic_only': '#e74c3c'}
    
    # 1. Final Budget Comparison
    ax1 = axes[0, 0]
    for mode in efe_modes:
        data = results_df[results_df['efe_mode'] == mode]['final_budget']
        ax1.boxplot([data], positions=[efe_modes.index(mode)], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors[mode]))
    ax1.set_xticklabels(['Full', 'Epistemic\nOnly', 'Pragmatic\nOnly'])
    ax1.set_ylabel('Budget Finale')
    ax1.set_title('Confronto Budget Finale')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Average Efficiency
    ax2 = axes[0, 1]
    for mode in efe_modes:
        data = results_df[results_df['efe_mode'] == mode]['avg_efficiency']
        ax2.boxplot([data], positions=[efe_modes.index(mode)], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors[mode]))
    ax2.set_xticklabels(['Full', 'Epistemic\nOnly', 'Pragmatic\nOnly'])
    ax2.set_ylabel('Efficienza Media')
    ax2.set_title('Confronto Efficienza')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Motor Temperature Over Time (sample runs)
    ax3 = axes[1, 0]
    for mode in efe_modes:
        ax3.plot(sample_runs[mode]['step'], sample_runs[mode]['true_motor_temp'], 
                label=mode, color=colors[mode], alpha=0.8)
    ax3.axhline(y=80, color='red', linestyle='--', label='T_safe', alpha=0.5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Temperatura Motore (°C)')
    ax3.set_title('Dinamica Temperatura Motore')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Budget Over Time (sample runs)
    ax4 = axes[1, 1]
    for mode in efe_modes:
        ax4.plot(sample_runs[mode]['step'], sample_runs[mode]['budget'], 
                label=mode, color=colors[mode], alpha=0.8)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Budget')
    ax4.set_title('Evoluzione Budget')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_efe_modes.png'), dpi=150)
    plt.close()
    
    # Summary statistics
    summary = results_df.groupby('efe_mode').agg({
        'final_budget': ['mean', 'std'],
        'avg_efficiency': ['mean', 'std'],
        'overheating_steps': ['mean', 'std'],
        'avg_load': ['mean', 'std']
    }).round(2)
    
    summary.to_csv(os.path.join(output_dir, 'exp1_summary.csv'))
    results_df.to_csv(os.path.join(output_dir, 'exp1_raw_data.csv'), index=False)
    
    print("\n  Summary:")
    print(summary)
    
    return results_df


def experiment_2_attack_impact(n_steps=500, n_runs=10, output_dir='.'):
    """
    ESPERIMENTO 2: Impatto degli Attacchi
    --------------------------------------
    Varia la probabilità di attacco per vedere come reagisce l'agente.
    """
    print("\n" + "="*60)
    print("ESPERIMENTO 2: Impatto Attacchi Informatici")
    print("="*60)
    
    attack_probs = [0.0, 0.01, 0.02, 0.05, 0.10]
    all_results = []
    
    for attack_prob in attack_probs:
        print(f"\n  Running {n_runs} simulations with attack_prob={attack_prob}...")
        
        for run_id in range(n_runs):
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=True,
                agent_type='intelligent',
                efe_mode='full'
            )
            plt.close('all')
            
            all_results.append({
                'experiment': 'Attack_Impact',
                'attack_prob': attack_prob,
                'run_id': run_id,
                'final_budget': df['budget'].iloc[-1],
                'avg_efficiency': df['performance'].mean(),
                'total_attacks': df['is_under_attack'].sum(),
                'attacks_detected': df['attack_detected'].sum(),
            })
        print(f"    Done.")
    
    results_df = pd.DataFrame(all_results)
    
    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Esperimento 2: Impatto Probabilità Attacchi', fontsize=14, fontweight='bold')
    
    # Group by attack_prob
    summary = results_df.groupby('attack_prob').agg({
        'final_budget': ['mean', 'std'],
        'avg_efficiency': ['mean', 'std'],
        'total_attacks': ['mean', 'std']
    })
    
    # 1. Budget vs Attack Probability
    ax1 = axes[0]
    means = summary['final_budget']['mean']
    stds = summary['final_budget']['std']
    ax1.errorbar(attack_probs, means, yerr=stds, marker='o', capsize=5, color='#3498db')
    ax1.set_xlabel('Probabilità Attacco')
    ax1.set_ylabel('Budget Finale (media ± std)')
    ax1.set_title('Budget vs Probabilità Attacco')
    ax1.grid(alpha=0.3)
    
    # 2. Efficiency vs Attack Probability
    ax2 = axes[1]
    means = summary['avg_efficiency']['mean']
    stds = summary['avg_efficiency']['std']
    ax2.errorbar(attack_probs, means, yerr=stds, marker='o', capsize=5, color='#2ecc71')
    ax2.set_xlabel('Probabilità Attacco')
    ax2.set_ylabel('Efficienza Media (media ± std)')
    ax2.set_title('Efficienza vs Probabilità Attacco')
    ax2.grid(alpha=0.3)
    
    # 3. Total Attacks
    ax3 = axes[2]
    means = summary['total_attacks']['mean']
    stds = summary['total_attacks']['std']
    ax3.errorbar(attack_probs, means, yerr=stds, marker='o', capsize=5, color='#e74c3c')
    ax3.set_xlabel('Probabilità Attacco')
    ax3.set_ylabel('Numero Attacchi (media ± std)')
    ax3.set_title('Attacchi Subiti vs Probabilità')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_attack_impact.png'), dpi=150)
    plt.close()
    
    results_df.to_csv(os.path.join(output_dir, 'exp2_raw_data.csv'), index=False)
    
    print("\n  Summary:")
    print(summary.round(2))
    
    return results_df


def experiment_3_static_vs_intelligent(n_steps=500, n_runs=10, attack_prob=0.02, output_dir='.'):
    """
    ESPERIMENTO 3: Agente Statico vs Intelligente
    ----------------------------------------------
    Confronta le performance dell'agente tradizionale (basato su regole)
    con l'agente Active Inference.
    """
    print("\n" + "="*60)
    print("ESPERIMENTO 3: Agente Statico vs Intelligente")
    print("="*60)
    
    configurations = [
        {'agent': 'static', 'defense': True, 'label': 'Statico + Defense'},
        {'agent': 'static', 'defense': False, 'label': 'Statico'},
        {'agent': 'intelligent', 'defense': True, 'efe_mode': 'full', 'label': 'AI Full + Defense'},
        {'agent': 'intelligent', 'defense': True, 'efe_mode': 'pragmatic_only', 'label': 'AI Pragmatic + Defense'},
    ]
    
    all_results = []
    sample_runs = {}
    
    for cfg in configurations:
        print(f"\n  Running {n_runs} simulations: {cfg['label']}...")
        
        for run_id in range(n_runs):
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=cfg.get('defense', True),
                agent_type=cfg['agent'],
                efe_mode=cfg.get('efe_mode', 'full')
            )
            plt.close('all')
            
            if run_id == 0:
                sample_runs[cfg['label']] = df
            
            all_results.append({
                'experiment': 'Static_vs_Intelligent',
                'config_label': cfg['label'],
                'agent_type': cfg['agent'],
                'run_id': run_id,
                'final_budget': df['budget'].iloc[-1],
                'avg_efficiency': df['performance'].mean(),
                'avg_load': df['true_load'].mean(),
                'max_motor_temp': df['true_motor_temp'].max(),
            })
        print(f"    Done.")
    
    results_df = pd.DataFrame(all_results)
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Esperimento 3: Statico vs Intelligente (Active Inference)', fontsize=14, fontweight='bold')
    
    labels = [cfg['label'] for cfg in configurations]
    colors_list = ['#95a5a6', '#7f8c8d', '#2ecc71', '#e74c3c']
    
    # 1. Final Budget
    ax1 = axes[0, 0]
    for i, label in enumerate(labels):
        data = results_df[results_df['config_label'] == label]['final_budget']
        ax1.boxplot([data], positions=[i], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors_list[i]))
    ax1.set_xticklabels([l.replace(' + ', '\n') for l in labels], fontsize=8)
    ax1.set_ylabel('Budget Finale')
    ax1.set_title('Confronto Budget Finale')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Average Efficiency
    ax2 = axes[0, 1]
    for i, label in enumerate(labels):
        data = results_df[results_df['config_label'] == label]['avg_efficiency']
        ax2.boxplot([data], positions=[i], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors_list[i]))
    ax2.set_xticklabels([l.replace(' + ', '\n') for l in labels], fontsize=8)
    ax2.set_ylabel('Efficienza Media')
    ax2.set_title('Confronto Efficienza')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Load Over Time
    ax3 = axes[1, 0]
    for i, label in enumerate(labels):
        if label in sample_runs:
            ax3.plot(sample_runs[label]['step'], sample_runs[label]['true_load'], 
                    label=label, color=colors_list[i], alpha=0.8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Carico')
    ax3.set_title('Dinamica Carico (sample run)')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Motor Temperature Over Time
    ax4 = axes[1, 1]
    for i, label in enumerate(labels):
        if label in sample_runs:
            ax4.plot(sample_runs[label]['step'], sample_runs[label]['true_motor_temp'], 
                    label=label, color=colors_list[i], alpha=0.8)
    ax4.axhline(y=80, color='red', linestyle='--', label='T_safe', alpha=0.5)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Temperatura Motore (°C)')
    ax4.set_title('Dinamica Temperatura (sample run)')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_static_vs_intelligent.png'), dpi=150)
    plt.close()
    
    results_df.to_csv(os.path.join(output_dir, 'exp3_raw_data.csv'), index=False)
    
    # Summary
    summary = results_df.groupby('config_label').agg({
        'final_budget': ['mean', 'std'],
        'avg_efficiency': ['mean', 'std'],
        'avg_load': ['mean', 'std']
    }).round(2)
    
    summary.to_csv(os.path.join(output_dir, 'exp3_summary.csv'))
    
    print("\n  Summary:")
    print(summary)
    
    return results_df


def experiment_4_defense_impact(n_steps=500, n_runs=10, attack_prob=0.05, output_dir='.'):
    """
    ESPERIMENTO 4: Impatto Cyber Defense
    -------------------------------------
    Confronta performance con e senza cyber defense attiva.
    """
    print("\n" + "="*60)
    print("ESPERIMENTO 4: Impatto Cyber Defense")
    print("="*60)
    
    configurations = [
        {'defense': False, 'efe_mode': 'full', 'label': 'No Defense'},
        {'defense': True, 'efe_mode': 'full', 'label': 'With Defense'},
    ]
    
    all_results = []
    
    for cfg in configurations:
        print(f"\n  Running {n_runs} simulations: {cfg['label']}...")
        
        for run_id in range(n_runs):
            df, _ = run_simulation(
                n_steps=n_steps,
                attack_prob=attack_prob,
                cyber_defense_active=cfg['defense'],
                agent_type='intelligent',
                efe_mode=cfg['efe_mode']
            )
            plt.close('all')
            
            all_results.append({
                'experiment': 'Defense_Impact',
                'defense': cfg['defense'],
                'label': cfg['label'],
                'run_id': run_id,
                'final_budget': df['budget'].iloc[-1],
                'avg_efficiency': df['performance'].mean(),
                'attacks_detected': df['attack_detected'].sum(),
                'total_attacks': df['is_under_attack'].sum(),
            })
        print(f"    Done.")
    
    results_df = pd.DataFrame(all_results)
    
    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Esperimento 4: Impatto Cyber Defense (attack_prob={attack_prob})', fontsize=14, fontweight='bold')
    
    colors = {'No Defense': '#e74c3c', 'With Defense': '#2ecc71'}
    
    # 1. Budget Comparison
    ax1 = axes[0]
    for i, label in enumerate(['No Defense', 'With Defense']):
        data = results_df[results_df['label'] == label]['final_budget']
        ax1.boxplot([data], positions=[i], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors[label]))
    ax1.set_xticklabels(['Senza Defense', 'Con Defense'])
    ax1.set_ylabel('Budget Finale')
    ax1.set_title('Budget Finale')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Efficiency
    ax2 = axes[1]
    for i, label in enumerate(['No Defense', 'With Defense']):
        data = results_df[results_df['label'] == label]['avg_efficiency']
        ax2.boxplot([data], positions=[i], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors[label]))
    ax2.set_xticklabels(['Senza Defense', 'Con Defense'])
    ax2.set_ylabel('Efficienza Media')
    ax2.set_title('Efficienza')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Attacks Detected
    ax3 = axes[2]
    for i, label in enumerate(['No Defense', 'With Defense']):
        data = results_df[results_df['label'] == label]['attacks_detected']
        ax3.boxplot([data], positions=[i], widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=colors[label]))
    ax3.set_xticklabels(['Senza Defense', 'Con Defense'])
    ax3.set_ylabel('Attacchi Rilevati')
    ax3.set_title('Attacchi Rilevati')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_defense_impact.png'), dpi=150)
    plt.close()
    
    results_df.to_csv(os.path.join(output_dir, 'exp4_raw_data.csv'), index=False)
    
    return results_df


def generate_thesis_summary(output_dir):
    """Genera un documento di sintesi per la tesi."""
    
    summary_text = """
# RIEPILOGO ESPERIMENTI TESI

## Esperimenti Eseguiti

### Esperimento 1: Confronto EFE Modes
Confronta il comportamento dell'agente Active Inference con diverse modalità 
di calcolo dell'Expected Free Energy:
- **full**: usa entrambi i termini (epistemico + pragmatico)
- **epistemic_only**: solo minimizzazione dell'entropia (curiosità)
- **pragmatic_only**: solo massimizzazione dell'utilità (obiettivi)

File: exp1_efe_modes.png, exp1_summary.csv

### Esperimento 2: Impatto Probabilità Attacchi
Varia la probabilità di attacchi informatici da 0% a 10% per studiare
la robustezza dell'agente.

File: exp2_attack_impact.png

### Esperimento 3: Agente Statico vs Intelligente
Confronta l'agente tradizionale (basato su regole PID-like) con 
l'agente Active Inference in diverse configurazioni.

File: exp3_static_vs_intelligent.png, exp3_summary.csv

### Esperimento 4: Impatto Cyber Defense
Valuta l'efficacia del layer di cyber defense nel rilevare attacchi
e proteggere il sistema.

File: exp4_defense_impact.png

## Parametri Chiave da Discutere nella Tesi

1. **EFE Mode**: Come il bilanciamento tra epistemico e pragmatico
   influenza le decisioni dell'agente

2. **Attack Probability**: Robustezza del sistema sotto stress

3. **Cyber Defense**: Trade-off tra costo della difesa e protezione

4. **Agent Type**: Vantaggi dell'approccio Active Inference vs regole statiche
"""
    
    with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser(description='Suite di esperimenti per la tesi')
    parser.add_argument('--n_steps', type=int, default=500, help='Steps per simulazione')
    parser.add_argument('--n_runs', type=int, default=10, help='Runs per configurazione')
    parser.add_argument('--output_dir', type=str, default='thesis_experiments', help='Directory output')
    parser.add_argument('--quick', action='store_true', help='Modalità veloce (meno runs)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.n_steps = 200
        args.n_runs = 3
        print("⚡ Quick mode: n_steps=200, n_runs=3")
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"\n📁 Output directory: {output_dir}\n")
    
    # Run all experiments
    print("🧪 Avvio Suite Esperimenti Tesi")
    print("="*60)
    
    experiment_1_efe_modes(args.n_steps, args.n_runs, output_dir=output_dir)
    experiment_2_attack_impact(args.n_steps, args.n_runs, output_dir=output_dir)
    experiment_3_static_vs_intelligent(args.n_steps, args.n_runs, output_dir=output_dir)
    experiment_4_defense_impact(args.n_steps, args.n_runs, output_dir=output_dir)
    
    # Generate summary
    generate_thesis_summary(output_dir)
    
    print("\n" + "="*60)
    print("✅ TUTTI GLI ESPERIMENTI COMPLETATI!")
    print("="*60)
    print(f"\n📁 Risultati salvati in: {output_dir}")
    print("\nFile generati:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
