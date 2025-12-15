import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

from pytorch_simulation.simulation import run_batch_simulation

def run_comparison():
    n_runs = 20
    n_steps = 500
    attack_prob = 0.05 # Increased to make attacks significant

    results = []

    scenarios = [
        {'agent': 'static', 'defense': True},
        {'agent': 'static', 'defense': False},
        {'agent': 'intelligent', 'defense': True},
        {'agent': 'intelligent', 'defense': False},
    ]

    print(f"Starting comparison ({n_runs} runs per scenario, {n_steps} steps, Attack Prob {attack_prob})...")

    for sc in scenarios:
        print(f"Running: Agent={sc['agent']}, Defense={sc['defense']}...")
        try:
            df = run_batch_simulation(n_runs=n_runs, n_steps=n_steps, 
                                      attack_prob=attack_prob, 
                                      cyber_defense_active=sc['defense'], 
                                      agent_type=sc['agent'])
            
            avg_eff = df['avg_efficiency'].mean()
            avg_budget = df['final_budget'].mean()
            
            results.append({
                'Agent': sc['agent'],
                'Defense': 'ON' if sc['defense'] else 'OFF',
                'Avg Efficiency': avg_eff,
                'Avg Final Budget': avg_budget
            })
        except Exception as e:
            print(f"Failed: {e}")

    print("\n--- Comparison Results ---")
    res_df = pd.DataFrame(results)
    print(res_df.to_markdown(index=False))

if __name__ == "__main__":
    run_comparison()
