import pandas as pd
from pytorch_simulation.simulation import run_batch_simulation

def investigate():
    print("Running investigation on Efficiency Paradox...")
    n_runs = 100
    
    print(f"1. Running {n_runs} simulations WITHOUT Defense...")
    df_off = run_batch_simulation(n_runs=n_runs, cyber_defense_active=False)
    mean_eff_off = df_off['avg_efficiency'].mean()
    print(f"   -> Mean Efficiency (OFF): {mean_eff_off:.4f}")
    
    print(f"2. Running {n_runs} simulations WITH Defense...")
    df_on = run_batch_simulation(n_runs=n_runs, cyber_defense_active=True)
    mean_eff_on = df_on['avg_efficiency'].mean()
    print(f"   -> Mean Efficiency (ON):  {mean_eff_on:.4f}")
    
    delta = mean_eff_on - mean_eff_off
    print(f"\nDelta (ON - OFF): {delta:.4f}")
    
    if delta < 0:
        print("\nCONFIRMED: Efficiency is LOWER with Defense ON.")
        print("Possible reasons:")
        print("- False positives triggering unnecessary reactions?")
        print("- Attacks (without defense) accidentally helping stability?")
    else:
        print("\nRESULT: Efficiency is HIGHER (or equal) with Defense ON.")
        print("The user might have seen variance in small batches.")

if __name__ == "__main__":
    investigate()
