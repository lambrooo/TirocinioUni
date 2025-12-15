import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from pytorch_simulation.simulation import run_simulation

def verify_agents():
    print("--- Testing Static Agent ---")
    try:
        df_static, _ = run_simulation(n_steps=100, agent_type='static')
        print(f"Static Agent Run Successful. Final Budget: {df_static['budget'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Static Agent Failed: {e}")
        return

    print("\n--- Testing Intelligent Agent ---")
    try:
        df_intelligent, _ = run_simulation(n_steps=100, agent_type='intelligent')
        print(f"Intelligent Agent Run Successful. Final Budget: {df_intelligent['budget'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Intelligent Agent Failed: {e}")
        return

    print("\nVerification Complete: Both agents run successfully.")

if __name__ == "__main__":
    verify_agents()
