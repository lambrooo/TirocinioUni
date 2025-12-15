import sys
import traceback
from pytorch_simulation.simulation import run_simulation

print("Starting debug run...")
try:
    df, fig = run_simulation(n_steps=100, agent_type='static')
    print("Simulation successful!")
    print(f"Dataframe shape: {df.shape}")
    print(f"Final Budget: {df['budget'].iloc[-1]}")
except Exception as e:
    print("Simulation FAILED!")
    traceback.print_exc()
