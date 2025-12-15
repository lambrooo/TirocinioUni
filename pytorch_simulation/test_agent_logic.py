import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_simulation.active_inference_agent import ActiveInferenceAgent

def test_agent_logic():
    print("Testing Active Inference Agent Logic...")
    agent = ActiveInferenceAgent()

    # --- Test 1: Inference (VFE) ---
    print("\n[Test 1] Inference (VFE Minimization)")
    # Observation: High Temp (2), Overheat (1), High Load (1), Safe (0)
    obs = [2, 1, 1, 0]
    qs = agent.infer_state(obs)
    
    # Check if belief updated correctly (Identity A-matrix should make this sharp)
    # Temp Factor (0) should have high prob for state 2 (High)
    prob_high_temp = qs[0][2]
    print(f"Observation: High Temp. Belief in High Temp: {prob_high_temp:.4f}")
    if prob_high_temp > 0.8:
        print("PASS: Agent correctly inferred High Temperature.")
    else:
        print("FAIL: Agent failed to infer High Temperature.")

    # --- Test 2: Planning (EFE) ---
    print("\n[Test 2] Planning (EFE Minimization)")
    # Force belief to be "High Temp" and "Overheating"
    agent.qs[0] = np.array([0.0, 0.0, 1.0]) # High Temp
    agent.qs[1] = np.array([0.0, 1.0])      # Overheating
    
    # We expect the agent to choose "Cool" (0) and "Decrease Load" (0)
    # to reach preferred states "Optimal Temp" and "Safe Motor".
    actions = agent.plan_action()
    
    print(f"Belief: High Temp, Overheating. Planned Actions: {actions}")
    
    # Check Thermal Control (Action 0)
    if actions[0] == 0: # Cool
        print("PASS: Agent chose to Cool down.")
    else:
        print(f"FAIL: Agent chose action {actions[0]} instead of Cool (0).")

    # Check Load Control (Action 1)
    if actions[1] == 0: # Decrease Load
        print("PASS: Agent chose to Decrease Load.")
    else:
        print(f"FAIL: Agent chose action {actions[1]} instead of Decrease Load (0).")

    # --- Test 3: Helpers ---
    print("\n[Test 3] Helpers (Discretization & Mapping)")
    
    # Discretization
    # Temp=80 (High), Motor=90 (Overheat), Load=10 (Low), Attack=False
    disc_obs = agent.discretize_observation(80.0, 90.0, 10.0, False)
    expected_obs = [2, 1, 0, 0]
    if disc_obs == expected_obs:
        print(f"PASS: Discretization correct. {disc_obs}")
    else:
        print(f"FAIL: Discretization incorrect. Got {disc_obs}, expected {expected_obs}")

    # Mapping
    # Action=[0, 2, 1] -> Cool (-5), Increase (+5), Verify (True)
    cmds = agent.map_action_to_control([0, 2, 1])
    expected_cmds = (-5.0, 5.0, True)
    if cmds == expected_cmds:
        print(f"PASS: Action Mapping correct. {cmds}")
    else:
        print(f"FAIL: Action Mapping incorrect. Got {cmds}, expected {expected_cmds}")

if __name__ == "__main__":
    test_agent_logic()
