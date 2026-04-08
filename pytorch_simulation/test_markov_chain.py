import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_simulation.active_inference_agent import ActiveInferenceAgent

def test_B_matrix():
    print("Testing B-matrix initialization...")
    agent = ActiveInferenceAgent()
    B = agent.B

    # Define expected dimensions
    # Factor 0: Temp (3 states), Control 0 (3 actions)
    # Factor 1: Motor (2 states), Control 1 (3 actions)
    # Factor 2: Load (2 states), Control 1 (3 actions)
    # Factor 3: Temp sensor health (2 states), Control 2 (3 actions)
    # Factor 4: Motor sensor health (2 states), Control 2 (3 actions)
    expected_shapes = [
        (3, 3, 3),
        (2, 2, 3),
        (2, 2, 3),
        (2, 2, 3),
        (2, 2, 3),
    ]

    for i, (b_fac, expected_shape) in enumerate(zip(B, expected_shapes)):
        print(f"\nChecking Factor {i}...")
        
        # Check Shape
        if b_fac.shape != expected_shape:
            print(f"FAIL: Shape mismatch. Expected {expected_shape}, got {b_fac.shape}")
            return
        else:
            print(f"PASS: Shape is {b_fac.shape}")

        # Check Probability Constraints (Columns must sum to 1)
        # Sum over next_state (axis 0)
        column_sums = np.sum(b_fac, axis=0)
        if not np.allclose(column_sums, 1.0):
            print(f"FAIL: Columns do not sum to 1. Max deviation: {np.max(np.abs(column_sums - 1.0))}")
            print(column_sums)
            return
        else:
            print("PASS: All columns sum to 1.")

    print("\nAll B-matrix tests passed!")

if __name__ == "__main__":
    test_B_matrix()
