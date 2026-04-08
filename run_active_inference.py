import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from pytorch_simulation.simulation import Environment, Sensor, Actuator
from pytorch_simulation.active_inference_agent import ActiveInferenceAgent

def run_active_inference_simulation(n_steps=500):
    print("Starting Active Inference Simulation...")

    # Cyber defense is disabled in this standalone demonstration so that
    # the agent relies only on its own verification policy.
    env = Environment(initial_budget=1000.0, beta=3.0, cyber_defense_active=False)

    sensors = {
        'temperature': Sensor('temperature', noise_std_dev=0.5), 
        'motor_temperature': Sensor('motor_temperature', noise_std_dev=0.5), 
        'load': Sensor('load', noise_std_dev=0.2) 
    }
    actuators = {
        'temperature': Actuator('temp_actuator'),
        'load': Actuator('load_actuator')
    }

    agent = ActiveInferenceAgent()

    log_data = []

    for step in range(n_steps):
        true_temp = env.temperature
        true_motor = env.motor_temperature
        true_load = env.load

        # Single attack injection used to visualize the agent response.
        if step == 200:
            print(f"Step {step}: Simulating Attack on Motor Sensor!")
            sensors['motor_temperature'].introduce_anomaly('bias', value=-30.0, duration=50)

        # Observation
        sensed_temp = sensors['temperature'].read(true_temp).item()
        sensed_motor = sensors['motor_temperature'].read(true_motor).item()
        sensed_load = sensors['load'].read(true_load).item()

        # During verification the simulator exposes the true sensor state.
        attack_detected = False
        if env.is_verifying_sensor:
            if sensors['motor_temperature'].anomaly:
                attack_detected = True

        obs_indices = agent.discretize_observation(sensed_temp, sensed_motor, sensed_load, attack_detected)
        qs = agent.infer_state(obs_indices)
        actions = agent.plan_action()
        temp_cmd, load_cmd, verify_cmd = agent.map_action_to_control(actions)

        temp_action = actuators['temperature'].apply(torch.tensor([temp_cmd]))
        load_action = actuators['load'].apply(torch.tensor([load_cmd]))

        if verify_cmd:
            env.trigger_verification('motor_temperature')

        cost = env.step(temp_action, load_action)
        efficiency = env.calculate_performance()

        revenue = efficiency * 0.8
        env.budget += revenue

        log_data.append({
            'step': step,
            'temp': true_temp.item(),
            'motor_temp': true_motor.item(),
            'load': true_load.item(),
            'budget': env.budget,
            'efficiency': efficiency,
            'belief_overheat': qs[1][1], # Belief in Overheating
            'action_cool': 1 if actions[0] == 0 else 0,
            'action_verify': 1 if verify_cmd else 0,
            'attack_active': 1 if sensors['motor_temperature'].anomaly else 0
        })

    df = pd.DataFrame(log_data)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(df['step'], df['temp'], label='Temp')
    axes[0].plot(df['step'], df['motor_temp'], label='Motor Temp')
    axes[0].axhline(80, color='r', linestyle='--', label='Overheat Thresh')
    axes[0].set_ylabel('Temperature')
    axes[0].legend()
    axes[0].set_title('System Temperatures')
    
    axes[1].plot(df['step'], df['load'], label='Load', color='orange')
    axes[1].plot(df['step'], df['efficiency']*100, label='Efficiency %', color='green')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    
    axes[2].plot(df['step'], df['belief_overheat'], label='Belief(Overheat)', color='purple')
    axes[2].plot(df['step'], df['action_verify'], label='Action(Verify)', color='blue', alpha=0.5)
    axes[2].fill_between(df['step'], 0, 1, where=df['attack_active']==1, color='red', alpha=0.2, label='Attack Active')
    axes[2].set_ylabel('Probability / Binary')
    axes[2].legend()
    axes[2].set_title('Agent Internal State')

    axes[3].plot(df['step'], df['budget'], label='Budget', color='black')
    axes[3].set_ylabel('Money')
    axes[3].set_xlabel('Step')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('active_inference_results.png')
    print("Simulation finished. Results saved to active_inference_results.png")

if __name__ == "__main__":
    run_active_inference_simulation()
