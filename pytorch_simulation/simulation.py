
import torch
import pandas as pd
import wandb
import random
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import numpy as np
import sys
import os

# Add current directory to path to allow imports
sys.path.append(os.getcwd())

from pytorch_simulation.active_inference_agent import ActiveInferenceAgent

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Inizializzazione di Weights & Biases
wandb.init(project="pytorch-simulation")

class Environment:
    def __init__(self, initial_temp=20.0, initial_motor_temp=30.0, initial_load=10.0, t_safe=80.0, beta=3.0, # Beta aumentato a 3.0
                 initial_budget=1000.0, 
                 base_heat_coeff=0.5, dissipation_coeff=0.2, cost_per_unit_of_load=0.5,
                 fixed_operational_cost_per_step=1.0, 
                 verification_cost=40.0, verification_downtime=3, verification_sensor_name='motor_temperature', # Tuned costs: 40.0, 3 steps
                 cyber_defense_active=False): 
        self.temperature = torch.tensor([initial_temp], dtype=torch.float32)
        self.motor_temperature = torch.tensor([initial_motor_temp], dtype=torch.float32)
        self.load = torch.tensor([initial_load], dtype=torch.float32)
        self.ambient_temp = 20.0
        self.t_safe = t_safe
        self.beta = beta

        # Nuove variabili per l'economia e l'investimento
        self.budget = initial_budget
        self.base_heat_coeff = base_heat_coeff 
        self.dissipation_coeff = dissipation_coeff 
        self.production_paused = 0 
        self.cost_per_unit_of_load = cost_per_unit_of_load 
        self.fixed_operational_cost_per_step = fixed_operational_cost_per_step 

        # Nuove variabili per l'azione epistemica di verifica
        self.is_verifying_sensor = False
        self.verification_cost = verification_cost
        self.verification_downtime = verification_downtime
        self.verification_sensor_name = verification_sensor_name
        self.verification_paused = 0 

        # Cybersecurity Defense
        self.cyber_defense_active = cyber_defense_active
        self.defense_cost_per_step = 2.0 if cyber_defense_active else 0.0 
        self.attacks_detected = 0

    def step(self, temp_action, load_action):
        cost_of_production_this_step = 0.0 
        
        # Deduci il costo operativo fisso e il costo della difesa
        self.budget -= (self.fixed_operational_cost_per_step + self.defense_cost_per_step)

        # Gestione del downtime per la verifica
        if self.verification_paused > 0: 
            self.load = torch.tensor([0.5], dtype=torch.float32) # Carico ridotto durante la verifica
            self.verification_paused -= 1
            if self.verification_paused == 0:
                self.is_verifying_sensor = False
                # print(f"Downtime verifica terminato per sensore {self.verification_sensor_name}.")

        else:
            # Calcola il load desiderato dall'agente
            desired_load = self.load + load_action 

            # Calcola il load massimo che il budget permette
            max_load_affordable = self.budget / self.cost_per_unit_of_load if self.cost_per_unit_of_load > 0 else float('inf')
            
            # Il load effettivo è il minimo tra il desiderato e l'affordabile
            actual_load = torch.clamp(desired_load, min=0.0, max=max_load_affordable)
            
            # Calcola il costo di produzione per questo step
            cost_of_production_this_step = actual_load.item() * self.cost_per_unit_of_load
            
            # Deduci il costo dal budget
            self.budget -= cost_of_production_this_step
            
            # Aggiorna il load dell'ambiente
            self.load = actual_load

            # La temperatura tende a tornare a quella ambiente
            self.temperature += 0.1 * (self.ambient_temp - self.temperature) + temp_action
            # La temperatura del motore aumenta con il carico e si dissipa
            self.motor_temperature += (self.base_heat_coeff * self.load) - self.dissipation_coeff * (self.motor_temperature - self.temperature)
            # Aggiungiamo un po' di rumore di processo
            self.temperature += torch.randn(1) * 0.1
            self.motor_temperature += torch.randn(1) * 0.1
            self.load += torch.randn(1) * 0.2 

        return cost_of_production_this_step 

    def calculate_performance(self):
        # efficiency = load / (1 + beta * max(0, motor_temperature - T_safe))
        overheat_penalty_term = self.beta * torch.clamp(self.motor_temperature - self.t_safe, min=0)
        denominator = 1 + overheat_penalty_term
        
        if self.load.item() <= 0.1: 
            efficiency = 0.0
        else:
            efficiency = self.load.item() / denominator.item()
        
        return max(0.0, efficiency) 

    def trigger_verification(self, sensor_name):
        if not self.is_verifying_sensor and self.budget >= self.verification_cost:
            self.budget -= self.verification_cost
            self.is_verifying_sensor = True
            self.verification_paused = self.verification_downtime
            self.verification_sensor_name = sensor_name 
            print(f"Verifica sensore '{sensor_name}' avviata! Budget rimanente: {self.budget:.2f}")
            return True
        return False

class Sensor:
    def __init__(self, name, noise_std_dev=0.5, sampling_interval=1): 
        self.name = name
        self.noise_std_dev = noise_std_dev
        self.anomaly = None
        self.trust_score = 1.0  
        self.sampling_interval = sampling_interval 
        self.last_reading = None 
        self.steps_since_last_sample = 0 

    def read(self, true_value):
        self.steps_since_last_sample += 1
        
        # Gestione DoS (Denial of Service)
        if self.anomaly and self.anomaly['type'] == 'dos':
             # Decrementa durata anche per DoS
             self.anomaly['duration'] -= 1
             if self.anomaly['duration'] <= 0:
                 self.anomaly = None
             return self.last_reading if self.last_reading is not None else torch.tensor([0.0])

        if self.steps_since_last_sample % self.sampling_interval == 0:
            reading = true_value + torch.randn(1) * self.noise_std_dev
            
            if self.anomaly:
                if self.anomaly['type'] == 'bias':
                    reading += self.anomaly['value']
                elif self.anomaly['type'] == 'outlier':
                    reading = torch.tensor([self.anomaly['value']], dtype=torch.float32)
                elif self.anomaly['type'] == 'spoofing':
                    reading = torch.tensor([self.anomaly['value']], dtype=torch.float32)
                
                # Decrementa durata dell'anomalia
                self.anomaly['duration'] -= 1
                if self.anomaly['duration'] <= 0:
                    self.anomaly = None 

            self.last_reading = reading
            self.steps_since_last_sample = 0 
        
        return self.last_reading if self.last_reading is not None else true_value 

    def get_verified_reading(self, true_value):
        return true_value

    def introduce_anomaly(self, anomaly_type, value=None, duration=1):
        self.anomaly = {'type': anomaly_type, 'value': value, 'duration': duration}

class Actuator:
    def __init__(self, name):
        self.name = name

    def apply(self, command):
        return command * 0.1

class Agent:
    def __init__(self, temp_setpoint, load_setpoint, p_gain=0.2):
        self.temp_setpoint = torch.tensor([temp_setpoint], dtype=torch.float32)
        self.load_setpoint = torch.tensor([load_setpoint], dtype=torch.float32)
        self.p_gain = p_gain

    def get_commands(self, temp_reading, load_reading):
        temp_error = self.temp_setpoint - temp_reading
        load_error = self.load_setpoint - load_reading
        temp_command = self.p_gain * temp_error
        load_command = self.p_gain * load_error
        return temp_command, load_command

    def step(self, temp_reading, motor_temp_reading, load_reading, attack_detected=False):
        """
        Standard interface for the Agent.
        Returns: temp_cmd, load_cmd, verify_cmd (bool)
        """
        # Static Agent Logic for Verification (Heuristic)
        # We need access to T_safe, but it's not passed here. 
        # For simplicity, we assume T_safe=80.0 or pass it in init.
        # Let's assume standard T_safe=80.0 for the heuristic.
        t_safe = 80.0
        
        # Heuristic: Verify if motor temp is suspiciously high OR spoofing suspected
        condition_overheat = motor_temp_reading > (t_safe + 10)
        condition_spoofing = (load_reading > 20.0) and (motor_temp_reading < 50.0)
        
        verify_cmd = False
        if condition_overheat or condition_spoofing:
            verify_cmd = True
            
        # Control Logic
        temp_cmd, load_cmd = self.get_commands(torch.tensor([temp_reading]), torch.tensor([load_reading]))
        
        return temp_cmd.item(), load_cmd.item(), verify_cmd



def run_simulation(n_steps=1000, attack_prob=0.02, natural_anomaly_prob=0.005, cyber_defense_active=True, agent_type='static', efe_mode='full'):
    # Costi bilanciati: Verifica 40.0, Downtime 3, Beta 3.0
    env = Environment(initial_budget=1000.0, 
                      base_heat_coeff=0.5, dissipation_coeff=0.2, cost_per_unit_of_load=0.5,
                      fixed_operational_cost_per_step=1.0, 
                      verification_cost=40.0, verification_downtime=3, 
                      beta=3.0, 
                      cyber_defense_active=cyber_defense_active) 
    
    sensors = {
        'temperature': Sensor('temperature', noise_std_dev=0.5, sampling_interval=5), 
        'motor_temperature': Sensor('motor_temperature', noise_std_dev=0.5, sampling_interval=5), 
        'load': Sensor('load', noise_std_dev=0.2, sampling_interval=5) 
    }
    actuators = {
        'temperature': Actuator('temp_actuator'),
        'load': Actuator('load_actuator')
    }
    
    # Initialize Agent based on type
    if agent_type == 'intelligent':
        agent = ActiveInferenceAgent(efe_mode=efe_mode)
        print(f"Initialized Intelligent Agent (Active Inference) with EFE mode: {efe_mode}")
    else:
        agent = Agent(temp_setpoint=60.0, load_setpoint=50.0)
        print("Initialized Static Agent")

    log_data = []
    verification_steps = [] 
    detected_attacks = [] 

    for step in range(n_steps):
        is_under_attack = 0
        natural_anomaly_active = 0
        attack_detected_this_step = 0
        cost_of_production_this_step = 0.0 

        # 1. Gestione Anomalie Naturali (Random Faults)
        if random.random() < natural_anomaly_prob:
            natural_anomaly_active = 1
            sensor_to_affect = random.choice(list(sensors.keys()))
            anomaly_type = random.choice(['outlier', 'bias'])
            value = 5.0 if anomaly_type == 'bias' else 50.0 
            duration = random.randint(1, 5) 
            sensors[sensor_to_affect].introduce_anomaly(anomaly_type, value, duration)

        # 2. Gestione Attacchi Informatici
        if random.random() < attack_prob:
            is_under_attack = 1
            sensor_to_affect = random.choice(list(sensors.keys()))
            attack_type = random.choice(['bias', 'outlier', 'spoofing', 'dos'])
            value = 0.0
            if attack_type == 'bias': value = 20.0 
            elif attack_type == 'outlier': value = 150.0 
            elif attack_type == 'spoofing': value = 40.0 
            elif attack_type == 'dos': value = 0.0 
            duration = random.randint(20, 100) 
            sensors[sensor_to_affect].introduce_anomaly(attack_type, value, duration)

        # 3. Cyber Defense Layer (Signature-based Detection)
        if env.cyber_defense_active:
            for name, sensor in sensors.items():
                if sensor.anomaly and sensor.anomaly['type'] == 'dos':
                    attack_detected_this_step = 1
                    env.attacks_detected += 1
                    sensor.anomaly = None
                if sensor.anomaly and sensor.anomaly['type'] == 'spoofing':
                     attack_detected_this_step = 1
                     env.attacks_detected += 1
                     sensor.anomaly = None 

        # Lettura dai sensori
        if env.is_verifying_sensor and env.verification_sensor_name == 'motor_temperature':
            temp_reading = sensors['temperature'].read(env.temperature) 
            motor_temp_reading = sensors['motor_temperature'].get_verified_reading(env.motor_temperature)
            load_reading = sensors['load'].read(env.load)
        else:
            temp_reading = sensors['temperature'].read(env.temperature)
            motor_temp_reading = sensors['motor_temperature'].read(env.motor_temperature)
            load_reading = sensors['load'].read(env.load)

        # Calcolo comandi dell'agente
        if env.verification_paused == 0:
            # Prepare inputs for agent
            # Extract scalar values
            t_val = temp_reading.item()
            m_val = motor_temp_reading.item()
            l_val = load_reading.item()
            
            # Agent Step
            temp_cmd_val, load_cmd_val, verify_cmd = agent.step(t_val, m_val, l_val, attack_detected=(attack_detected_this_step == 1))
            
            # Execute Verification if requested
            if verify_cmd:
                 if env.trigger_verification('motor_temperature'): 
                    verification_steps.append(step)
            
            temp_command = torch.tensor([temp_cmd_val])
            load_command = torch.tensor([load_cmd_val])
            
        else: 
            temp_command, load_command = torch.tensor([0.0]), torch.tensor([0.0])

        # Azioni degli attuatori
        temp_action = actuators['temperature'].apply(temp_command)
        load_action = actuators['load'].apply(load_command) 

        # Aggiornamento dell'ambiente
        cost_of_production_this_step = env.step(temp_action, load_action) 

        # Calcolo performance ed economia
        efficiency = env.calculate_performance()
        revenue = efficiency * 1.5 # Revenue aumentato a 1.5 per permettere profitto se efficiente
        env.budget += revenue 

        # Logging
        log_entry = {
            'step': step,
            'true_temp': env.temperature.item(),
            'sensed_temp': temp_reading.item(),
            'true_motor_temp': env.motor_temperature.item(),
            'sensed_motor_temp': motor_temp_reading.item(),
            'true_load': env.load.item(),
            'sensed_load': load_reading.item(),
            'performance': efficiency, 
            'is_under_attack': is_under_attack,
            'natural_anomaly_active': natural_anomaly_active,
            'attack_detected': attack_detected_this_step, 
            'budget': env.budget,
            'revenue': revenue,
            'cost_of_production': cost_of_production_this_step, 
            'fixed_operational_cost': env.fixed_operational_cost_per_step, 
            'defense_cost': env.defense_cost_per_step, 
            'is_verifying_sensor': env.is_verifying_sensor, 
            'verification_paused': env.verification_paused
        }
        log_data.append(log_entry)
        if os.getenv('WANDB_API_KEY'):
            wandb.log(log_entry)

    # Creazione DataFrame
    df = pd.DataFrame(log_data)

    # Creazione grafici
    fig = plt.figure(figsize=(15, 18)) 
    
    # Grafico Temperatura
    plt.subplot(6, 1, 1) 
    plt.plot(df['step'], df['true_temp'], label='True Temperature')
    plt.plot(df['step'], df['sensed_temp'], label='Sensed Temperature', alpha=0.7)
    plt.title('Temperature')
    plt.legend()
    for step in verification_steps:
        plt.axvline(x=step, color='orange', linestyle=':', label='Verification' if 'Verification' not in plt.gca().get_legend_handles_labels()[1] else "")
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Temperatura Motore
    plt.subplot(6, 1, 2)
    plt.plot(df['step'], df['true_motor_temp'], label='True Motor Temperature')
    plt.plot(df['step'], df['sensed_motor_temp'], label='Sensed Motor Temperature', alpha=0.7)
    plt.axhline(y=env.t_safe, color='r', linestyle='--', label='T_safe')
    plt.title('Motor Temperature')
    plt.legend()
    for step in verification_steps:
        plt.axvline(x=step, color='orange', linestyle=':', label='Verification' if 'Verification' not in plt.gca().get_legend_handles_labels()[1] else "")
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Carico
    plt.subplot(6, 1, 3)
    plt.plot(df['step'], df['true_load'], label='True Load')
    plt.plot(df['step'], df['sensed_load'], label='Sensed Load', alpha=0.7)
    plt.title('Load')
    plt.legend()
    for step in verification_steps:
        plt.axvline(x=step, color='orange', linestyle=':', label='Verification' if 'Verification' not in plt.gca().get_legend_handles_labels()[1] else "")
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Performance (Efficienza)
    plt.subplot(6, 1, 4)
    plt.plot(df['step'], df['performance'], label='System Efficiency', color='green')
    plt.title('System Efficiency')
    plt.legend()
    for step in verification_steps:
        plt.axvline(x=step, color='orange', linestyle=':', label='Verification' if 'Verification' not in plt.gca().get_legend_handles_labels()[1] else "")
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)
    
    # Grafico Budget
    plt.subplot(6, 1, 5)
    plt.plot(df['step'], df['budget'], label='Budget', color='purple')
    plt.title('Budget Over Time')
    plt.legend()
    for step in verification_steps:
        plt.axvline(x=step, color='orange', linestyle=':', label='Verification' if 'Verification' not in plt.gca().get_legend_handles_labels()[1] else "")
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Attacchi Rilevati
    plt.subplot(6, 1, 6)
    plt.plot(df['step'], df['attack_detected'], label='Attack Detected (Defense)', color='blue')
    plt.title('Cyber Defense Interventions')
    plt.legend()
    
    plt.tight_layout()
    
    return df, fig

def run_batch_simulation(n_runs=10, n_steps=1000, attack_prob=0.02, natural_anomaly_prob=0.005, cyber_defense_active=True, agent_type='static', efe_mode='full'):
    """
    Esegue N simulazioni e restituisce statistiche aggregate.
    """
    results = []
    
    for i in range(n_runs):
        # Eseguiamo la simulazione (ignoriamo la figura per ora)
        df, _ = run_simulation(n_steps, attack_prob, natural_anomaly_prob, cyber_defense_active, agent_type, efe_mode)
        
        # Raccogliamo le metriche chiave dell'ultima riga
        final_metrics = {
            'run_id': i,
            'final_budget': df['budget'].iloc[-1],
            'avg_efficiency': df['performance'].mean(),
            'total_attacks': df['is_under_attack'].sum(),
            'attacks_detected': df['attack_detected'].sum(),
            'defense_active': cyber_defense_active
        }
        results.append(final_metrics)
        
    # Creiamo un DataFrame con i risultati di tutti i run
    batch_df = pd.DataFrame(results)
    
    return batch_df


if __name__ == "__main__":
    # Eseguiamo con difesa attiva per default per mostrare il funzionamento
    df, fig = run_simulation(cyber_defense_active=True)
    
    # Salvataggio artefatti se eseguito come script
    df.to_csv("simulation_log.csv", index=False)
    fig.savefig("simulation_charts.png")
    print("Dati salvati in simulation_log.csv e grafici in simulation_charts.png")
    
    if os.getenv('WANDB_API_KEY'):
        wandb.log({"simulation_charts": wandb.Image("simulation_charts.png")})

