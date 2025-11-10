
import torch
import pandas as pd
import wandb
import random
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Inizializzazione di Weights & Biases
wandb.init(project="pytorch-simulation")

class Environment:
    def __init__(self, initial_temp=20.0, initial_motor_temp=30.0, initial_load=10.0, t_safe=80.0, beta=2.0):
        self.temperature = torch.tensor([initial_temp], dtype=torch.float32)
        self.motor_temperature = torch.tensor([initial_motor_temp], dtype=torch.float32)
        self.load = torch.tensor([initial_load], dtype=torch.float32)
        self.ambient_temp = 20.0
        self.t_safe = t_safe
        self.beta = beta

    def step(self, temp_action, load_action):
        # La temperatura tende a tornare a quella ambiente
        self.temperature += 0.1 * (self.ambient_temp - self.temperature) + temp_action
        # La temperatura del motore aumenta con il carico e si dissipa
        self.motor_temperature += 0.5 * self.load - 0.2 * (self.motor_temperature - self.temperature)
        # Il carico viene modificato dall'attuatore
        self.load += load_action
        # Aggiungiamo un po' di rumore di processo
        self.temperature += torch.randn(1) * 0.1
        self.motor_temperature += torch.randn(1) * 0.1
        self.load += torch.randn(1) * 0.2

    def calculate_performance(self):
        overheat_penalty = self.beta * torch.clamp(self.motor_temperature - self.t_safe, min=0)
        performance = self.load - overheat_penalty
        return performance.item()

class Sensor:
    def __init__(self, name, noise_std_dev=0.5):
        self.name = name
        self.noise_std_dev = noise_std_dev
        self.anomaly = None
        self.trust_score = 1.0  # Punteggio di fiducia del sensore (per usi futuri)

    def read(self, true_value):
        reading = true_value + torch.randn(1) * self.noise_std_dev
        if self.anomaly:
            if self.anomaly['type'] == 'bias':
                reading += self.anomaly['value']
            elif self.anomaly['type'] == 'outlier':
                reading = torch.tensor([self.anomaly['value']], dtype=torch.float32)
            self.anomaly = None  # L'outlier è istantaneo
        return reading

    def introduce_anomaly(self, anomaly_type, value):
        self.anomaly = {'type': anomaly_type, 'value': value}

class Actuator:
    def __init__(self, name):
        self.name = name

    def apply(self, command):
        # L'azione è una frazione del comando
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

def run_simulation(n_steps=1000, anomaly_interval=100):
    env = Environment()
    sensors = {
        'temperature': Sensor('temperature', noise_std_dev=0.5),
        'motor_temperature': Sensor('motor_temperature', noise_std_dev=0.5),
        'load': Sensor('load', noise_std_dev=0.2)
    }
    actuators = {
        'temperature': Actuator('temp_actuator'),
        'load': Actuator('load_actuator')
    }
    agent = Agent(temp_setpoint=60.0, load_setpoint=50.0)

    log_data = []

    for step in range(n_steps):
        is_under_attack = 0

        # Introduzione periodica di anomalie
        if step > 0 and step % anomaly_interval == 0:
            is_under_attack = 1
            sensor_to_affect = random.choice(list(sensors.keys()))
            anomaly_type = random.choice(['bias', 'outlier'])
            value = 10.0 if anomaly_type == 'bias' else 100.0
            sensors[sensor_to_affect].introduce_anomaly(anomaly_type, value)
            print(f"Step {step}: Attacco '{anomaly_type}' con valore {value} su sensore '{sensor_to_affect}'")

        # Lettura dai sensori
        temp_reading = sensors['temperature'].read(env.temperature)
        motor_temp_reading = sensors['motor_temperature'].read(env.motor_temperature)
        load_reading = sensors['load'].read(env.load)

        # Calcolo comandi dell'agente
        temp_command, load_command = agent.get_commands(temp_reading, load_reading)

        # Azioni degli attuatori
        temp_action = actuators['temperature'].apply(temp_command)
        load_action = actuators['load'].apply(load_command)

        # Aggiornamento dell'ambiente
        env.step(temp_action, load_action)

        # Calcolo della performance
        performance = env.calculate_performance()

        # Logging
        log_entry = {
            'step': step,
            'true_temp': env.temperature.item(),
            'sensed_temp': temp_reading.item(),
            'true_motor_temp': env.motor_temperature.item(),
            'sensed_motor_temp': motor_temp_reading.item(),
            'true_load': env.load.item(),
            'sensed_load': load_reading.item(),
            'temp_command': temp_command.item(),
            'load_command': load_command.item(),
            'performance': performance,
            'is_under_attack': is_under_attack
        }
        log_data.append(log_entry)
        wandb.log(log_entry)

    # Salvataggio in CSV
    df = pd.DataFrame(log_data)
    df.to_csv("simulation_log.csv", index=False)
    print("Dati di simulazione salvati in simulation_log.csv")

    # Creazione grafici
    plt.figure(figsize=(15, 12))
    
    # Grafico Temperatura
    plt.subplot(4, 1, 1)
    plt.plot(df['step'], df['true_temp'], label='True Temperature')
    plt.plot(df['step'], df['sensed_temp'], label='Sensed Temperature', alpha=0.7)
    plt.title('Temperature')
    plt.legend()
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Temperatura Motore
    plt.subplot(4, 1, 2)
    plt.plot(df['step'], df['true_motor_temp'], label='True Motor Temperature')
    plt.plot(df['step'], df['sensed_motor_temp'], label='Sensed Motor Temperature', alpha=0.7)
    plt.axhline(y=env.t_safe, color='r', linestyle='--', label='T_safe')
    plt.title('Motor Temperature')
    plt.legend()
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Carico
    plt.subplot(4, 1, 3)
    plt.plot(df['step'], df['true_load'], label='True Load')
    plt.plot(df['step'], df['sensed_load'], label='Sensed Load', alpha=0.7)
    plt.title('Load')
    plt.legend()
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15)

    # Grafico Performance
    plt.subplot(4, 1, 4)
    plt.plot(df['step'], df['performance'], label='System Performance', color='green')
    plt.title('System Performance')
    plt.legend()
    for i, row in df[df['is_under_attack'] == 1].iterrows():
        plt.axvspan(row['step'], row['step'] + 1, color='red', alpha=0.15, label='Attack Period' if i == df[df['is_under_attack'] == 1].index[0] else "")
    
    plt.tight_layout()
    plt.savefig("simulation_charts.png")
    print("Grafici salvati in simulation_charts.png")
    wandb.log({"simulation_charts": wandb.Image("simulation_charts.png")})


if __name__ == "__main__":
    # Esegui solo se la chiave API di W&B è impostata
    if os.getenv('WANDB_API_KEY'):
        run_simulation()
    else:
        print("La variabile d'ambiente WANDB_API_KEY non è impostata. "
              "Impostala per eseguire la simulazione con Weights & Biases.")
        # Fallback: esegui senza W&B se non è configurato
        # Per fare ciò, dovremmo commentare le chiamate a wandb.
        # Per semplicità, ora chiediamo solo di impostare la chiave.

