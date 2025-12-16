# Active Inference for Cyber-Physical Systems Security

A simulation testbed for evaluating **Active Inference agents** in cyber-physical systems under adversarial conditions.

## 🎯 Project Overview

This project implements an intelligent agent based on **Active Inference** (Free Energy Principle) that can:
- Detect and respond to cyber attacks on sensor systems
- Balance **epistemic** (uncertainty reduction) and **pragmatic** (goal-directed) actions
- Operate in a simulated industrial control system environment

## 📁 Project Structure

```
├── pytorch_simulation/          # Core simulation code
│   ├── active_inference_agent.py   # Active Inference agent implementation
│   ├── simulation.py               # CPS environment simulation
│   ├── dashboard.py                # Streamlit interactive dashboard
│   └── requirements.txt            # Python dependencies
├── ActiveInference.md           # Mathematical framework documentation
├── run_thesis_experiments.py    # Batch experiment runner
├── run_efe_experiments.py       # EFE mode comparison experiments
└── thesis_experiments/          # Experiment results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r pytorch_simulation/requirements.txt
```

### 2. Run the Dashboard

```bash
python3 -m streamlit run pytorch_simulation/dashboard.py
```

### 3. Access the Interface

Open your browser at `http://localhost:8501`

## 🧠 EFE Modes (Expected Free Energy)

| Mode | Behavior | Use Case |
|------|----------|----------|
| `full` | Balances exploration and exploitation | General operation |
| `epistemic_only` | Prioritizes uncertainty reduction | High-threat environments |
| `pragmatic_only` | Prioritizes goal achievement | Trusted environments |

## 📊 Key Features

- **Real-time Simulation**: Monitor temperature, load, and budget dynamics
- **Batch Experiments**: Statistical comparison of configurations
- **WandB Integration**: Track and analyze experiment history
- **Cyber Attack Simulation**: Test agent resilience under sensor spoofing

## 🔬 Research Context

This project is part of a thesis on applying Active Inference to cyber-physical systems security, demonstrating how agents can autonomously detect and mitigate sensor-based attacks.

## 📝 License

Academic use only.