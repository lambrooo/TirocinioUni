# Get Started

Guida rapida per configurare ed eseguire il testbed di simulazione.

## 1. Prerequisiti

Prerequisiti:
- **Python 3.9+**
- **pip**

## 2. Installazione Dipendenze

Crea un ambiente virtuale dalla cartella principale del progetto (`Tirocinio`) e installa le dipendenze:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pytorch_simulation/requirements.txt
```

## 3. Avvio della Dashboard

Per lanciare l'interfaccia grafica interattiva:

```bash
.venv/bin/python -m streamlit run pytorch_simulation/dashboard.py
```

Una volta avviato, il browser si aprirà automaticamente all'indirizzo `http://localhost:8501`.

## 4. Esecuzione Script (Senza GUI)

Per validare il codice:

```bash
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_agent_logic.py
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_markov_chain.py
WANDB_MODE=disabled .venv/bin/python test_all_features.py
```

Per eseguire la campagna principale della tesi:

```bash
WANDB_MODE=disabled .venv/bin/python run_learning_experiments.py --n_runs 10 --seed 42
```

## 5. Struttura del Progetto

- `pytorch_simulation/dashboard.py`: Codice dell'interfaccia Streamlit.
- `pytorch_simulation/simulation.py`: Logica core della simulazione (Environment, Agent, Sensors).
- `pytorch_simulation/active_inference_agent.py`: Agenti Active Inference statico e con learning.
- `run_learning_experiments.py`: Esperimenti centrati sul confronto Static AI vs Learning AI.
- `Docs/`: Documentazione del progetto.
