# 🚀 Get Started

Segui questi passaggi per configurare ed eseguire il **Cyber-Physical Simulation Testbed**.

## 1. Prerequisiti

Assicurati di avere installato:
- **Python 3.8+**
- **pip** (Python package manager)

## 2. Installazione Dipendenze

Esegui questo comando dalla cartella principale del progetto (`Tirocinio`):

```bash
pip3 install -r pytorch_simulation/requirements.txt
```

> **Nota:** Se usi un ambiente virtuale (consigliato), attivalo prima di installare i pacchetti.

## 3. Avvio della Dashboard

Per lanciare l'interfaccia grafica interattiva:

```bash
python3 -m streamlit run pytorch_simulation/dashboard.py
```

Una volta avviato, il browser si aprirà automaticamente all'indirizzo `http://localhost:8501`.

## 4. Esecuzione Script (Senza GUI)

Se vuoi eseguire una singola simulazione da riga di comando e vedere i grafici statici:

```bash
python3 pytorch_simulation/simulation.py
```

## 5. Struttura del Progetto

- `pytorch_simulation/dashboard.py`: Codice dell'interfaccia Streamlit.
- `pytorch_simulation/simulation.py`: Logica core della simulazione (Environment, Agent, Sensors).
- `Docs/`: Documentazione del progetto.
