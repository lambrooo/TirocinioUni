
# Spiegazione della Simulazione in Python (per Sviluppatori Java)

Questo documento analizza lo script `simulation.py`, spiegandone la struttura e la sintassi con parallelismi al linguaggio Java per facilitare la comprensione.

---

## 1. Concetti di Base del Progetto

Prima di analizzare il codice, è importante capire l'idea di fondo. Stiamo creando un "Digital Twin" (un gemello digitale) estremamente semplificato di un sistema fisico, come potrebbe essere un macchinario industriale, un motore o una stanza.

### a. I Componenti Fondamentali

- **Environment (Ambiente)**
  - **Cosa rappresenta?** È il mondo fisico o il sistema che vogliamo simulare. Ha uno stato interno (es. la sua temperatura attuale, il carico su un motore) e delle "leggi fisiche" che lo governano (es. un oggetto caldo tende a raffreddarsi, un motore sotto sforzo si scalda).
  - **Nel codice:** La classe `Environment` mantiene lo stato corrente delle variabili e il suo metodo `step()` calcola come questo stato cambia nel tempo, anche in base a influenze esterne.

- **Sensor (Sensore)**
  - **Cosa rappresenta?** Un dispositivo di misurazione del mondo reale (un termometro, un tachimetro). I sensori non sono perfetti: hanno sempre un margine di errore (il **rumore**) e a volte possono guastarsi o dare letture completamente sballate (le **anomalie**).
  - **Nel codice:** La classe `Sensor` non conosce il valore "vero" dell'ambiente. Lo legge e ci aggiunge del rumore casuale (gaussiano) per simulare l'imprecisione. Può anche introdurre anomalie come `bias` (un errore costante) o `outlier` (un valore singolo e assurdo).

- **Actuator (Attuatore)**
  - **Cosa rappresenta?** Un dispositivo che può *agire* sull'ambiente per modificarlo. Esempi sono un condizionatore che raffredda una stanza, un acceleratore che aumenta il carico su un motore, una valvola che regola un flusso.
  - **Nel codice:** La classe `Actuator` riceve un comando generico e lo traduce in un'azione concreta che viene passata all'ambiente.

- **Agent (Agente)**
  - **Cosa rappresenta?** È il "cervello" del sistema, il controllore. Il suo compito è leggere i dati (imperfetti) provenienti dai sensori e decidere quali comandi inviare agli attuatori per far sì che l'ambiente raggiunga o mantenga uno stato desiderato (chiamato **setpoint**).
  - **Nel codice:** La classe `Agent` implementa la logica di controllo. In questo caso, un semplice controllo proporzionale: più il valore misurato è lontano dall'obiettivo, più forte sarà il comando per correggerlo.

### b. Il Concetto Chiave: Il "Feedback Loop" (Ciclo di Retroazione)

Questi componenti non agiscono in modo isolato, ma creano un ciclo continuo, che è il cuore di ogni sistema di controllo:

1.  Il **Sensore** misura lo stato dell'**Ambiente** -> `(lettura rumorosa)`
2.  L'**Agente** riceve la lettura, la confronta con il suo obiettivo (`setpoint`) e calcola un'azione correttiva -> `(comando)`
3.  L'**Attuatore** riceve il comando e lo traduce in un'azione fisica -> `(azione)`
4.  L'azione modifica lo stato dell'**Ambiente**.
5.  ...e il ciclo ricomincia dal punto 1.

Questo ciclo `Ambiente -> Sensore -> Agente -> Attuatore -> Ambiente` è la base della simulazione. Il nostro codice non fa altro che ripetere questo ciclo per `N` passi, registrando tutto ciò che accade.

---

## 2. Obiettivo del Programma

Lo script implementa la simulazione di questo ciclo di feedback. Lo scopo è generare dati realistici (ma simulati) che potrebbero essere usati, ad esempio, per addestrare un modello di machine learning a predire guasti (anomaly detection) o a ottimizzare il controllo.

---

## 3. Come Eseguire lo Script

1.  **Prerequisiti**: Assicurati di avere Python 3 e `pip3` installati.
2.  **Setup Chiave API**: Apri il file `.env` in questa cartella e inserisci la tua chiave API di Weights & Biases.
3.  **Installazione Dipendenze**: Esegui questo comando dalla directory radice del progetto (`Tirocinio`):
    ```bash
    pip3 install -r pytorch_simulation/requirements.txt
    ```
4.  **Esecuzione**: Esegui lo script con il comando:
    ```bash
    python3 pytorch_simulation/simulation.py
    ```

---

## 3. Analisi del Codice

### a. Importazioni

```python
import torch
import pandas as pd
import wandb
import random
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
```

Questo blocco è l'equivalente dell'`import` in Java. Importiamo le librerie necessarie:
- `torch`: PyTorch, usato per i "tensori". Immaginalo come un `float[]` specializzato per calcoli scientifici.
- `pandas`: Una libreria per la manipolazione di dati tabellari. L'oggetto principale, il `DataFrame`, è simile a una `List<Map<String, Object>>` ma molto più potente.
- `wandb`: La libreria di Weights & Biases per il logging.
- `random`: Equivalente alla classe `java.util.Random`.
- `matplotlib.pyplot`: Una libreria per creare grafici (simile a JFreeChart).
- `os`: Per interagire con il sistema operativo (es. leggere variabili d'ambiente).
- `dotenv`: Per caricare configurazioni da file `.env`.

### b. Inizializzazione

```python
load_dotenv()
wandb.init(project="pytorch-simulation")
```
- `load_dotenv()`: Cerca un file `.env` e carica le variabili definite al suo interno.
- `wandb.init()`: Inizializza la connessione con Weights & Biases, creando un nuovo "esperimento" (run).

### c. Le Classi

In Python, le classi sono definite dalla parola chiave `class`. Il costruttore si chiama `__init__` e il primo parametro, `self`, è l'equivalente del `this` di Java.

#### `Environment`
```python
class Environment:
    def __init__(self, initial_temp=20.0, ...):
        self.temperature = torch.tensor([initial_temp], dtype=torch.float32)
        # ...
    def step(self, temp_action, load_action):
        # ...
```
- **Costruttore (`__init__`)**: Inizializza le variabili d'istanza (es. `self.temperature`). I parametri con un valore di default (es. `initial_temp=20.0`) sono opzionali.
- **Metodo `step`**: Simula l'evoluzione dell'ambiente in un passo temporale, modificando le variabili interne in base alle azioni degli attuatori e a delle leggi fisiche predefinite.

#### `Sensor`
```python
class Sensor:
    def __init__(self, name, noise_std_dev=0.5):
        self.anomaly = None # 'None' è il 'null' di Python
        # ...
    def read(self, true_value):
        reading = true_value + torch.randn(1) * self.noise_std_dev
        if self.anomaly: # Controllo rapido per 'self.anomaly != None'
            # ...
        return reading
    def introduce_anomaly(self, anomaly_type, value):
        self.anomaly = {'type': anomaly_type, 'value': value}
```
- **`torch.randn(1)`**: Genera un numero casuale da una distribuzione normale (gaussiana), usato per simulare il rumore del sensore.
- **`self.anomaly = {'type': ...}`**: Crea un **dizionario** (l'equivalente di una `Map<String, Object>`) per memorizzare le informazioni sull'anomalia.

#### `Agent`
Implementa la logica di controllo. Il metodo `get_commands` calcola l'errore tra il valore desiderato (`setpoint`) e quello misurato, restituendo un comando proporzionale a tale errore.

### d. Funzione Principale (`run_simulation`)

```python
def run_simulation(n_steps=1000, ...):
    # 1. Inizializzazione
    env = Environment()
    sensors = { 'temperature': Sensor(...), ... } # Dizionario (Map) di sensori
    log_data = [] # Lista (ArrayList) vuota

    # 2. Loop di simulazione
    for step in range(n_steps): # Equivalente di for(int i=0; i<n; i++)
        # ... Logica del passo ...
        log_data.append(log_entry) # Aggiunge un elemento alla lista

    # 3. Salvataggio
    df = pd.DataFrame(log_data) # Crea una tabella dai dati raccolti
    df.to_csv("simulation_log.csv", index=False) # Salva la tabella in CSV
```
Questa funzione orchestra l'intera simulazione:
1.  Crea le istanze di tutti gli oggetti.
2.  Esegue un loop per `n_steps` volte. In ogni ciclo: legge i sensori, calcola i comandi, aggiorna l'ambiente e registra i dati.
3.  Alla fine, usa `pandas` per convertire la lista di log in un `DataFrame` e salvarlo facilmente come file CSV.

### e. Punto di Ingresso

```python
if __name__ == "__main__":
    run_simulation()
```
Questo è un costrutto standard in Python. Il codice al suo interno viene eseguito solo se lo script è il file principale che viene eseguito. È l'equivalente del `public static void main(String[] args)` in Java.
