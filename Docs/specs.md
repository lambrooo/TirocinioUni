# Specifiche Tecniche del Progetto: Simulazione di Controllo con Agente Strategico e Minacce

---

## 1. Introduzione

Questo documento descrive le specifiche tecniche di una simulazione a tempo discreto progettata come "testbed" per la valutazione di agenti di controllo in ambienti dinamici, economici e caratterizzati da **incertezza**. L'obiettivo è studiare come agenti (sia statici che intelligenti basati su Machine Learning/Active Inference) gestiscono risorse economiche, prendono **decisioni strategiche per ridurre l'incertezza** (es. la verifica di un sensore) e mantengono le performance in presenza di rumore, dati obsoleti e attacchi informatici.

---

## 2. Architettura della Simulazione

La simulazione è composta da quattro componenti principali che interagiscono in un ciclo di feedback continuo.

### 2.1 Ambiente (Environment)

Rappresenta il sistema fisico e le sue dinamiche, arricchito da una dimensione economica.

*   **Variabili di Stato Fisiche:**
    *   `temperature` (°C): Temperatura ambiente (costante).
    *   `motor_temperature` (°C): Temperatura del motore.
    *   `load`: Carico di lavoro/produzione.
*   **Variabili di Stato Economiche e di Incertezza:**
    *   `budget`: Risorsa economica dell'agente.
    *   `cost_per_unit_of_load`, `fixed_operational_cost_per_step`: Parametri per i costi.
    *   `is_verifying_sensor` (bool): Flag che indica se è in corso un'azione di verifica.
    *   `verification_paused` (int): Contatore per il downtime dovuto alla verifica.
*   **Dinamiche:**
    *   La `motor_temperature` evolve in base al `load` e alla dissipazione.
    *   Il `load` effettivo è limitato dal `budget` disponibile.
    *   Il `budget` evolve in base a costi fissi, costi di produzione e `revenue` (guadagni) generati dall'efficienza.
*   **Azione Epistemica (Verifica):**
    *   L'ambiente gestisce l'azione di verifica:
        *   **Costo:** `verification_cost` (budget) e `verification_downtime` (step di produzione ridotta).
        *   **Beneficio:** Permette all'agente di accedere a una misurazione certa del sensore.
*   **Metrica di Performance (`efficiency`):**
    *   Calcolata ad ogni step: `efficiency = load / (1 + beta * max(0, motor_temperature - T_safe))`.
    *   `T_safe`: Soglia di temperatura sicura.
    *   `beta`: Fattore di penalità per il surriscaldamento.

### 2.2 Sensori (Sensors)

Misurano lo stato dell'ambiente, introducendo imperfezioni e incertezza.

*   **Lettura:** Forniscono letture di `temperature`, `motor_temperature` e `load`.
*   **Incertezza Strutturale:**
    *   **Rumore:** Ogni lettura è affetta da rumore gaussiano (`noise_std_dev`).
    *   **Campionamento Infrequente:** Le letture vengono aggiornate solo ogni `sampling_interval` step. Negli step intermedi, il sensore restituisce l'ultima lettura valida (dato obsoleto).
*   **Anomalie/Attacchi:** Possono introdurre `bias` o `outlier` nelle letture.
*   **Verifica:** Dispongono di un metodo `get_verified_reading()` che restituisce il valore vero del sensore, bypassando rumore e anomalie. Viene usato durante l'azione epistemica di verifica.
*   **`trust_score`:** Attributo per future implementazioni di gestione della fiducia.

### 2.3 Attuatori (Actuators)

Eseguono le azioni decise dall'agente sull'ambiente.

*   **Azioni:** Modificano `temperature` e `load` in base ai comandi ricevuti.
*   **Efficacia:** L'azione applicata è una frazione del comando ricevuto (`command * 0.1`).

### 2.4 Agenti (Agents)

Il sistema supporta tre tipi di agenti:

#### 2.4.1 Agente Statico
*   **Controllo:** Implementa un controllo proporzionale (P-controller) per `temperature` e `load`.
*   **Decisione Epistemica:** Segue una regola euristica: se `motor_temperature > T_safe + 10`, attiva la verifica.
*   **Limitazioni:** Non modella l'incertezza, non impara.

#### 2.4.2 Agente Intelligente (Active Inference)
*   **Controllo:** Basato su Active Inference con matrici A, B, C, D.
*   **Modalità EFE:**
    *   `full`: Bilancia azioni epistemiche e pragmatiche
    *   `epistemic_only`: Prioritizza la riduzione dell'incertezza
    *   `pragmatic_only`: Prioritizza il raggiungimento degli obiettivi
*   **Precision (τ):** Parametro configurabile (1-20) che controlla esplorazione/sfruttamento.
*   **Decisione Epistemica:** Ottimizzata in base all'Expected Free Energy, considerando costo opportunità.

#### 2.4.3 Agente Adattivo (Active Inference con Learning)
*   **Eredita da:** Agente Intelligente
*   **Learning Rate:** Parametro configurabile (0.001-0.1)
*   **Apprendimento Online:** Aggiorna la matrice B (transizioni) in base all'esperienza
*   **Vantaggio:** Su simulazioni lunghe, impara le dinamiche reali del sistema

---

## 3. Sistema di Cyber Defense

Layer di difesa automatico che rileva e neutralizza attacchi informatici.

*   **Costo:** Fisso 2.0 per step (quando attivo)
*   **Rileva:** Attacchi `dos` e `spoofing`
*   **Non Rileva:** Anomalie naturali (`bias`, `outlier`)
*   **Integrazione con Agente:** Fornisce osservazione `cyber_alert` (sì/no)

---

## 4. Ciclo di Simulazione (Feedback Loop)

La simulazione procede a passi discreti, con il seguente ciclo ad ogni step:

1.  **Introduzione Minacce:** Possibile introduzione di anomalie/attacchi nei sensori.
2.  **Cyber Defense:** Se attiva, rileva e neutralizza attacchi dos/spoofing.
3.  **Decisione Epistemica Agente:** L'agente valuta se è necessario eseguire un'azione di verifica.
4.  **Lettura Sensori:** I sensori leggono lo stato dell'ambiente.
5.  **Decisione di Controllo Agente:** L'agente decide i comandi per gli attuatori.
6.  **Apprendimento (solo Adaptive):** Aggiorna la matrice B con l'esperienza.
7.  **Azione Attuatori:** Gli attuatori applicano i comandi.
8.  **Aggiornamento Ambiente:** L'ambiente evolve.
9.  **Calcolo Performance ed Economia:** Viene calcolata l'efficienza e aggiornato il budget.
10. **Logging:** Tutti i dati rilevanti vengono registrati (WandB opzionale).

---

## 5. Metriche di Performance e Valutazione

La valutazione del sistema e degli agenti si basa su:

*   **`efficiency`:** Metrica istantanea di performance del sistema.
*   **`budget`:** Risorsa economica e indicatore di successo a lungo termine.
*   **`revenue`:** Guadagno generato ad ogni step, basato sull'efficienza.
*   **`budget_finale`:** Budget residuo alla fine della simulazione (indicatore chiave).
*   `performance_media`: Media dell'efficienza sull'intera simulazione.
*   `tempo_in_surriscaldamento`: Percentuale di step con `motor_temperature > T_safe`.
*   `errore_quadratico_medio_setpoint`: Stabilità del controllo.
*   `verifica_effettuata`: Flag (o contatore) per tracciare l'uso dell'azione epistemica.

---

## 6. Modello di Minacce (Anomalie e Attacchi)

*   **Anomalie Naturali:** `bias` e `outlier` introdotti casualmente (guasti sensore)
*   **Attacchi Informatici:** `spoofing` e `dos` (rilevabili dalla cyber defense)
*   **Probabilità:** Configurabili via dashboard (`attack_prob`)

---

## 7. Azione Epistemica (Verifica Sensore)

È l'azione strategica fondamentale che l'agente può compiere per gestire l'incertezza.

*   **Contesto:** A causa del campionamento infrequente e di possibili anomalie, l'agente opera con informazioni potenzialmente obsolete o false.
*   **Azione:** L'agente può decidere di attivare una "verifica" su un sensore.
*   **Costo:** `verification_cost` (40.0) + `verification_downtime` (3 step)
*   **Beneficio:** Misurazione immediata e certa, rimuove incertezza.
*   **Bilanciamento:** L'agente intelligente valuta il trade-off tramite EFE.

---

## 8. Dashboard Interattiva

Dashboard Streamlit per esperimenti e visualizzazione.

*   **URL:** `http://localhost:8501`
*   **Comando:** `python3 -m streamlit run pytorch_simulation/dashboard.py`

### Parametri Configurabili

| Parametro | Range | Descrizione |
|-----------|-------|-------------|
| Simulation Steps | 100 - 1,000,000 | Durata simulazione |
| Attack Probability | 0.0 - 0.1 | Probabilità attacco per step |
| Cyber Defense | On/Off | Attiva difesa automatica |
| Agent Type | static, intelligent, intelligent_adaptive | Tipo agente |
| EFE Mode | full, epistemic_only, pragmatic_only | Modalità EFE |
| Precision (τ) | 1.0 - 20.0 | Esplorazione/sfruttamento |
| Learning Rate | 0.001 - 0.1 | Velocità apprendimento (solo adaptive) |

### Tab Disponibili

1. **Single Simulation:** Esecuzione singola con grafici real-time
2. **Batch Experiment:** Confronto statistico Defense ON vs OFF
3. **Comparative Experiment:** Confronto multiplo configurazioni
4. **WandB History:** Storico esperimenti (richiede API key)

---

## 9. Strumenti e Tecnologie

*   **Linguaggio:** Python 3.9+
*   **Calcolo Numerico:** PyTorch, NumPy
*   **Analisi Dati:** Pandas
*   **Visualizzazione:** Plotly, Matplotlib
*   **Dashboard:** Streamlit
*   **Tracking Esperimenti:** Weights & Biases (opzionale)
*   **Gestione Configurazione:** python-dotenv

---

## 10. Confronti per Tesi

Il progetto supporta i seguenti confronti sperimentali:

1. **Static vs Intelligent:** Dimostra vantaggio di Active Inference
2. **Intelligent vs Adaptive:** Trova il "crossover point" dopo cui l'apprendimento paga
3. **EFE Modes:** Confronta full vs epistemic_only vs pragmatic_only
4. **Con/Senza Cyber Defense:** Valuta se l'agente può sostituire difese tradizionali
5. **Variazione Precision:** Effetto sul trade-off esplorazione/sfruttamento

