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

### 2.4 Agente (Agent)

Il controllore del sistema.

*   **Agente Statico (attuale):**
    *   **Controllo:** Implementa un controllo proporzionale (P-controller) per `temperature` e `load`.
    *   **Decisione Epistemica (Verifica):** Segue una regola euristica: se la temperatura del motore letta (`sensed_motor_temp`) supera una soglia di allarme (es. `T_safe + 10`), e se ha budget sufficiente, l'agente attiva l'azione di verifica per ottenere una misurazione certa e dirimere l'incertezza.
*   **Agente Intelligente (futuro):** Sarà basato su Machine Learning o Active Inference, imparando a ottimizzare le decisioni di controllo e, soprattutto, a decidere quando è economicamente vantaggioso eseguire l'azione di verifica.

---

## 3. Ciclo di Simulazione (Feedback Loop)

La simulazione procede a passi discreti, con il seguente ciclo ad ogni step:

1.  **Introduzione Minacce:** Possibile introduzione di anomalie/attacchi nei sensori.
2.  **Decisione Epistemica Agente:** L'agente valuta se è necessario eseguire un'azione di verifica del sensore in base alle letture ricevute e al suo stato interno.
3.  **Lettura Sensori:** I sensori leggono lo stato dell'ambiente (con rumore, dati obsoleti, e potenziali anomalie). Se la verifica è attiva, il sensore interessato fornisce una lettura certa.
4.  **Decisione di Controllo Agente:** L'agente riceve le letture (eventualmente verificate) e decide i comandi per gli attuatori.
5.  **Azione Attuatori:** Gli attuatori applicano i comandi all'ambiente.
6.  **Aggiornamento Ambiente:** L'ambiente evolve in base alle sue dinamiche e alle azioni.
7.  **Calcolo Performance ed Economia:** Viene calcolata l'efficienza e aggiornato il budget.
8.  **Logging:** Tutti i dati rilevanti vengono registrati.

---

## 4. Metriche di Performance e Valutazione

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

## 5. Modello di Minacce (Anomalie e Attacchi)

*   **Attuale:** Le "minacce" sono implementate come anomalie dei sensori (`bias` o `outlier`) introdotte periodicamente. Il flag `is_under_attack` le identifica.
*   **Futuro (FASE 5):** Il modello verrà esteso per distinguere tra anomalie naturali (guasti casuali a bassa probabilità) e attacchi informatici sofisticati (`spoofing`, `DoS`), con trigger e logiche separate.

---

## 6. Azione Epistemica (Verifica Sensore)

È l'azione strategica fondamentale che l'agente può compiere per gestire l'incertezza.

*   **Contesto:** A causa del campionamento infrequente e di possibili anomalie, l'agente opera con informazioni potenzialmente obsolete o false. Un picco di temperatura potrebbe essere reale o un errore del sensore.
*   **Azione:** L'agente può decidere di attivare una "verifica" su un sensore.
*   **Costo:** L'azione ha un costo monetario (`verification_cost`) e un costo operativo (`verification_downtime` con produzione ridotta).
*   **Beneficio:** L'agente ottiene una misurazione immediata e certa dal sensore, rimuovendo ogni incertezza e permettendogli di prendere una decisione di controllo informata.
*   **Scopo:** Testare la capacità dell'agente di gestire il trade-off tra il costo di acquisire informazione e il rischio di agire sulla base di informazione incerta.

---

## 7. Strumenti e Tecnologie

*   **Linguaggio:** Python
*   **Calcolo Numerico:** PyTorch (per i tensori)
*   **Analisi Dati:** Pandas
*   **Visualizzazione:** Matplotlib, Weights & Biases (wandb)
*   **Gestione Configurazione:** python-dotenv

---

## 8. Prospettive Future

Il progetto è concepito per evolvere verso:
*   L'implementazione di un agente intelligente basato su Active Inference (`pymdp`).
*   Lo sviluppo di meccanismi di cybersecurity avanzati e la valutazione della loro efficacia.
*   L'analisi comparativa delle performance tra agenti statici e intelligenti in scenari complessi e ostili.
