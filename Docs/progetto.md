# Report Tecnico del Progetto: Simulazione di Controllo con Agente Strategico e Minacce

**Autore:** Leonardo Lambruschi
**Versione:** 1.0
**Data:** 17 Novembre 2025

---

## 1. Introduzione e Obiettivo Generale

Il progetto mira a sviluppare una simulazione a tempo discreto che funga da "testbed" per la valutazione di agenti di controllo in ambienti dinamici, economici e caratterizzati da **incertezza**. L'obiettivo finale è confrontare le performance di agenti statici con quelle di agenti intelligenti (basati su Machine Learning o Active Inference), misurando la loro capacità di gestire risorse, prendere decisioni strategiche per **ridurre l'incertezza** e mantenere l'efficienza in presenza di rumore, dati obsoleti e attacchi informatici.

Questo report descrive l'architettura attuale della simulazione, le funzionalità implementate, le considerazioni di progettazione e le prospettive future.

---

## 2. Architettura della Simulazione

La simulazione è basata su un'architettura a ciclo di feedback continuo, composta da quattro componenti principali: Ambiente, Sensori, Attuatori e Agente.

### 2.1 Ambiente (Environment)

Rappresenta il sistema fisico controllato, arricchito da dinamiche economiche.

*   **Variabili di Stato Fisiche:**
    *   `temperature` (°C): Temperatura ambiente (costante).
    *   `motor_temperature` (°C): Temperatura del motore.
    *   `load`: Carico di lavoro/produzione.
*   **Dinamiche Fisiche:**
    *   La `motor_temperature` evolve in base al `load` (riscaldamento, influenzato da `base_heat_coeff`) e alla differenza con `temperature` (dissipazione, influenzata da `dissipation_coeff`).
    *   Il `load` è influenzato dagli attuatori e dal budget disponibile.
*   **Variabili di Stato Economiche e Strategiche:**
    *   `budget`: Risorsa economica dell'agente, che evolve ad ogni step.
    *   `cost_per_unit_of_load`: Costo per produrre una unità di `load`.
    *   `fixed_operational_cost_per_step`: Costo fisso dedotto dal budget ad ogni step.
    *   `is_verifying_sensor` (bool): Flag che indica se è in corso un'azione di verifica di un sensore.
    *   `verification_paused` (int): Contatore per il downtime dovuto alla verifica.
*   **Meccanismi di Azione Strategica:**
    *   **Azione Epistemica di Verifica Sensore**: È l'azione strategica chiave. Permette all'agente di pagare un costo per ottenere una misurazione certa di un sensore, riducendo l'incertezza causata da campionamento infrequente o anomalie.
        *   **Costo:** `verification_cost` (budget) e `verification_downtime` (step di produzione ridotta).
        *   **Beneficio:** Ottenimento di una misurazione "certa" (valore vero) del sensore, che permette decisioni di controllo più informate.
    *   **Investimento nel Cooling System (Temporaneamente Disabilitato):**
        *   La logica per un investimento a lungo termine che migliora l'efficienza del sistema è presente ma disabilitata nel codice attuale per focalizzare l'analisi sulla gestione dell'incertezza.
*   **Metrica di Performance (`efficiency`):**
    *   Calcolata ad ogni step tramite la formula: `efficiency = load / (1 + beta * max(0, motor_temperature - T_safe))`.
    *   `T_safe`: Soglia di temperatura sicura.
    *   `beta`: Fattore di penalità per il surriscaldamento.

### 2.2 Sensori (Sensors)

Misurano lo stato dell'ambiente, introducendo imperfezioni e incertezza.

*   **Lettura:** Forniscono letture di `temperature`, `motor_temperature` e `load`.
*   **Rumore:** Ogni lettura è affetta da rumore gaussiano (`noise_std_dev`).
*   **Campionamento Infrequente:** Le letture vengono fornite solo ogni `sampling_interval` step. Negli step intermedi, il sensore restituisce l'ultima lettura valida, aumentando l'incertezza.
*   **Anomalie/Attacchi:** Possono introdurre `bias` (scarto costante) o `outlier` (valore anomalo singolo) nelle letture. Il flag `is_under_attack` le identifica.
*   **`trust_score`:** Un attributo (attualmente statico a 1.0) per future implementazioni di autenticazione e gestione della fiducia.
*   **`get_verified_reading()`:** Metodo che restituisce il valore vero del sensore, utilizzato durante l'azione epistemica di verifica.

### 2.3 Attuatori (Actuators)

Eseguono le azioni decise dall'agente sull'ambiente.

*   **Azioni:** Modificano `temperature` e `load` in base ai comandi ricevuti.
*   **Efficacia:** L'azione applicata è una frazione del comando ricevuto (`command * 0.1`).

### 2.4 Agente (Agent)

Il controllore del sistema.

*   **Agente Statico (attuale):**
    *   **Controllo:** Implementa un controllo proporzionale (P-controller) per `temperature` e `load`.
    *   **Decisione Epistemica (Verifica):** Segue una regola euristica: se la `motor_temperature` letta è sospettosamente alta (`> T_safe + 10`), e se ha budget sufficiente, l'agente attiva l'azione di verifica per ottenere una misurazione certa. Questa è la sua unica capacità strategica al momento.
    *   **Decisione di Investimento (Disabilitata):** La logica per l'investimento a lungo termine è disabilitata.
*   **Agente Intelligente (futuro):** Sarà basato su Machine Learning o Active Inference, con l'obiettivo di apprendere una policy ottimale per decidere quando è vantaggioso pagare il costo della verifica.

---

## 3. Ciclo di Simulazione e Flusso Economico

La simulazione procede a passi discreti (`N_steps`), con il seguente ciclo ad ogni step:

1.  **Costi Operativi Fissi:** `fixed_operational_cost_per_step` viene dedotto dal budget.
2.  **Introduzione Minacce:** Possibile introduzione di anomalie (`bias`/`outlier`) nei sensori.
3.  **Decisione Epistemica dell'Agente:**
    *   L'agente valuta se attivare la verifica di un sensore in base alla sua euristica.
4.  **Lettura Sensori:** I sensori forniscono letture (potenzialmente incerte). Se in verifica, il sensore interrogato fornisce il valore vero.
5.  **Comandi Agente:** L'agente calcola i comandi di controllo in base alle letture (eventualmente verificate). L'agente è inattivo durante il downtime da verifica.
6.  **Azione Attuatori:** Gli attuatori applicano i comandi.
7.  **Aggiornamento Ambiente:**
    *   L'ambiente evolve (temperatura, ecc.).
    *   Il `load` effettivo è limitato dal budget. Il suo costo viene dedotto.
    *   Gestione del timer di downtime per la verifica.
8.  **Calcolo Performance ed Economia:**
    *   Viene calcolata l'`efficiency` e il `revenue` corrispondente viene aggiunto al `budget`.
9.  **Logging:** Tutti i dati rilevanti vengono registrati.

---

## 4. Metriche di Performance e Valutazione

La valutazione del sistema e degli agenti si basa su:

*   **`budget_finale`:** Il budget residuo alla fine della simulazione (indicatore chiave di successo economico).
*   **`performance_media`:** Media dell'efficienza sull'intera simulazione.
*   `budget_max_raggiunto`: Il picco di budget raggiunto.
*   `tempo_in_surriscaldamento`: Percentuale di step con `motor_temperature > T_safe`.
*   `errore_quadratico_medio_setpoint`: Stabilità del controllo.
*   `verifica_effettuata`: Flag o contatore per tracciare l'uso dell'azione epistemica.
*   `costo_produzione_totale`: Somma dei costi di produzione.
*   `costo_operativo_fisso_totale`: Somma dei costi fissi.
*   `ricavi_totali`: Somma dei ricavi.

---

## 5. Modello di Minacce e Incertezza

*   **Fonti di Incertezza:**
    *   **Rumore Gaussiano:** Nelle letture dei sensori.
    *   **Campionamento Infrequente:** Dati "stale" tra un campionamento e l'altro.
    *   **Anomalie/Attacchi:** `bias` o `outlier` introdotti periodicamente nei sensori.
*   **Futuro (FASE 5):** Il modello verrà esteso per distinguere tra anomalie naturali (guasti casuali a bassa probabilità) e attacchi informatici sofisticati (`spoofing`, `DoS`), con trigger e logiche separate.

---

## 6. Considerazioni e Decisioni di Progettazione

*   **Ruolo dell'Agente Statico:** Serve come baseline per dimostrare i limiti di una strategia reattiva e basata su regole semplici in un ambiente complesso. Le sue "carenze" sono funzionali a evidenziare il valore di un agente intelligente.
*   **Evoluzione del Budget:** Il budget è passato da una semplice metrica a una risorsa attiva e consumabile ad ogni step, rendendo la gestione economica centrale.
*   **Valore dell'Azione Epistemica:** Il campionamento infrequente e i costi operativi fissi rendono l'azione di verifica cruciale per la gestione dell'incertezza, giustificandone il costo.
*   **Azione di Investimento Commentata:** La logica decisionale per l'investimento nel cooling è stata commentata per permettere una focalizzazione iniziale sull'azione di verifica e per semplificare l'analisi del comportamento dell'agente statico.

---

## 7. Prospettive Future

Il progetto è concepito per evolvere verso:

*   **Implementazione Agente Intelligente (ML/Active Inference):** Sostituire l'agente statico con un agente che apprende a ottimizzare le decisioni di controllo e strategiche (investimento, verifica) in base all'esperienza.
*   **Modello Avanzato di Minacce e Difese Cybersecurity:** Implementare attacchi più sofisticati e meccanismi di difesa, valutando la robustezza dell'agente.
*   **Analisi Comparativa:** Confrontare sistematicamente le performance tra agenti statici e intelligenti in scenari complessi e ostili.

---

## 8. Strumenti e Tecnologie

*   **Linguaggio:** Python
*   **Calcolo Numerico:** PyTorch (per i tensori)
*   **Analisi Dati:** Pandas
*   **Visualizzazione:** Matplotlib, Weights & Biases (wandb)
*   **Gestione Configurazione:** python-dotenv
*   **Gestione Versioni:** Git/GitHub
