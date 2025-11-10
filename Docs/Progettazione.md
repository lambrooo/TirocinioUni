# Documento di Progettazione: Simulazione Avanzata per Agenti di Controllo

**Autore:** Leonardo Lambruschi 
**Data:** 6 Novembre 2025

---

### **Obiettivo Generale**

L'obiettivo di questo progetto è sviluppare una simulazione avanzata che serva come banco di prova ("testbed") per la valutazione di agenti di controllo. Il sistema simulato permetterà di confrontare le performance di agenti statici (basati su regole) e agenti intelligenti (es. Active Inference) in condizioni operative normali e sotto l'effetto di attacchi informatici mirati all'integrità dei dati dei sensori.

---

### **Fase 1: Ridefinizione dell'Ambiente e delle Metriche (Durata Stimata: 1 settimana)**

In questa fase, l'obiettivo è introdurre un concetto di "successo" misurabile all'interno della simulazione, passando da un modello puramente descrittivo a uno prescrittivo.

*   **Task Tecnici:**
    1.  **Introdurre la Metrica di Performance**: Modificare la classe `Environment` per calcolare una metrica di performance ad ogni step. La formula scelta è:
        `Performance = Carico - Beta * max(0, TemperaturaMotore - T_safe)`
        Questa metrica bilancia la produttività (`Carico`) con la sicurezza operativa (penalizzando il surriscaldamento).
    2.  **Aggiornare il Logging**: Estendere il logging su file CSV e su Weights & Biases per includere la nuova metrica di `performance` e le altre variabili rilevanti.
    3.  **Aggiornare la Visualizzazione**: Aggiungere un quarto subplot ai grafici generati per mostrare l'andamento della `performance` nel tempo.

*   **Metriche Chiave da Misurare:**
    *   `performance_istantanea`: Calcolata ad ogni step della simulazione.
    *   `performance_media`: Media della performance sull'intera durata della simulazione.
    *   `tempo_in_surriscaldamento`: Percentuale di step in cui `motor_temperature > T_safe`.
    *   `errore_quadratico_medio_setpoint`: Deviazione quadratica media dei valori misurati dai loro setpoint, per misurare la stabilità del controllo.

---

### **Fase 2: Introduzione alla Cybersecurity (Concettuale e Base di Codice) (Durata Stimata: 1 settimana)**

Questa fase prepara il terreno per la ricerca sugli attacchi informatici, documentando il piano e implementando solo le fondamenta nel codice. Nella sua forma attuale, il concetto di "anomalia" e "attacco" sono unificati. Il piano a lungo termine prevede una distinzione più netta.

*   **Task Tecnici Iniziali:**
    1.  **Documentare Tipi di Attacco**: Dettagliare gli attacchi da simulare (`bias`, `outlier`, `spoofing`, `DoS`).
    2.  **Introdurre il Concetto di `fiducia`**: Aggiungere un attributo `trust_score` alla classe `Sensor`.
    3.  **Implementare un Flag di Attacco Generico**: Aggiungere al log una colonna `is_under_attack` che si attiva in presenza di qualsiasi anomalia deliberata.

*   **Metriche Chiave da Misurare (in futuro):**
    *   `robustezza_performance`: `performance_media_scenario_pulito - performance_media_scenario_attaccato`.
    *   `tempo_di_rilevamento`: Tempo impiegato da un agente intelligente per identificare un'anomalia.

---

### **Fase 2.1: Modello Avanzato di Minacce (Sviluppo Futuro)**

Per consentire una valutazione rigorosa dei meccanismi di cybersecurity, verrà implementato un modello di minacce più sofisticato che distingue tra due tipi di eventi:

1.  **Anomalie Naturali:**
    *   **Cosa sono:** Guasti hardware casuali e non malevoli.
    *   **Come si simulano:** Con una probabilità molto bassa ad ogni step (es. 0.1%), un sensore può generare un'anomalia lieve (piccolo `bias` o `outlier`).
    *   **Scopo:** Rappresentano l'inaffidabilità intrinseca del mondo reale, il "rumore di fondo" che ogni sistema deve tollerare.

2.  **Attacchi Informatici:**
    *   **Cosa sono:** Eventi deliberati e malevoli.
    *   **Come si simulano:** Attivati da un trigger separato (es. probabilistico con probabilità più alta o in intervalli specifici), possono essere più gravi e complessi (`spoofing`, `DoS`, `bias` elevato).
    *   **Scopo:** Rappresentano le minacce attive contro cui i meccanismi di cybersecurity devono difendere il sistema.

Questa distinzione abiliterà una **metodologia di test a tre scenari**:
-   **Scenario A (Baseline):** Simulazione con sole anomalie naturali. Misura la performance ideale del sistema.
-   **Scenario B (Attacco senza Difese):** Aggiunge gli attacchi informatici. Misura la vulnerabilità del sistema.
-   **Scenario C (Attacco con Difese):** Aggiunge gli attacchi ma con i meccanismi di cybersecurity attivi. Misura l'efficacia delle difese confrontando i risultati con gli scenari A e B.

---

### **Fase 3: Sviluppo dell'Agente Intelligente (Active Inference) (Durata Stimata: 2+ settimane)**

Questa è la fase di ricerca principale, che sfrutta le fondamenta costruite nelle fasi precedenti.

*   **Task Tecnici:**
    1.  **Sviluppo dell'Agente Active Inference**: Sostituire l'agente statico con un agente basato su Active Inference, utilizzando la libreria `pymdp`.
    2.  **Discretizzazione dell'Ambiente**: Sviluppare una funzione che mappi gli stati continui dell'ambiente (es. 85°C) in stati discreti (es. "Caldo", "Surriscaldato") comprensibili per l'agente.
    3.  **Costruzione del Modello Generativo**: Definire le matrici probabilistiche (A, B, C, D) che codificano la "conoscenza" e gli "obiettivi" dell'agente.
        *   `A`: Come gli stati del mondo generano le osservazioni.
        *   `B`: Come le azioni cambiano gli stati del mondo.
        *   `C`: Le preferenze dell'agente (le osservazioni desiderate).
    4.  **Valutazione Comparativa**: Confrontare sistematicamente le metriche di performance (`performance_media`, `robustezza`, `stabilità`) dell'agente intelligente con quelle dell'agente statico, in scenari con e senza attacchi.

---