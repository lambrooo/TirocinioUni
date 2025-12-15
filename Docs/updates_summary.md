# Riepilogo Aggiornamenti Recenti del Progetto

Questo documento riassume le principali funzionalità implementate e i documenti aggiornati nell'ultima iterazione del progetto.

---

## 1. Funzionalità Implementate nel Codice (`simulation.py`)

*   **Introduzione Metriche di Performance:**
    *   Implementata la nuova formula di `efficienza` (`load / (1 + beta * penalità_surriscaldamento)`) nella classe `Environment`.
    *   Calcolo e logging dell'efficienza ad ogni step.
*   **Sistema Economico:**
    *   Aggiunto un `budget` iniziale all'agente.
    *   Implementato il calcolo del `revenue` basato sull'efficienza, che aggiorna il `budget` ad ogni step.
*   **Azione Epistemica (Investimento):**
    *   Implementata la logica di investimento nella classe `Environment` (`trigger_investment()`).
    *   L'investimento ha un `costo` (budget) e causa un `downtime` (fermo produzione per un numero di step).
    *   Il beneficio è un miglioramento `permanente` del coefficiente di raffreddamento (`base_heat_coeff`).
*   **Logica Decisionale Agente Statico:**
    *   L'agente statico ora decide di investire se l'efficienza media recente scende sotto una soglia (`0.8`) e se ha budget sufficiente.
*   **Preparazione Cybersecurity (Base):**
    *   Aggiunto l'attributo `trust_score` alla classe `Sensor` (non ancora utilizzato).
    *   Il flag `is_under_attack` viene impostato quando si verifica un'anomalia (bias/outlier) nei sensori.
*   **Logging e Visualizzazione:**
    *   Il `simulation_log.csv` e i log di Weights & Biases includono ora tutte le nuove variabili (`budget`, `efficiency`, `revenue`, `investment_done`, `production_paused`, `base_heat_coeff`, `is_under_attack`).
    *   I grafici sono stati estesi a 5 subplot, includendo l'andamento del `budget` e dell'`efficienza`.
    *   Il momento dell'investimento è segnato da una linea verticale blu sui grafici.
    *   I periodi di "attacco" sono evidenziati con uno sfondo rosso.

---

## 2. Documentazione Aggiornata

*   **`Progettazione.md`:** Aggiornato con il piano di lavoro dettagliato, inclusa la nuova "Fase 4: Economia e Azione Epistemica" e la riorganizzazione delle fasi future.
*   **`Update.txt`:** Trasformato in una checklist operativa, con i task completati e i nuovi task per la Fase 4, e la riorganizzazione delle fasi future.
*   **`Analisi.md`:** Creato per documentare l'analisi delle performance dell'agente statico e motivare la necessità di un agente intelligente.
*   **`specs.md`:** Creato per fornire una panoramica tecnica completa e concisa del progetto per un lettore esterno.

---

## 3. Prossimi Passi (Pianificati)

*   **FASE 5: MODELLO AVANZATO DI MINACCE** (Implementazione di attacchi sofisticati e distinzione tra anomalie naturali e attacchi).
*   **FASE 6: AGENTE INTELLIGENTE (ACTIVE INFERENCE)** (Sviluppo dell'agente basato su ML/Active Inference).
