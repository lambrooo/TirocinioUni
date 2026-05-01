# Active Inference per Sicurezza Cyber-Physical

Questo progetto e' un testbed di simulazione per confrontare agenti di controllo in un sistema cyber-physical industriale. Il caso d'uso e' un motore industriale con dinamiche termiche, sensori rumorosi o attaccabili, costi di produzione, azioni di verifica e budget economico.

Il focus della tesi e' il confronto tra due agenti Active Inference con la stessa struttura decisionale:

- `ActiveInferenceAgent`: agente intelligente con modello di transizione fisso.
- `AdaptiveActiveInferenceAgent`: agente intelligente con learning online della B-matrix.

L'obiettivo e' misurare quando e quanto l'apprendimento del modello interno rende l'agente piu' efficace rispetto a un agente intelligente statico. Il controllore proporzionale `Agent` e gli agenti Q-Learning sono mantenuti come baseline secondarie.

## Idea Sperimentale

La simulazione mette l'agente davanti a un trade-off realistico:

- aumentare il carico produce piu' revenue, ma scalda il motore;
- il surriscaldamento riduce drasticamente l'efficienza;
- i sensori possono essere rumorosi, obsoleti o manipolati;
- verificare un sensore riduce l'incertezza, ma costa budget e downtime;
- il cyber-defense layer rileva solo alcune classi di attacco e ha un costo fisso.

In questo scenario, l'agente con learning puo' aggiornare il proprio modello delle transizioni osservate e adattarsi meglio alle dinamiche effettive dell'ambiente.

## Agenti Implementati

| Agente | Classe | Ruolo | Learning |
|--------|--------|-------|----------|
| Static controller | `Agent` | Baseline proporzionale | No |
| Active Inference statico | `ActiveInferenceAgent` | Confronto principale, modello fisso | No |
| Active Inference learning | `AdaptiveActiveInferenceAgent` | Confronto principale, B-matrix adattiva | Si |
| Q-Learning | `QLearningAgent` | Baseline model-free | Si |
| Double Q-Learning | `DoubleQLearningAgent` | Baseline model-free con riduzione overestimation | Si |

## Come Impara l'Agente Adaptive

L'agente learning mantiene una B-matrix per modellare le transizioni `P(s_next | s_current, action)`. A ogni step:

1. aggiorna le credenze sullo stato corrente tramite inferenza;
2. confronta credenze precedenti e correnti;
3. costruisce una transizione attesa con un outer product tra `curr_qs` e `prev_qs`;
4. aggiorna i conteggi `B_counts` con il learning rate corrente;
5. normalizza ogni colonna per mantenere una distribuzione di probabilita' valida.

Questo schema rende il learning interpretabile: non viene appresa direttamente una policy opaca, ma il modello interno delle dinamiche. Le metriche `model_divergence`, `avg_prediction_error` e `learning_rate` permettono di osservare come il modello cambia nel tempo.

## Risultati Inclusi

La run finale inclusa e':

```text
learning_experiments/run_20260430_180334
```

Contiene i dati grezzi, i summary CSV e i grafici degli esperimenti principali:

- confronto short-term Static AI vs Learning AI;
- confronto long-term Static AI vs Learning AI;
- sweep del learning rate;
- confronto tra schedule del learning rate;
- learning curves.

Le figure pronte per la tesi sono in `thesis_figures/`; le tabelle LaTeX sono in `thesis_tables/`.

## Struttura del Repository

```text
├── pytorch_simulation/
│   ├── active_inference_agent.py      # Active Inference statico e adaptive
│   ├── simulation.py                   # Ambiente, sensori, attuatori e loop
│   ├── dashboard.py                    # Dashboard Streamlit
│   ├── qlearning_agent.py              # Baseline Q-Learning
│   ├── b_matrix_viz.py                 # Visualizzazione B-matrix
│   ├── statistical_analysis.py         # Test statistici
│   ├── curriculum_learning.py          # Scenari progressivi
│   ├── test_agent_logic.py
│   ├── test_markov_chain.py
│   └── requirements.txt
├── Docs/
│   ├── README.md                       # Indice della documentazione
│   ├── TECHNICAL_DOCUMENTATION.md      # Documento tecnico principale
│   ├── agent_explanation.md            # Spiegazione Active Inference
│   ├── specs.md                        # Specifiche tecniche
│   └── get_started.md                  # Guida rapida
├── run_learning_experiments.py         # Esperimenti principali della tesi
├── generate_thesis_figures.py          # Figure e tabelle finali
└── test_all_features.py                # Test integrato delle feature
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pytorch_simulation/requirements.txt
```

## Esecuzione

Dashboard interattiva:

```bash
.venv/bin/python -m streamlit run pytorch_simulation/dashboard.py
```

Test principali:

```bash
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_agent_logic.py
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_markov_chain.py
WANDB_MODE=disabled .venv/bin/python test_all_features.py
```

Campagna principale della tesi:

```bash
WANDB_MODE=disabled .venv/bin/python run_learning_experiments.py --n_runs 10 --seed 42
```

Smoke test rapido:

```bash
WANDB_MODE=disabled .venv/bin/python run_learning_experiments.py --quick --exp 1 3 --seed 123
```

Generazione figure e tabelle:

```bash
WANDB_MODE=disabled .venv/bin/python generate_thesis_figures.py --full --seed 42
```

## Documentazione

Per una lettura veloce del progetto:

1. `README.md`
2. `Docs/README.md`
3. `Docs/TECHNICAL_DOCUMENTATION.md`
4. `Docs/agent_explanation.md`

I documenti `Docs/Progettazione.md`, `Docs/Update.txt` e `Docs/Analisi.md` restano utili come traccia progettuale e storico dello sviluppo, ma non sono necessari per capire il funzionamento finale.

## Note

`wandb/`, ambienti virtuali, cache Python e output temporanei non fanno parte del sorgente da consegnare. Gli artifact curati della tesi sono invece inclusi: run finale, figure e tabelle.
