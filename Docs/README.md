# Documentazione del Progetto

Questa cartella contiene la documentazione tecnica e progettuale del testbed. L'obiettivo e' rendere chiaro il progetto anche a chi non ha seguito lo sviluppo: scenario simulato, agenti implementati, metodo di apprendimento e risultati sperimentali.

## Lettura Consigliata

1. `../README.md`
   Panoramica generale del progetto, focus della tesi, setup ed esecuzione.

2. `TECHNICAL_DOCUMENTATION.md`
   Documento tecnico principale. Descrive architettura, agenti, B-matrix learning, strumenti di analisi e risultati inclusi.

3. `agent_explanation.md`
   Spiegazione compatta dell'agente Active Inference, del modello generativo e della differenza tra versione fissa e versione con learning.

4. `specs.md`
   Specifiche operative della simulazione: ambiente, sensori, attuatori, agenti, metriche e dashboard.

5. `get_started.md`
   Guida rapida con i comandi principali.

## Documenti di Supporto

- `EXPLAIN.md`: spiegazione concettuale di azioni epistemiche, pragmatiche e matrici del modello.
- `progetto.md`: report progettuale aggiornato con il ruolo dei diversi agenti.
- `Analisi.md`: analisi iniziale della baseline statica.
- `Progettazione.md`: traccia storica della progettazione iniziale.
- `Update.txt`: checklist dello sviluppo.

## Focus della Tesi

Il confronto principale non e' semplicemente "agente statico vs agente intelligente", ma:

```text
ActiveInferenceAgent fisso
vs
AdaptiveActiveInferenceAgent con learning online della B-matrix
```

Entrambi sono agenti intelligenti basati su Active Inference. La differenza controllata e' la capacita' del secondo di aggiornare il modello interno delle transizioni durante la simulazione.

## Risultati e Artifact

La run finale usata come riferimento e':

```text
learning_experiments/run_20260430_180334
```

Le figure finali sono in:

```text
thesis_figures/
```

Le tabelle LaTeX sono in:

```text
thesis_tables/
```

Questi artifact permettono di verificare direttamente le conclusioni sperimentali senza dover rigenerare subito tutta la campagna.
