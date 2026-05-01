
# THESIS EXPERIMENTS: STATIC vs LEARNING AGENT
=============================================

## Obiettivo della Tesi
Confronto tra un agente Active Inference con modello statico (B matrix fissa)
e un agente che apprende le dinamiche di transizione online.

## Configurazione di Esecuzione
- W&B disabilitato di default nello script (`WANDB_MODE=disabled`) per evitare
  run online e file locali non necessari.
- Seed base: usare `--seed 42` per riproducibilita'.
- Campagna consigliata per la tesi: `--n_runs 10 --seed 42`.
- Interprete consigliato nel repository: `.venv/bin/python`.

## Esperimenti Eseguiti

### Esperimento 1: Static vs Learning (Short-term)
- Confronta performance su simulazioni brevi (500 steps)
- Ipotesi: L'agente statico potrebbe avere un vantaggio iniziale
- File: exp1_static_vs_learning_short.png

### Esperimento 2: Static vs Learning (Long-term)
- Confronta performance su simulazioni lunghe (5000 steps)
- Ipotesi: L'agente che apprende dovrebbe migliorare nel tempo
- Metriche chiave: miglioramento efficienza tra prima e seconda meta
- File: exp2_static_vs_learning_long.png

### Esperimento 3: Learning Rate Comparison
- Valuta l'impatto di diversi valori di learning rate
- Range testato: 0.001 - 0.05
- File: exp3_learning_rate.png

### Esperimento 4: LR Schedule Comparison
- Confronta diversi approcci di scheduling:
  * Constant: LR fisso
  * Exponential Decay: LR decresce esponenzialmente
  * Adaptive: LR si adatta in base all'errore di predizione
- File: exp4_lr_schedules.png

### Esperimento 5: Learning Curves Analysis
- Analisi dettagliata della convergenza del modello
- Metriche: divergenza del modello, errore di predizione, performance
- File: exp5_learning_curves.png

## Interpretazione

1. **Short-term**: misura se il learning paga anche quando il tempo di
   adattamento e' limitato.

2. **Long-term**: verifica se l'aggiornamento online della B-matrix produce
   un vantaggio stabile rispetto al modello fisso.

3. **Learning Rate**: valori troppo alti possono causare instabilita',
   valori troppo bassi rallentano l'apprendimento.

4. **LR Schedule**: gli schedule confrontano stabilita' e velocita'
   di adattamento.
