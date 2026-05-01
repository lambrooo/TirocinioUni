# Documentazione Tecnica Completa

**Progetto**: Active Inference per Sicurezza IIoT  
**Autore**: Leonardo Lambruschi  
**Ultimo aggiornamento**: Aprile 2026

---

## Indice

1. [Panoramica del Sistema](#1-panoramica-del-sistema)
2. [Feature A: Save/Load Model](#2-feature-a-saveload-model)
3. [Feature B: B Matrix Visualization](#3-feature-b-b-matrix-visualization)
4. [Feature C: Q-Learning Agent](#4-feature-c-q-learning-agent)
5. [Feature D: Statistical Analysis](#5-feature-d-statistical-analysis)
6. [Feature E: Curriculum Learning](#6-feature-e-curriculum-learning)
7. [Risultati Sperimentali](#7-risultati-sperimentali)
8. [Guida all'Uso](#8-guida-alluso)

---

## 1. Panoramica del Sistema

### 1.1 Architettura

Il sistema simula un **motore industriale** con sensori che possono essere attaccati da cyber-attacchi. L'agente deve:
- Mantenere la temperatura del motore sotto la soglia di sicurezza (80°C)
- Massimizzare la produttività (load alto)
- Gestire un budget economico
- Rilevare e mitigare attacchi ai sensori

### 1.2 Tipi di Agenti Implementati

| Agente | Classe | Apprendimento | Modello |
|--------|--------|---------------|---------|
| PID Statico | `Agent` | No | Proporzionale |
| Active Inference Statico | `ActiveInferenceAgent` | No | Generativo (B fissa) |
| Active Inference Learning | `AdaptiveActiveInferenceAgent` | Sì (B matrix) | Generativo (B adattiva) |
| Q-Learning | `QLearningAgent` | Sì (Q-table) | Model-free |

### 1.3 Differenza Chiave: Static vs Learning

**Agente Statico**:
```
B matrix definita a priori → Può non corrispondere alla vera dinamica
```

**Agente Learning**:
```
B matrix inizializzata → Aggiornata ad ogni step con l'esperienza osservata
```

L'aggiornamento della B matrix segue uno schema a conteggi probabilistici. Le credenze precedenti e correnti non vengono convertite in stati hard; viene invece usata un'assegnazione soft, coerente con la natura probabilistica dell'Active Inference:

```python
expected_transition = np.outer(curr_qs[f], prev_qs[f])
B_counts[f][:, :, action] += expected_transition * learning_rate * 10
B[f][:, state, action] = normalize(B_counts[f][:, state, action])
```

Questa scelta ha tre vantaggi:

1. mantiene la B-matrix come matrice stocastica valida, con colonne normalizzate;
2. permette di imparare anche quando lo stato non e' osservato con certezza;
3. rende interpretabile il cambiamento del modello tramite divergenza dalla B iniziale, prediction error e learning rate.

---

## 2. Feature A: Save/Load Model

### 2.1 Scopo

Permettere il **transfer learning**: addestrare un agente in uno scenario e deployarlo in un altro.

### 2.2 Implementazione Tecnica

**File**: `pytorch_simulation/active_inference_agent.py`

```python
def save_model(self, filepath):
    """Salva la B matrix appresa e lo stato dell'agente."""
    np.savez(filepath,
        B_0=self.B[0], B_1=self.B[1], ...,  # Matrici apprese
        B_counts_0=self.B_counts[0], ...,   # Conteggi per continuare
        total_updates=self.total_updates,
        learning_rate=self.learning_rate,
        efe_mode=self.efe_mode,
        ...
    )

def load_model(self, filepath, continue_learning=True):
    """Carica un modello salvato."""
    data = np.load(filepath)
    self.B = [data['B_0'], data['B_1'], ...]
    
    if continue_learning:
        self.B_counts = [data['B_counts_0'], ...]  # Per continuare
    else:
        # Modello "congelato" per deployment
        self.B_counts = [B * 10 for B in self.B]
```

### 2.3 Casi d'Uso

1. **Transfer Learning**: Addestra su simulazione lunga → Deploya su nuova istanza
2. **Checkpoint**: Salva progresso durante esperimenti lunghi
3. **Deployment**: Modello pre-addestrato per produzione (senza learning online)

### 2.4 Formato File

File `.npz` (NumPy compressed) contenente:
- 5 matrici B (una per fattore)
- 5 matrici di conteggi
- Metadati (step, learning rate, schedule, ecc.)

---

## 3. Feature B: B Matrix Visualization

### 3.1 Scopo

Visualizzare come l'agente **modifica il suo modello interno** durante l'apprendimento.

### 3.2 Implementazione Tecnica

**File**: `pytorch_simulation/b_matrix_viz.py`

#### 3.2.1 Snapshot della B Matrix

```python
def get_b_matrix_snapshot(self):
    """Restituisce dati per visualizzazione."""
    snapshot = {}
    for f in range(5):
        snapshot[factor_name] = {
            'matrix': self.B[f].copy(),
            'initial_matrix': self.B_initial[f].copy(),
            'change': self.B[f] - self.B_initial[f],
            'actions': ['Cool', 'Maintain', 'Heat'],
            'states': ['Low', 'Optimal', 'High'],
        }
    return snapshot
```

#### 3.2.2 Heatmap di Confronto

```python
def plot_b_matrix_heatmap(agent, factor_idx=0):
    """Genera heatmap Initial vs Learned vs Difference."""
    # Usa seaborn per heatmap con annotazioni
    sns.heatmap(B_init, ...)  # Matrice iniziale
    sns.heatmap(B_learned, ...)  # Matrice appresa
    sns.heatmap(B_diff, cmap='RdBu_r', center=0)  # Differenza
```

#### 3.2.3 Registrazione Evoluzione

```python
class BMatrixRecorder:
    """Registra snapshot durante la simulazione."""
    
    def __init__(self, agent, interval=100):
        self.history = []
    
    def record(self, step):
        if step % self.interval == 0:
            self.history.append((step, [b.copy() for b in agent.B]))
    
    def create_animation(self):
        """Crea animazione GIF/MP4 dell'evoluzione."""
```

### 3.3 Output Generati

| File | Descrizione |
|------|-------------|
| `b_matrix_summary.png/pdf` | Riassunto cambiamenti per tutti i fattori |
| `b_matrix_temperature.png/pdf` | Dettaglio fattore temperatura |
| `b_matrix_temp_health.png/pdf` | Fattore salute sensore temperatura |
| `evolution.png` | Evoluzione elementi B nel tempo |

---

## 4. Feature C: Q-Learning Agent

### 4.1 Scopo

Fornire un **baseline di confronto** con un algoritmo di Reinforcement Learning classico (model-free).

### 4.2 Implementazione Tecnica

**File**: `pytorch_simulation/qlearning_agent.py`

#### 4.2.1 Spazio Stati/Azioni

```python
# Discretizzazione (stessa dell'Active Inference)
States: Temperature(3) × Motor(2) × Load(2) × Cyber(2) = 24 stati
Actions: Thermal(3) × Load(3) × Verify(2) = 18 azioni
```

#### 4.2.2 Algoritmo Q-Learning

```python
def update(self, state, action, reward, next_state):
    # TD Update
    target = reward + gamma * max(Q[next_state])
    td_error = target - Q[state][action]
    Q[state][action] += alpha * td_error
    
    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

#### 4.2.3 Funzione Reward

```python
def compute_reward(temp_state, motor_state, load_state, cyber_state):
    reward = 0
    if temp_state == 'Optimal': reward += 2.0
    if motor_state == 'Safe': reward += 3.0
    if motor_state == 'Overheating': reward -= 5.0  # Penalità forte
    if load_state == 'High': reward += 1.5
    if verify_action: reward -= 2.0  # Costo opportunità
    return reward
```

#### 4.2.4 Double Q-Learning

Implementato anche `DoubleQLearningAgent` per ridurre l'overestimation bias:

```python
# Invece di: target = r + γ * max_a Q(s', a)
# Usa: target = r + γ * Q2(s', argmax_a Q1(s', a))
```

### 4.3 Differenze con Active Inference

| Aspetto | Active Inference | Q-Learning |
|---------|-----------------|------------|
| Modello | Generativo (B, A, C, D) | Nessuno (model-free) |
| Apprendimento | Transizioni P(s'|s,a) | Valori Q(s,a) |
| Esplorazione | Epistemic value | Epsilon-greedy |
| Sample efficiency | Alta | Bassa |

---

## 5. Feature D: Statistical Analysis

### 5.1 Scopo

Fornire **rigore statistico** per la tesi con test di significatività e effect size.

### 5.2 Implementazione Tecnica

**File**: `pytorch_simulation/statistical_analysis.py`

#### 5.2.1 T-Test Indipendente

```python
def independent_ttest(group1, group2):
    # Test di Levene per omogeneità varianze
    levene_p = stats.levene(group1, group2)[1]
    equal_var = levene_p > 0.05
    
    # T-test (Welch se varianze disomogenee)
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    # Effect size (Cohen's d)
    pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size_interpretation': interpret(cohens_d),  # small/medium/large
    }
```

#### 5.2.2 ANOVA con Post-Hoc

```python
def one_way_anova(groups):
    # ANOVA
    F, p = stats.f_oneway(*groups.values())
    
    # Effect size (eta-squared)
    eta_squared = SS_between / SS_total
    
    # Post-hoc con correzione Bonferroni
    if p < 0.05:
        bonferroni_alpha = 0.05 / n_comparisons
        for pair in combinations(groups, 2):
            pairwise_ttest(...)
```

#### 5.2.3 Intervalli di Confidenza Bootstrap

```python
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, confidence=0.95):
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(sample))
    
    return np.percentile(bootstrap_stats, [2.5, 97.5])
```

#### 5.2.4 Generazione Tabelle LaTeX

```python
def generate_latex_stats_table(analysis, metric):
    latex = r"\begin{table}[htbp]..."
    for group in groups:
        latex += f"{group} & {mean:.2f} & {std:.2f} & [{ci_low:.2f}, {ci_high:.2f}] \\\\"
    latex += r"\end{table}"
    return latex
```

### 5.3 Interpretazione Effect Size (Cohen's d)

| |d| | Interpretazione |
|-----|-----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| ≥ 0.8 | Large |

---

## 6. Feature E: Curriculum Learning

### 6.1 Scopo

Aumentare **gradualmente la difficoltà** degli attacchi per:
1. Permettere all'agente di imparare progressivamente
2. Simulare scenari realistici (threat landscape evolve)
3. Migliorare la robustezza finale

### 6.2 Implementazione Tecnica

**File**: `pytorch_simulation/curriculum_learning.py`

#### 6.2.1 Stadi del Curriculum

```python
STAGES = {
    'easy': CurriculumStage(
        attack_prob=0.005,
        attack_types=['bias'],
        duration_range=(5, 20),
    ),
    'medium': CurriculumStage(
        attack_prob=0.015,
        attack_types=['bias', 'outlier'],
        duration_range=(10, 50),
    ),
    'hard': CurriculumStage(
        attack_prob=0.025,
        attack_types=['bias', 'outlier', 'spoofing'],
        duration_range=(20, 80),
    ),
    'adversarial': CurriculumStage(
        attack_prob=0.04,
        attack_types=['bias', 'outlier', 'spoofing', 'dos'],
        duration_range=(30, 100),
    ),
}
```

#### 6.2.2 Strategie di Progressione

```python
class CurriculumScheduler:
    def should_progress(self, step, efficiency, prediction_error):
        if self.strategy == 'step_based':
            return steps_in_stage >= threshold
        
        elif self.strategy == 'performance_based':
            return rolling_avg(efficiency) >= min_efficiency
        
        elif self.strategy == 'loss_based':
            return prediction_error <= max_error
```

#### 6.2.3 Integrazione con Simulazione

```python
def run_curriculum_simulation(agent, n_steps):
    curriculum = CurriculumScheduler(strategy='step_based')
    
    for step in range(n_steps):
        # Genera attacchi secondo curriculum
        config = curriculum.get_attack_config()
        if random() < config['attack_prob']:
            attack_type = choice(config['attack_types'])
            inject_attack(...)
        
        # Step agente
        agent.step(...)
        
        # Aggiorna curriculum
        curriculum.step(step, efficiency=efficiency)
```

### 6.3 Tipi di Attacco

| Tipo | Effetto | Difficoltà |
|------|---------|------------|
| `bias` | Aggiunge offset costante al sensore | Bassa |
| `outlier` | Valore estremo momentaneo | Media |
| `spoofing` | Valore falso controllato dall'attaccante | Alta |
| `dos` | Sensore non risponde (ultimo valore) | Alta |

---

## 7. Risultati Sperimentali

I risultati finali inclusi nel repository sono nella run
`learning_experiments/run_20260430_180334`, generata con seed base fissato.

### 7.1 Confronto Short-term (500 step, 10 run)

| Configurazione | Budget finale medio | Efficienza media | Step medi in overheating |
|----------------|--------------------:|-----------------:|-------------------------:|
| Static Fixed Model | -149.52 | 0.94 | 8.0 |
| Learning LR=0.01 | 2719.98 | 6.99 | 18.6 |
| Learning LR=0.05 | 4310.32 | 10.38 | 22.2 |

### 7.2 Confronto Long-term (5000 step, 10 run)

| Configurazione | Budget finale medio | Efficienza prima metà | Efficienza seconda metà |
|----------------|--------------------:|----------------------:|------------------------:|
| Static Fixed Model | -12277.12 | 0.55 | 0.31 |
| Learning constant | 1926.66 | 4.03 | 3.13 |
| Learning decay | 2411.27 | 4.01 | 3.32 |

### 7.3 Learning-rate schedule

| Schedule | Budget finale medio | Efficienza media seconda metà |
|----------|--------------------:|------------------------------:|
| Constant | 2880.04 | 3.33 |
| Exponential Decay | 2820.56 | 3.25 |
| Adaptive | 3468.31 | 3.37 |

Questi risultati sostengono il focus della tesi: a parità di struttura Active
Inference, l'aggiornamento online della matrice B produce un vantaggio netto
rispetto al modello fisso, soprattutto nelle run lunghe.

---

## 8. Guida all'Uso

### 8.1 Eseguire i Test

```bash
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_agent_logic.py
WANDB_MODE=disabled .venv/bin/python pytorch_simulation/test_markov_chain.py
WANDB_MODE=disabled .venv/bin/python test_all_features.py
```

### 8.2 Eseguire Esperimenti per Tesi

```bash
# Smoke test
WANDB_MODE=disabled .venv/bin/python run_learning_experiments.py --quick --exp 1 3 --seed 123

# Campagna finale consigliata
WANDB_MODE=disabled .venv/bin/python run_learning_experiments.py --n_runs 10 --seed 42
```

### 8.3 Generare Figure per Tesi

```bash
WANDB_MODE=disabled .venv/bin/python generate_thesis_figures.py --full --seed 42
```

Il confronto principale e' tra `ActiveInferenceAgent` con B-matrix fissa
e `AdaptiveActiveInferenceAgent` con aggiornamento online della B-matrix.
Q-Learning, Curriculum Learning e visualizzazioni della B-matrix completano
l'analisi come baseline e strumenti diagnostici.

### 8.4 Salvare/Caricare Modello

```python
from pytorch_simulation.active_inference_agent import AdaptiveActiveInferenceAgent

# Addestra
agent = AdaptiveActiveInferenceAgent(learning_rate=0.02)
for step in range(5000):
    agent.step(temp, motor, load, attack)

# Salva
agent.save_model('models/trained_agent')

# Carica per transfer learning
new_agent = AdaptiveActiveInferenceAgent()
new_agent.load_model('models/trained_agent', continue_learning=True)

# Carica per deployment (no learning)
deploy_agent = AdaptiveActiveInferenceAgent()
deploy_agent.load_model('models/trained_agent', continue_learning=False)
```

### 8.5 Analisi Statistica

```python
from pytorch_simulation.statistical_analysis import *

# Confronto due gruppi
result = independent_ttest(static_results, learning_results)
print(f"p = {result['p_value']:.4f}, d = {result['cohens_d']:.2f}")

# Analisi completa DataFrame
analysis = analyze_experiment_results(df)
print_analysis_report(analysis)

# Genera tabella LaTeX
latex = generate_latex_stats_table(analysis, 'final_budget')
```

---

## Appendice: File Principali

| File | Descrizione |
|------|-------------|
| `pytorch_simulation/active_inference_agent.py` | Active Inference statico, Adaptive Active Inference, save/load e snapshot B-matrix |
| `pytorch_simulation/simulation.py` | Ambiente, sensori, attuatori, cyber-defense layer e loop sperimentale |
| `pytorch_simulation/qlearning_agent.py` | Baseline Q-Learning e Double Q-Learning |
| `pytorch_simulation/b_matrix_viz.py` | Visualizzazione e analisi della B-matrix |
| `pytorch_simulation/statistical_analysis.py` | Test statistici, effect size e tabelle LaTeX |
| `pytorch_simulation/curriculum_learning.py` | Scenari progressivi a difficolta' crescente |
| `run_learning_experiments.py` | Campagna principale Static AI vs Learning AI |
| `generate_thesis_figures.py` | Generazione di figure e tabelle finali |
| `test_all_features.py` | Test integrato delle feature principali |
