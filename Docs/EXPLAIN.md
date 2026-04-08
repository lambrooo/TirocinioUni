# Sistema Active Inference - Spiegazione Completa

## 1. Il Sistema di Cyber Defense

### Cosa fa?

Il sistema di cyber defense è una **difesa esterna** che rileva automaticamente certi tipi di attacchi:

```python
# Nel codice (simulation.py linee 278-287):
if env.cyber_defense_active:
    for name, sensor in sensors.items():
        if sensor.anomaly and sensor.anomaly['type'] == 'dos':
            attack_detected_this_step = 1
            sensor.anomaly = None  # Rimuove l'attacco
        if sensor.anomaly and sensor.anomaly['type'] == 'spoofing':
            attack_detected_this_step = 1
            sensor.anomaly = None  # Rimuove l'attacco
```

### Caratteristiche

| Aspetto | Descrizione |
|---------|-------------|
| **Costo** | Fisso: 2.0 per step (sempre, indipendentemente dagli attacchi) |
| **Cosa rileva** | Solo `dos` e `spoofing` (non `bias` o `outlier`) |
| **Come funziona** | Signature-based: confronta patterns noti |
| **Effetto sull'agente** | Setta `attack_detected=1` che l'agente riceve come osservazione |

### È un'azione epistemica?

**NO.** La cyber defense è passiva e automatica. Non è una scelta dell'agente.

L'agente riceve solo il **risultato** della difesa (alert sì/no) come osservazione nella modalità Cyber:
- `obs_cyber = 0` → Tutto ok
- `obs_cyber = 1` → Alert! Possibile attacco

---

## 2. Azioni Epistemiche vs Pragmatiche

### Azione Pragmatica

> *"Fai qualcosa per raggiungere il tuo obiettivo"*

| Controllo | Azioni | Scopo |
|-----------|--------|-------|
| **Thermal** | Cool, Maintain, Heat | Mantenere temperatura ottimale |
| **Load** | Decrease, Maintain, Increase | Massimizzare produzione |

**Caratteristiche:**
- Hanno effetto diretto sul mondo fisico
- Influenzano il budget (più load = più guadagno)
- Sono guidate dalla matrice **C** (preferenze)

### Azione Epistemica

> *"Controlla se quello che credi è vero"*

| Controllo | Azioni | Scopo |
|-----------|--------|-------|
| **Verify** | Wait, Verify Temp, Verify Motor | Ridurre incertezza sui sensori |

**Caratteristiche:**
- NON cambiano il mondo fisico
- Hanno un **costo** (budget - 40.0) e causano **downtime** (3 step a produzione ridotta)
- Riducono l'**entropia** (incertezza) sullo stato dei sensori
- Dopo la verifica: `qs[sensor_health] = [1.0, 0.0]` → certezza che funziona

### Come Vengono Scelte

```
Per ogni azione possibile:
    G(a) = Pragmatic_Value - Epistemic_Value - Opportunity_Cost
    
Dove:
    Pragmatic_Value = quanto l'azione porta verso le preferenze
    Epistemic_Value = quanto l'azione riduce l'entropia
    Opportunity_Cost = 2.0 (solo per Verify, rappresenta produzione persa)
```

L'agente sceglie l'azione con **G più alto**.

---

## 3. Agente Statico vs Intelligente

### Agente Statico

```python
class Agent:
    def get_commands(self, temp_reading, load_reading):
        temp_error = self.temp_setpoint - temp_reading
        load_error = self.load_setpoint - load_reading
        return self.p_gain * temp_error, self.p_gain * load_error
```

| Aspetto | Comportamento |
|---------|---------------|
| **Tipo di controllo** | PID classico (Proporzionale) |
| **Gestisce incertezza?** | NO - crede ciecamente ai sensori |
| **Azioni epistemiche?** | Solo euristiche fisse (se motor_temp > 90 → verifica) |
| **Adattivo?** | NO - stessi parametri sempre |

### Agente Intelligente (Active Inference)

```python
class ActiveInferenceAgent:
    def step(self, temp_reading, motor_temp_reading, load_reading, attack_detected):
        # 1. Percezione: Aggiorna credenze dato osservazioni
        obs = self.discretize_observation(...)
        self.infer_state(obs)  # Minimizza VFE
        
        # 2. Azione: Scegli basandosi su EFE
        actions = self.plan_action()  # Minimizza EFE
        
        return self.map_action_to_control(actions)
```

| Aspetto | Comportamento |
|---------|---------------|
| **Tipo di controllo** | Bayesiano (credenze probabilistiche) |
| **Gestisce incertezza?** | SÌ - modella affordabilità dei sensori |
| **Azioni epistemiche?** | Scelte ottimamente in base a EFE |
| **Adattivo?** | Con learning_rate (versione futura) |

### Quando Verifica l'Agente Intelligente?

L'agente verifica quando:

1. **Alta incertezza** sul sensore: `qs[sensor_health] ≈ [0.5, 0.5]`
2. **Il guadagno informativo supera il costo opportunità**:
   ```
   Riduzione_Entropia (0.69 → 0) > Opportunity_Cost (2.0)
   ```

Se l'agente è già sicuro (`qs = [0.95, 0.05]`), non verifica perché non ne vale la pena.

---

## 4. Matrice Stocastica e Modello Generativo

### Cos'è un Modello Generativo?

È un modello probabilistico che descrive **come il mondo genera le osservazioni**:

```
P(osservazioni, stati) = P(osservazioni | stati) × P(stati)
                       =        A             ×      D
```

### Le Matrici del Modello

| Matrice | Nome | Cosa Modella | Stocastica? |
|---------|------|--------------|-------------|
| **A** | Likelihood | P(o \| s) - Come gli stati causano osservazioni | Sì (colonne sommano a 1) |
| **B** | Transition | P(s' \| s, a) - Come le azioni cambiano gli stati | Sì (colonne sommano a 1) |
| **C** | Preferences | log P(o) - Quali osservazioni preferisci | No (log-probabilità) |
| **D** | Prior | P(s₀) - Credenza iniziale sugli stati | Sì (somma a 1) |

### Cos'è una Matrice Stocastica?

Una matrice dove **ogni colonna somma a 1** (rappresenta una distribuzione di probabilità):

```
Esempio: A per temperatura (3 osservazioni × 3 stati)

         Stato_Low  Stato_Opt  Stato_High
Obs_Low    0.90       0.05       0.02
Obs_Opt    0.08       0.90       0.08
Obs_High   0.02       0.05       0.90
           ────       ────       ────
           1.00       1.00       1.00  ← Ogni colonna somma a 1
```

Significato: "Se lo stato vero è Optimal, ho 90% probabilità di osservare Optimal, 8% Low, 2% High"

### Varianza nel Modello Generativo

Il tuo collega probabilmente usa un modello **continuo** invece che discreto:

```
Modello Discreto (il tuo):
    P(o = "Low" | s = "Optimal") = 0.08

Modello Continuo (del collega):
    P(o | s) = N(μ = s, σ² = varianza)
    
    Esempio: Se stato = 50°C, osservazione ~ N(50, 4)
             → Osservi valori intorno a 50±2°C
```

| Aspetto | Tuo Modello | Modello Continuo |
|---------|-------------|------------------|
| **Stati** | Discreti (Low, Opt, High) | Continui (34.5°C, 67.2°C, ...) |
| **Incertezza** | Nelle probabilità della matrice | Nella varianza σ² |
| **Apprendimento** | Aggiorna celle della matrice | Aggiorna μ e σ² |

Il vantaggio del modello continuo è che può rappresentare valori infiniti, ma è più complesso matematicamente.

---

## 5. Riassunto Visivo

```
┌─────────────────────────────────────────────────────────────┐
│                         MONDO FISICO                        │
│  [Temperatura] [Motore] [Carico] [Sensori]                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     CYBER DEFENSE (Passiva)                 │
│  - Costo: 2.0/step (fisso)                                  │
│  - Rileva: DoS, Spoofing                                    │
│  - Output: Alert (sì/no) → passa all'agente                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    AGENTE ACTIVE INFERENCE                  │
├─────────────────────────────────────────────────────────────┤
│  PERCEZIONE (Minimizza VFE)                                 │
│  - Riceve: temp, motor_temp, load, cyber_alert              │
│  - Aggiorna: credenze qs[stato] usando matrici A, B, D      │
├─────────────────────────────────────────────────────────────┤
│  AZIONE (Minimizza EFE)                                     │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ PRAGMATICHE │    │ EPISTEMICHE │    │    EFE      │      │
│  │ Cool/Heat   │ +  │   Verify    │ →  │ G(a) = U-H  │      │
│  │ Load ±      │    │   sensor    │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │
│  Sceglie azione con G massimo                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                     [Azione sul mondo]
```
