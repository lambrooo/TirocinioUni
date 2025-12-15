
# RIEPILOGO ESPERIMENTI TESI

## Esperimenti Eseguiti

### Esperimento 1: Confronto EFE Modes
Confronta il comportamento dell'agente Active Inference con diverse modalità 
di calcolo dell'Expected Free Energy:
- **full**: usa entrambi i termini (epistemico + pragmatico)
- **epistemic_only**: solo minimizzazione dell'entropia (curiosità)
- **pragmatic_only**: solo massimizzazione dell'utilità (obiettivi)

File: exp1_efe_modes.png, exp1_summary.csv

### Esperimento 2: Impatto Probabilità Attacchi
Varia la probabilità di attacchi informatici da 0% a 10% per studiare
la robustezza dell'agente.

File: exp2_attack_impact.png

### Esperimento 3: Agente Statico vs Intelligente
Confronta l'agente tradizionale (basato su regole PID-like) con 
l'agente Active Inference in diverse configurazioni.

File: exp3_static_vs_intelligent.png, exp3_summary.csv

### Esperimento 4: Impatto Cyber Defense
Valuta l'efficacia del layer di cyber defense nel rilevare attacchi
e proteggere il sistema.

File: exp4_defense_impact.png

## Parametri Chiave da Discutere nella Tesi

1. **EFE Mode**: Come il bilanciamento tra epistemico e pragmatico
   influenza le decisioni dell'agente

2. **Attack Probability**: Robustezza del sistema sotto stress

3. **Cyber Defense**: Trade-off tra costo della difesa e protezione

4. **Agent Type**: Vantaggi dell'approccio Active Inference vs regole statiche
