# Note per la Tesi

## 1. Il Paradosso della Reattività Statica
Abbiamo osservato un fenomeno interessante: rendendo l'agente statico (PID) "più intelligente" (cioè dandogli accesso agli allarmi del sistema di difesa informatica), la sua efficienza complessiva è **calata** invece di aumentare.
- **Motivo**: L'agente statico segue regole rigide ("Se Allarme o Sospetto -> Verifica"). Questo lo porta a spendere budget in verifiche costose (40.0 unità + downtime) anche per falsi positivi o attacchi minori gestibili, drenando le risorse economiche.
- **Conclusione**: L'accesso all'informazione (allarme) senza la capacità di valutarne il *valore economico* (Active Inference) può essere controproducente.

## 2. Realismo del Fallimento (Budget Negativo)
La modifica alla simulazione per permettere il budget negativo è cruciale per una comparazione onesta.
- Prima, il budget bloccato a 0 nascondeva l'entità del disastro.
- Ora, il budget negativo (es. -2000) quantifica esattamente il "costo della rigidità".
- Questo evita l'obiezione metodologica secondo cui l'agente statico sarebbe stato programmato per fallire; al contrario, è stato programmato per reagire, ma fallisce economicamente perché non sa adattarsi al contesto.

## 3. Vantaggio dell'Active Inference
L'agente Active Inference vince non perché "sa" dell'attacco (lo sa anche lo statico ora), ma perché calcola l'**Expected Free Energy (EFE)**.
- Valuta se l'azione di verifica (riduzione dell'incertezza epistemica) vale il costo in termini di perdita di utilità (budget/produzione).
- Decide dinamicamente se "fidarsi e rischiare" o "pagare per verificare", ottimizzando la sopravvivenza a lungo termine.
