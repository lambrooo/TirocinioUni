# Analisi delle Performance dell'Agente Statico

**Autore:** Leonardo Lambruschi 
**Data:** 6 Novembre 2025

---

### 1. Introduzione

Questo documento analizza i risultati ottenuti dalla simulazione eseguita con l'agente di controllo statico (controllore proporzionale). L'obiettivo è valutare le performance di questo agente, identificarne le carenze e motivare la necessità di sviluppare un agente intelligente basato su machine learning o Active Inference.

---

### 2. Osservazioni Qualitative (dai Grafici)

L'analisi visiva dei grafici generati dalla simulazione ha rivelato un comportamento problematico e sub-ottimale del sistema:

-   **Temperatura del Motore:** La temperatura del motore (`true_motor_temp`) mostra una tendenza a superare rapidamente la soglia di sicurezza (`T_safe = 80°C`) e a rimanere in uno stato di surriscaldamento per la quasi totalità della simulazione.
-   **Performance del Sistema:** Il grafico della performance, dopo un breve picco iniziale, crolla a valori bassi e spesso negativi. Questo indica che la penalità dovuta al surriscaldamento è quasi costantemente attiva, vanificando la produttività generata dal carico di lavoro.
-   **Impatto degli Attacchi:** Le aree evidenziate in rosso (periodi di attacco) mostrano un'ulteriore destabilizzazione del sistema, con picchi anomali nelle letture dei sensori che l'agente non è in grado di gestire efficacemente, portando a comandi erratici e a un peggioramento della performance.

---

### 3. Analisi Quantitativa (dai Dati di Log)

L'analisi numerica del file `simulation_log.csv` conferma in modo inequivocabile le osservazioni qualitative:

-   **Temperatura Media Eccessiva:** La temperatura media del motore si è attestata a **145.71°C**, quasi il doppio della soglia di sicurezza.
-   **Stato di Surriscaldamento Cronico:** Il sistema ha operato al di sopra della temperatura di sicurezza per il **97.70%** del tempo.
-   **Performance Media Negativa:** La performance media totale è risultata pari a **-84.56**. Questo dato è allarmante, poiché significa che, in media, il sistema è più "costoso" da mantenere (a causa delle penalità) di quanto sia "produttivo".
-   **Crollo della Performance:** Il confronto tra gli stati operativi è netto: la performance media in condizioni normali era positiva (`17.48`), mentre è crollata a `-86.96` durante il surriscaldamento.

---

### 4. Diagnosi del Problema: La "Miopia" dell'Agente Statico

La causa di queste scarse performance non è un errore nel codice, ma una **carenza fondamentale nella logica dell'agente**.

L'agente attuale è un semplice **controllore proporzionale**, ed è caratterizzato da una logica "miope" e puramente **reattiva**:
1.  **Non comprende le relazioni causa-effetto:** L'agente non sa che aumentare il `carico` per raggiungere il setpoint di produttività è la causa principale dell'aumento di `temperatura`.
2.  **Agisce in modo conflittuale:** Tenta simultaneamente di aumentare il carico (per produrre di più) e di raffreddare il sistema (per abbassare la temperatura), senza capire che la prima azione vanifica la seconda. È come accelerare e frenare allo stesso tempo.
3.  **Manca di strategia a lungo termine:** Il suo unico obiettivo è minimizzare l'errore *istantaneo* rispetto ai setpoint, anche se questo comporta un peggioramento della performance complessiva a lungo termine.

---

### 5. Conclusione e Prospettive Future

L'analisi ha dimostrato che un agente statico semplice non è in grado di gestire in modo efficiente le dinamiche complesse e interconnesse del nostro ambiente simulato. Sebbene controllori statici più avanzati (come i PID) potrebbero migliorare la stabilità, anch'essi faticano ad adattarsi a condizioni impreviste o a ottimizzare performance in sistemi non-lineari.

Questa conclusione fornisce una solida motivazione per la **Fase 3** del progetto: lo sviluppo di un **agente intelligente** (basato su Active Inference o altro machine learning). Un tale agente avrebbe il potenziale per:
-   **Apprendere** il modello interno del sistema (la relazione tra carico e temperatura).
-   **Sviluppare strategie predittive** e a lungo termine.
-   **Massimizzare la performance** in modo più robusto, bilanciando produttività e sicurezza in modo proattivo anziché reattivo.
