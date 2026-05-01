# Active Inference Agent Explanation

This document explains the design and logic of the Active Inference Agent implemented for the industrial simulation.

## Overview

The agent uses **Active Inference** to control the system. It maintains a probabilistic internal model of the world (Generative Model) and selects actions through an **Expected Free Energy (EFE)** value proxy, which balances:
1.  **Pragmatic Value**: Achieving preferred states (e.g., Safe Motor Temperature).
2.  **Epistemic Value**: Reducing uncertainty (e.g., Verifying a sensor to know the true state).

## Generative Model

The agent models the world as a **Partially Observable Markov Decision Process (POMDP)** with the following components:

### 1. Hidden States ($s$)
The agent simplifies the continuous environment into discrete factors:
-   **Temperature**: Low (<40°C), Optimal (40-70°C), High (>70°C).
-   **Motor Temperature**: Safe (<80°C), Overheating (>=80°C).
-   **Load**: Low (<30), High (>=30).
-   **Temperature Sensor Health**: Reliable, Spoofed.
-   **Motor Sensor Health**: Reliable, Spoofed.

### 2. Observations ($o$)
The agent receives observations that correspond to these states.
-   **A-Matrix (Likelihood)**: Defines $P(o|s)$. Sensor-health factors modulate the observation model: reliable sensors keep the state-observation mapping close to identity, while spoofed sensors bias the readings toward attacker-controlled values.

### 3. Transitions ($B$)
The agent predicts how states evolve based on actions.
-   **B-Matrix (Transition)**: Defines $P(s_{t+1}|s_t, a_t)$.
    -   *Example*: Action "Cool" increases the probability of transitioning from "High Temp" to "Optimal Temp".
    -   *Example*: Action "Increase Load" increases the probability of "Overheating".

### 4. Preferences ($C$)
The agent has innate preferences for certain observations (Goals).
-   **C-Matrix**: Defines $P(o)$.
    -   **Motor**: Strong preference for "Safe" (Avoid Overheating).
    -   **Temp**: Preference for "Optimal".
    -   **Cyber**: Strong preference for "Safe".

## Agent Loop

### 1. Perception (Minimize VFE)
When the agent receives a continuous observation (e.g., Temp=85.0), it:
1.  **Discretizes** it (85.0 -> "High").
2.  **Minimizes Variational Free Energy (VFE)** to update its belief ($q(s)$) about the current state.
    -   It combines the **Likelihood** (what it sees) with the **Prior** (what it predicted from the last step).
    -   *Result*: A belief distribution, e.g., "I am 90% sure the Motor is Overheating".

### 2. Action Selection (EFE Value Proxy)
To decide what to do, the agent calculates an EFE-derived score for each possible action:
-   **score(action) = Pragmatic Value - Entropy - Opportunity Cost**
    -   **Pragmatic**: "Will this action lead to my preferred states?" (e.g., Cooling leads to Safe Motor).
    -   **Epistemic**: "Will this action reduce uncertainty?" (e.g., Verifying a sensor can make the sensor-health belief sharper).
-   The agent selects an action stochastically with Softmax, preferring actions with higher score.

### 3. Actuation
The selected discrete action (e.g., "Cool") is mapped to a continuous command (e.g., -5.0) and sent to the environment.

## Why this approach?
-   **Robustness**: The probabilistic nature handles noise and uncertainty better than rigid rules.
-   **Adaptive**: It naturally balances multiple goals (Production vs. Safety).
-   **Curiosity**: The Epistemic term drives it to verify sensors when uncertain, detecting attacks without explicit "if-then" rules.

## Learning vs. Pre-configuration

### Is this agent "trained"?
The fixed Active Inference agent is **pre-configured**.
-   In traditional **Deep Reinforcement Learning (RL)**, an agent starts with a random brain and must be "trained" for millions of steps to learn a policy.
-   In **Tabular Active Inference**, we can explicitly **design** the Generative Model (the A, B, C matrices) if we understand the environment physics. This allows the agent to be "smart" from Step 0, without a training phase.

### Can it learn?
Yes. The adaptive version implemented in `AdaptiveActiveInferenceAgent` enables **online learning** by treating the transition matrix as probability distributions updated through experience:
-   **B-matrix learning**: The agent tries an action, observes the result, and updates its transition probabilities ("Oh, I thought 'Cool' worked instantly, but it takes time").
-   The thesis comparison isolates this difference: same Active Inference structure, fixed B matrix versus learned B matrix.
