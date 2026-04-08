# Active Inference for Cyber-Physical Systems Security

Simulation framework for studying resilient control of an industrial motor under sensor attacks, uncertainty, and economic constraints.

## Research Focus

The project investigates whether an Active Inference agent that updates its transition model online can outperform a comparable agent with a fixed model. The comparison is centered on:

- industrial motor safety under thermal stress,
- spoofing and anomaly conditions on IIoT sensors,
- trade-off between production, verification cost, and long-term budget,
- static versus adaptive decision-making.

## Implemented Components

### Agents

| Agent | Description | Learning |
|-------|-------------|----------|
| `Agent` | Proportional static controller | No |
| `ActiveInferenceAgent` | Active Inference agent with fixed B matrix | No |
| `AdaptiveActiveInferenceAgent` | Active Inference agent with online B-matrix updates | Yes |
| `QLearningAgent` | Model-free reinforcement learning baseline | Yes |
| `DoubleQLearningAgent` | Double Q-Learning baseline | Yes |

### Environment

The simulation includes:

- thermal dynamics of an industrial motor,
- load-dependent heating and actuation,
- noisy and intermittently updated sensors,
- anomaly and attack injection on sensor readings,
- optional cyber-defense layer,
- economic budget with operating costs, verification costs, and production revenue.

### Analysis Tools

- Streamlit dashboard for interactive experiments,
- batch scripts for repeated runs,
- statistical analysis utilities,
- B-matrix visualization for adaptive Active Inference,
- curriculum-learning utilities for staged evaluation.

## Repository Structure

```text
├── pytorch_simulation/
│   ├── active_inference_agent.py
│   ├── simulation.py
│   ├── dashboard.py
│   ├── qlearning_agent.py
│   ├── b_matrix_viz.py
│   ├── statistical_analysis.py
│   ├── curriculum_learning.py
│   ├── test_agent_logic.py
│   ├── test_markov_chain.py
│   └── requirements.txt
├── Docs/
│   ├── TECHNICAL_DOCUMENTATION.md
│   ├── progetto.md
│   ├── specs.md
│   ├── Analisi.md
│   └── EXPLAIN.md
├── run_active_inference.py
├── run_learning_experiments.py
├── run_thesis_experiments.py
└── test_all_features.py
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r pytorch_simulation/requirements.txt
```

## Execution

Run the dashboard:

```bash
python3 -m streamlit run pytorch_simulation/dashboard.py
```

Run the core validation scripts:

```bash
python3 pytorch_simulation/test_agent_logic.py
python3 pytorch_simulation/test_markov_chain.py
python3 test_all_features.py
```

## Documentation

The main technical references included in the repository are:

- `Docs/TECHNICAL_DOCUMENTATION.md` for the system overview and implemented modules,
- `Docs/progetto.md` for the project report,
- `Docs/specs.md` for the technical specification,
- `Docs/Analisi.md` for the analysis of the static controller,
- `Docs/EXPLAIN.md` for the conceptual explanation of epistemic and pragmatic actions.

## Notes

Local experiment outputs such as `wandb/`, generated figures, and temporary test artifacts are not part of the core source code and can be excluded from version control or submission packages when needed.
