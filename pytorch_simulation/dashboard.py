"""
IIoT Security Testbed Dashboard v3.0
=====================================

Complete dashboard with:
- Active Inference agents (Static & Learning)
- Q-Learning agent with Q-table visualization
- B Matrix visualization and evolution
- Curriculum Learning experiments
- Advanced statistical analysis with LaTeX export
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import io
from dotenv import load_dotenv

# Import simulation components
from simulation import run_simulation, run_batch_simulation, set_global_seed
from active_inference_agent import ActiveInferenceAgent, AdaptiveActiveInferenceAgent
from qlearning_agent import QLearningAgent, DoubleQLearningAgent

# Import B Matrix visualization
try:
    from b_matrix_viz import (
        plot_b_matrix_heatmap,
        plot_all_factors_summary,
        BMatrixRecorder,
    )

    B_MATRIX_VIZ_AVAILABLE = True
except ImportError:
    B_MATRIX_VIZ_AVAILABLE = False

# Import Curriculum Learning
try:
    from curriculum_learning import (
        CurriculumScheduler,
        run_curriculum_simulation,
        run_curriculum_experiment,
    )

    CURRICULUM_AVAILABLE = True
except ImportError:
    CURRICULUM_AVAILABLE = False

# Import Statistical Analysis
try:
    from statistical_analysis import (
        independent_ttest,
        paired_ttest,
        one_way_anova,
        compute_confidence_interval,
        compute_cohens_d,
        bootstrap_ci,
        compare_survival_rates,
        analyze_experiment_results,
        generate_latex_stats_table,
        print_analysis_report,
    )

    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IIoT Security Testbed v3.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444e;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #fafafa !important;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8em;
        font-weight: bold;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fafafa !important;
    }
    
    /* Success/Warning boxes */
    .stSuccess {
        background-color: #1e3a2a;
        color: #d4edda;
        border-color: #c3e6cb;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        color: #fafafa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #41444e;
        border-bottom: 2px solid #3498db;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
         color: #fafafa !important;
    }
    
    .css-1aumxhk {
        color: #fafafa !important;
    }
    
    /* Agent type badges */
    .agent-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
    }
    .agent-static { background-color: #3498db; color: white; }
    .agent-learning { background-color: #2ecc71; color: white; }
    .agent-qlearning { background-color: #e74c3c; color: white; }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: #262730;
        border-radius: 8px;
    }

    /* Input labels */
    .stSlider label, .stSelectbox label, .stNumberInput label, .stTextArea label, .stTextInput label {
        color: #fafafa !important;
    }
    
    /* Markdown text */
    .stMarkdown p {
        color: #fafafa !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def resolve_run_seed(base_seed, run_id=0):
    """Use aligned seeds across agents when reproducibility is enabled."""
    return None if base_seed is None else int(base_seed + run_id)


def build_export_filename(prefix, seed_value=None):
    if seed_value is None:
        return f"{prefix}.csv"
    return f"{prefix}_seed_{seed_value}.csv"


def summarize_run(
    df,
    run_id,
    seed,
    agent_type,
    learning_rate_value=np.nan,
    lr_schedule_value="n/a",
):
    second_half_efficiency = df["performance"].iloc[len(df) // 2 :].mean()
    return {
        "run_id": run_id,
        "seed": seed,
        "agent_type": agent_type,
        "learning_rate": learning_rate_value,
        "lr_schedule": lr_schedule_value,
        "n_steps": int(df["step"].iloc[-1] + 1),
        "attack_prob": df["attack_prob"].iloc[0],
        "final_budget": df["budget"].iloc[-1],
        "avg_efficiency": df["performance"].mean(),
        "second_half_efficiency": second_half_efficiency,
        "survival": df["budget"].iloc[-1] > 0,
    }


def find_crossover_step(static_series, learning_series, steps):
    gap = learning_series - static_series
    positive_steps = steps[gap > 0]
    if len(positive_steps) == 0:
        return None
    return int(positive_steps.iloc[0])


def build_gap_dataframe(static_df, learning_df, performance_window=50):
    rolling_static = static_df["performance"].rolling(
        window=performance_window, min_periods=1
    ).mean()
    rolling_learning = learning_df["performance"].rolling(
        window=performance_window, min_periods=1
    ).mean()

    return pd.DataFrame(
        {
            "step": static_df["step"],
            "budget_gap": learning_df["budget"] - static_df["budget"],
            "efficiency_gap": rolling_learning - rolling_static,
        }
    )

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Configuration")
st.sidebar.caption("Dashboard for reproducible static-versus-learning comparisons")
st.sidebar.markdown("---")

# Simulation Parameters
st.sidebar.markdown("### Simulation")
n_steps = st.sidebar.slider(
    "Simulation Steps", min_value=100, max_value=5000, value=1000, step=100
)
attack_prob = st.sidebar.slider(
    "Attack Probability", min_value=0.0, max_value=0.1, value=0.02, step=0.005
)
defense_active = st.sidebar.toggle("Cyber Defense Active", value=True)
use_fixed_seed = st.sidebar.toggle(
    "Use fixed seed for reproducible thesis runs", value=False
)
base_seed = None
if use_fixed_seed:
    base_seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        help="Shared across agents so static and learning see the same stochastic scenario.",
    )

st.sidebar.markdown("---")

# Agent Configuration
st.sidebar.markdown("### Agent")
agent_type = st.sidebar.selectbox(
    "Agent Type",
    ["static", "intelligent", "intelligent_adaptive", "qlearning", "double_qlearning"],
    index=2,
    format_func=lambda x: {
        "static": "PID Controller",
        "intelligent": "Active Inference (Static)",
        "intelligent_adaptive": "Active Inference (Learning)",
        "qlearning": "Q-Learning",
        "double_qlearning": "Double Q-Learning",
    }[x],
)

# Show agent description
agent_descriptions = {
    "static": "Simple proportional controller. No learning capability.",
    "intelligent": "Active Inference with fixed B matrix (generative model).",
    "intelligent_adaptive": "Active Inference that learns B matrix online from observations.",
    "qlearning": "Model-free RL. Learns Q(s,a) values via temporal difference.",
    "double_qlearning": "Double Q-Learning to reduce overestimation bias.",
}
st.sidebar.caption(agent_descriptions[agent_type])

# Default values are defined before the agent-specific controls are rendered.
efe_mode = "full"
precision = 5.0
learning_rate = 0.01
lr_schedule = "constant"
epsilon = 0.1
discount_factor = 0.95

# Active Inference Settings
if agent_type in ["intelligent", "intelligent_adaptive"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Inference")

    efe_mode = st.sidebar.selectbox(
        "EFE Mode",
        ["full", "epistemic_only", "pragmatic_only"],
        index=0,
        help="full: balanced, epistemic: exploration, pragmatic: exploitation",
    )

    precision = st.sidebar.slider(
        "Precision (tau)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Low=exploration, High=exploitation",
    )

# Learning Settings
if agent_type == "intelligent_adaptive":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Learning")

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.001,
        max_value=0.1,
        value=0.02,
        step=0.005,
        format="%.3f",
    )

    lr_schedule = st.sidebar.selectbox(
        "LR Schedule",
        ["constant", "decay", "warmup", "cosine", "adaptive"],
        index=1,
    )
elif agent_type in ["qlearning", "double_qlearning"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Q-Learning")

    learning_rate = st.sidebar.slider(
        "Learning Rate (alpha)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
    )

    epsilon = st.sidebar.slider(
        "Epsilon (initial)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Initial exploration rate (decays over time)",
    )

    discount_factor = st.sidebar.slider(
        "Discount Factor (gamma)",
        min_value=0.8,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Future reward discounting",
    )
    lr_schedule = "constant"

st.sidebar.markdown("---")
st.sidebar.info(
    "**Tip:** Use longer simulations (2000+ steps) to see learning effects."
)
if use_fixed_seed:
    st.sidebar.success(f"Reproducible thesis mode enabled. Base seed: {base_seed}")
else:
    st.sidebar.caption("Exploratory mode: runs remain stochastic.")

# --- MAIN CONTENT ---
st.title("IIoT Security Testbed v3.0")
st.caption(
    "Reproducible comparison of static and learning controllers under IIoT sensor attacks"
)

# Tabs - thesis comparison is intentionally promoted near the front
tab1, tab4, tab2, tab3, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Single Run",
        "Thesis Comparison",
        "Batch Experiment",
        "Agent Comparison",
        "Q-Learning",
        "B Matrix",
        "Curriculum",
        "Statistics",
    ]
)

# --- TAB 1: SINGLE RUN ---
with tab1:
    st.header("Single Simulation")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_button = st.button(
            "Run Simulation", type="primary", use_container_width=True
        )
    with col_info:
        agent_label = {
            "static": "PID Controller",
            "intelligent": "AI Static",
            "intelligent_adaptive": "AI Learning",
            "qlearning": "Q-Learning",
            "double_qlearning": "Double Q-Learning",
        }[agent_type]
        st.info(
            f"**Agent:** {agent_label} | **Steps:** {n_steps} | **Attack Prob:** {attack_prob}"
        )

    if "single_run_data" not in st.session_state:
        st.session_state["single_run_data"] = None

    if run_button:
        with st.spinner(f"Running {n_steps} steps..."):
            try:
                # Keep a direct agent reference when post-run inspection is required.
                if agent_type == "qlearning":
                    custom_agent = QLearningAgent(
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        discount_factor=discount_factor,
                    )
                elif agent_type == "double_qlearning":
                    custom_agent = DoubleQLearningAgent(
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        discount_factor=discount_factor,
                    )
                elif agent_type == "intelligent_adaptive":
                    custom_agent = AdaptiveActiveInferenceAgent(
                        learning_rate=learning_rate,
                        lr_schedule=lr_schedule,
                        efe_mode=efe_mode,
                        precision=precision,
                    )
                else:
                    custom_agent = None

                single_seed = resolve_run_seed(base_seed, 0)
                df, _ = run_simulation(
                    n_steps=n_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=defense_active,
                    agent_type=agent_type,
                    efe_mode=efe_mode,
                    precision=precision,
                    learning_rate=learning_rate,
                    lr_schedule=lr_schedule,
                    seed=single_seed,
                    agent=custom_agent,
                )
                st.session_state["single_run_data"] = df
                st.success("Simulation completed!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state["single_run_data"] = None

    if st.session_state["single_run_data"] is not None:
        df = st.session_state["single_run_data"]

        # --- METRICS ROW ---
        col1, col2, col3, col4, col5 = st.columns(5)

        final_budget = df["budget"].iloc[-1]
        survived = final_budget > 0

        with col1:
            st.metric(
                "Final Budget",
                f"{final_budget:.0f}",
                delta=f"{final_budget - 1000:.0f}",
            )
        with col2:
            st.metric("Status", "SURVIVED" if survived else "FAILED")
        with col3:
            st.metric("Avg Efficiency", f"{df['performance'].mean():.2f}")
        with col4:
            st.metric("Attacks", int(df["is_under_attack"].sum()))
        with col5:
            st.metric("Detected", int(df["attack_detected"].sum()))

        if use_fixed_seed:
            st.caption(f"Reproducible run with seed {int(df['seed'].iloc[0])}.")
        else:
            st.caption("Exploratory run without fixed seed.")

        # --- MAIN PLOTS ---
        st.markdown("---")

        # Temperature & Motor Temp
        fig_temp = make_subplots(
            rows=1, cols=2, subplot_titles=("Temperature", "Motor Temperature")
        )

        fig_temp.add_trace(
            go.Scatter(
                x=df["step"], y=df["true_temp"], name="True", line=dict(color="#3498db")
            ),
            row=1,
            col=1,
        )
        fig_temp.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["sensed_temp"],
                name="Sensed",
                line=dict(color="#85c1e9", dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig_temp.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["true_motor_temp"],
                name="True",
                line=dict(color="#e74c3c"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig_temp.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["sensed_motor_temp"],
                name="Sensed",
                line=dict(color="#f5b7b1", dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig_temp.add_hline(
            y=80,
            line_dash="dash",
            line_color="red",
            annotation_text="T_safe",
            row=1,
            col=2,
        )

        fig_temp.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_temp, use_container_width=True)

        # Budget & Efficiency
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            fig_budget = go.Figure()
            fig_budget.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["budget"],
                    fill="tozeroy",
                    fillcolor="rgba(46, 204, 113, 0.3)",
                    line=dict(color="#2ecc71", width=2),
                )
            )
            fig_budget.add_hline(y=0, line_dash="dash", line_color="red")
            fig_budget.update_layout(
                title="Budget Over Time", height=300, margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_budget, use_container_width=True)

        with col_plot2:
            # Rolling efficiency
            window = min(50, len(df) // 10)
            df_plot = df.copy()
            df_plot["eff_smooth"] = (
                df["performance"].rolling(window=window, min_periods=1).mean()
            )

            fig_eff = go.Figure()
            fig_eff.add_trace(
                go.Scatter(
                    x=df_plot["step"],
                    y=df_plot["performance"],
                    name="Raw",
                    line=dict(color="#bdc3c7", width=1),
                )
            )
            fig_eff.add_trace(
                go.Scatter(
                    x=df_plot["step"],
                    y=df_plot["eff_smooth"],
                    name="Smoothed",
                    line=dict(color="#9b59b6", width=2),
                )
            )
            fig_eff.update_layout(
                title="Efficiency Over Time", height=300, margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig_eff, use_container_width=True)

        # Learning metrics (if available)
        if agent_type == "intelligent_adaptive" and "learning_rate" in df.columns:
            st.markdown("### Learning Metrics")
            col_lr1, col_lr2 = st.columns(2)

            with col_lr1:
                fig_lr = px.line(
                    df, x="step", y="learning_rate", title="Learning Rate Schedule"
                )
                fig_lr.update_traces(line_color="#e74c3c")
                fig_lr.update_layout(height=250)
                st.plotly_chart(fig_lr, use_container_width=True)

            with col_lr2:
                if "model_divergence" in df.columns:
                    fig_div = px.line(
                        df,
                        x="step",
                        y="model_divergence",
                        title="Model Divergence from Initial",
                    )
                    fig_div.update_traces(line_color="#3498db")
                    fig_div.update_layout(height=250)
                    st.plotly_chart(fig_div, use_container_width=True)

        # Download
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=build_export_filename(
                "simulation_run",
                int(df["seed"].iloc[0]) if pd.notna(df["seed"].iloc[0]) else None,
            ),
            mime="text/csv",
        )

# --- TAB 2: BATCH EXPERIMENT ---
with tab2:
    st.header("Batch Experiment")

    col1, col2 = st.columns(2)
    with col1:
        n_batch_runs = st.slider("Number of Runs", 5, 50, 10)
    with col2:
        n_batch_steps = st.number_input("Steps per Run", 500, 5000, 1000, 100)

    if st.button("Run Batch", type="primary"):
        with st.spinner(f"Running {n_batch_runs} simulations..."):
            df_batch = run_batch_simulation(
                n_runs=n_batch_runs,
                n_steps=n_batch_steps,
                attack_prob=attack_prob,
                cyber_defense_active=defense_active,
                agent_type=agent_type,
                efe_mode=efe_mode,
                precision=precision,
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
                seed=base_seed,
            )

        st.success("Batch complete!")
        st.session_state["batch_results"] = df_batch

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        survival_rate = (df_batch["final_budget"] > 0).mean() * 100

        with col1:
            st.metric("Survival Rate", f"{survival_rate:.0f}%")
        with col2:
            st.metric("Avg Budget", f"{df_batch['final_budget'].mean():.0f}")
        with col3:
            st.metric("Std Budget", f"{df_batch['final_budget'].std():.0f}")
        with col4:
            st.metric("Avg Efficiency", f"{df_batch['avg_efficiency'].mean():.2f}")

        # Plots
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig = px.histogram(
                df_batch, x="final_budget", nbins=15, title="Budget Distribution"
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        with col_p2:
            fig = px.histogram(
                df_batch, x="avg_efficiency", nbins=15, title="Efficiency Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(df_batch)

        st.download_button(
            "Download Batch CSV",
            data=df_batch.to_csv(index=False).encode("utf-8"),
            file_name=build_export_filename("batch_results", base_seed),
            mime="text/csv",
        )

# --- TAB 3: AGENT COMPARISON ---
with tab3:
    st.header("Agent Comparison")
    st.info(
        "Secondary comparison tab. The thesis narrative should prioritize the dedicated Static vs Learning tab."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_comp_runs = st.slider("Runs per Agent", 3, 20, 5, key="comp_runs")
    with col2:
        n_comp_steps = st.number_input("Steps", 500, 5000, 1000, 100, key="comp_steps")

    agents_to_compare = st.multiselect(
        "Agents to Compare",
        [
            "static",
            "intelligent",
            "intelligent_adaptive",
            "qlearning",
            "double_qlearning",
        ],
        default=["intelligent", "intelligent_adaptive"],
        format_func=lambda x: {
            "static": "PID Controller",
            "intelligent": "AI Static",
            "intelligent_adaptive": "AI Learning",
            "qlearning": "Q-Learning",
            "double_qlearning": "Double Q-Learning",
        }[x],
    )

    comp_learning_rate = st.number_input(
        "Adaptive Learning Rate",
        min_value=0.001,
        max_value=0.1,
        value=float(learning_rate if agent_type == "intelligent_adaptive" else 0.01),
        step=0.001,
        format="%.3f",
        help="Used when AI Learning is included in the comparison.",
    )
    comp_lr_schedule = st.selectbox(
        "Adaptive LR Schedule",
        ["constant", "decay", "warmup", "cosine", "adaptive"],
        index=["constant", "decay", "warmup", "cosine", "adaptive"].index(
            lr_schedule if agent_type == "intelligent_adaptive" else "decay"
        ),
        help="Used when AI Learning is included in the comparison.",
    )

    if st.button("Run Comparison", type="primary") and agents_to_compare:
        all_results = []
        progress = st.progress(0)
        status = st.empty()

        total = len(agents_to_compare) * n_comp_runs
        completed = 0

        for agent in agents_to_compare:
            status.text(f"Running {agent}...")

            for run_id in range(n_comp_runs):
                run_seed = resolve_run_seed(base_seed, run_id)
                run_kwargs = {
                    "n_steps": n_comp_steps,
                    "attack_prob": attack_prob,
                    "cyber_defense_active": defense_active,
                    "agent_type": agent,
                    "seed": run_seed,
                }
                if agent in ["intelligent", "intelligent_adaptive"]:
                    run_kwargs["efe_mode"] = "full"
                    run_kwargs["precision"] = precision
                if agent == "intelligent_adaptive":
                    run_kwargs["learning_rate"] = comp_learning_rate
                    run_kwargs["lr_schedule"] = comp_lr_schedule
                elif agent in ["qlearning", "double_qlearning"]:
                    run_kwargs["learning_rate"] = learning_rate

                df, _ = run_simulation(
                    **run_kwargs,
                )

                all_results.append(
                    {
                        "agent": {
                            "static": "PID",
                            "intelligent": "AI Static",
                            "intelligent_adaptive": "AI Learning",
                            "qlearning": "Q-Learning",
                            "double_qlearning": "Double Q-Learning",
                        }[agent],
                        "agent_type": agent,
                        "run_id": run_id,
                        "seed": run_seed,
                        "n_steps": n_comp_steps,
                        "attack_prob": attack_prob,
                        "learning_rate": comp_learning_rate
                        if agent == "intelligent_adaptive"
                        else (learning_rate if agent in ["qlearning", "double_qlearning"] else np.nan),
                        "lr_schedule": comp_lr_schedule
                        if agent == "intelligent_adaptive"
                        else "n/a",
                        "final_budget": df["budget"].iloc[-1],
                        "avg_efficiency": df["performance"].mean(),
                        "second_half_efficiency": df["performance"].iloc[
                            len(df) // 2 :
                        ].mean(),
                        "survived": df["budget"].iloc[-1] > 0,
                        "attacks_detected": df["attack_detected"].sum(),
                    }
                )

                completed += 1
                progress.progress(completed / total)

        status.text("Complete!")
        results_df = pd.DataFrame(all_results)
        st.session_state["comparison_results"] = results_df

        # Summary
        st.markdown("### Results Summary")
        summary = (
            results_df.groupby("agent")
            .agg(
                {
                    "final_budget": ["mean", "std"],
                    "avg_efficiency": "mean",
                    "survived": "mean",
                }
            )
            .round(2)
        )
        summary.columns = [
            "Avg Budget",
            "Std Budget",
            "Avg Efficiency",
            "Survival Rate",
        ]
        summary["Survival Rate"] = (summary["Survival Rate"] * 100).astype(int).astype(
            str
        ) + "%"
        st.dataframe(summary)
        st.caption(
            f"Adaptive setting used in this comparison: lr={comp_learning_rate:.3f}, schedule={comp_lr_schedule}."
        )

        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                results_df,
                x="agent",
                y="final_budget",
                color="agent",
                title="Budget Distribution by Agent",
                points="all",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                results_df.groupby("agent")["survived"].mean().reset_index(),
                x="agent",
                y="survived",
                color="agent",
                title="Survival Rate by Agent",
                labels={"survived": "Rate"},
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        # Statistical test
        if STATS_AVAILABLE and len(agents_to_compare) >= 2:
            st.markdown("### Statistical Analysis")

            for i, agent1 in enumerate(agents_to_compare):
                for agent2 in agents_to_compare[i + 1 :]:
                    name1 = {
                        "static": "PID",
                        "intelligent": "AI Static",
                        "intelligent_adaptive": "AI Learning",
                        "qlearning": "Q-Learning",
                        "double_qlearning": "Double Q-Learning",
                    }[agent1]
                    name2 = {
                        "static": "PID",
                        "intelligent": "AI Static",
                        "intelligent_adaptive": "AI Learning",
                        "qlearning": "Q-Learning",
                        "double_qlearning": "Double Q-Learning",
                    }[agent2]

                    data1 = results_df[results_df["agent"] == name1][
                        "final_budget"
                    ].values
                    data2 = results_df[results_df["agent"] == name2][
                        "final_budget"
                    ].values

                    result = independent_ttest(data1, data2)

                    sig = (
                        "**"
                        if result["p_value"] < 0.01
                        else ("*" if result["p_value"] < 0.05 else "")
                    )
                    effect = result["effect_size_interpretation"]

                    st.write(
                        f"**{name1} vs {name2}**: t={result['t_statistic']:.2f}, "
                        f"p={result['p_value']:.4f}{sig}, Cohen's d={result['cohens_d']:.2f} ({effect})"
                    )

        st.download_button(
            "Download Comparison CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=build_export_filename("agent_comparison", base_seed),
            mime="text/csv",
        )

# --- TAB 4: LEARNING ANALYSIS ---
with tab4:
    st.header("Thesis Comparison: Static vs Learning")
    st.info(
        "Core thesis view: same scenario, same attack settings, same seeds when enabled. Short runs may favor the fixed model; long runs should expose the learning advantage."
    )

    col1, col2 = st.columns(2)
    with col1:
        horizon_preset = st.radio(
            "Horizon",
            ["Short run", "Long run", "Custom"],
            horizontal=True,
            help="Use short runs to test cold-start behavior and long runs to expose the learning advantage.",
        )
    with col2:
        n_learn_runs = st.slider("Runs per Agent", 1, 10, 3, key="learn_runs")

    if horizon_preset == "Short run":
        n_learn_steps = 500
        st.caption("Short preset: 500 steps. Useful to show that learning may not win immediately.")
    elif horizon_preset == "Long run":
        n_learn_steps = 3000
        st.caption("Long preset: 3000 steps. Useful to show when learning overtakes the fixed model.")
    else:
        n_learn_steps = st.number_input(
            "Custom Steps", 200, 10000, 2000, 100, key="learn_steps_custom"
        )

    thesis_learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.001,
        max_value=0.1,
        value=float(learning_rate if agent_type == "intelligent_adaptive" else 0.01),
        step=0.001,
        format="%.3f",
        help="Adaptive agent setting used in this comparison.",
    )
    thesis_lr_schedule = st.selectbox(
        "Learning Schedule",
        ["constant", "decay", "warmup", "cosine", "adaptive"],
        index=["constant", "decay", "warmup", "cosine", "adaptive"].index(
            lr_schedule if agent_type == "intelligent_adaptive" else "decay"
        ),
        help="Adaptive agent schedule used in this comparison.",
    )

    if st.button("Run Learning Comparison", type="primary"):
        results_static = []
        results_learning = []
        sample_static = None
        sample_learning = None

        progress = st.progress(0)
        total = n_learn_runs * 2
        completed = 0

        # Static
        for i in range(n_learn_runs):
            run_seed = resolve_run_seed(base_seed, i)
            df, _ = run_simulation(
                n_steps=n_learn_steps,
                attack_prob=attack_prob,
                cyber_defense_active=defense_active,
                agent_type="intelligent",
                efe_mode="full",
                precision=precision,
                seed=run_seed,
            )
            if i == 0:
                sample_static = df
            results_static.append(
                summarize_run(
                    df,
                    run_id=i,
                    seed=run_seed,
                    agent_type="intelligent",
                    learning_rate_value=np.nan,
                    lr_schedule_value="fixed",
                )
            )
            completed += 1
            progress.progress(completed / total)

        # Learning
        for i in range(n_learn_runs):
            run_seed = resolve_run_seed(base_seed, i)
            df, _ = run_simulation(
                n_steps=n_learn_steps,
                attack_prob=attack_prob,
                cyber_defense_active=defense_active,
                agent_type="intelligent_adaptive",
                efe_mode="full",
                precision=precision,
                learning_rate=thesis_learning_rate,
                lr_schedule=thesis_lr_schedule,
                seed=run_seed,
            )
            if i == 0:
                sample_learning = df
            results_learning.append(
                summarize_run(
                    df,
                    run_id=i,
                    seed=run_seed,
                    agent_type="intelligent_adaptive",
                    learning_rate_value=thesis_learning_rate,
                    lr_schedule_value=thesis_lr_schedule,
                )
            )
            completed += 1
            progress.progress(completed / total)

        df_static = pd.DataFrame(results_static)
        df_learning = pd.DataFrame(results_learning)
        comparison_export = pd.concat(
            [
                df_static.assign(label="Static AI"),
                df_learning.assign(label="Learning AI"),
            ],
            ignore_index=True,
        )
        st.session_state["thesis_comparison_results"] = comparison_export

        # Metrics
        st.markdown("### Results")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric(
                "Static Survival",
                f"{df_static['survival'].mean() * 100:.0f}%",
            )
        with col2:
            st.metric(
                "Learning Survival",
                f"{df_learning['survival'].mean() * 100:.0f}%",
                delta=f"{(df_learning['survival'].mean() - df_static['survival'].mean()) * 100:+.0f}%",
            )
        with col3:
            st.metric("Static Budget", f"{df_static['final_budget'].mean():.0f}")
        with col4:
            st.metric(
                "Learning Budget",
                f"{df_learning['final_budget'].mean():.0f}",
                delta=f"{df_learning['final_budget'].mean() - df_static['final_budget'].mean():+.0f}",
            )
        with col5:
            st.metric(
                "Static 2nd Half Eff",
                f"{df_static['second_half_efficiency'].mean():.2f}",
            )
        with col6:
            st.metric(
                "Learning 2nd Half Eff",
                f"{df_learning['second_half_efficiency'].mean():.2f}",
                delta=f"{df_learning['second_half_efficiency'].mean() - df_static['second_half_efficiency'].mean():+.2f}",
            )

        # Plots
        if sample_static is not None and sample_learning is not None:
            st.markdown("### Sample Run Dynamics")
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sample_static["step"],
                        y=sample_static["budget"],
                        name="Static",
                        line=dict(color="#3498db"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=sample_learning["step"],
                        y=sample_learning["budget"],
                        name="Learning",
                        line=dict(color="#2ecc71"),
                    )
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="Budget Over Time", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                window = 50
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sample_static["step"],
                        y=sample_static["performance"].rolling(window).mean(),
                        name="Static",
                        line=dict(color="#3498db"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=sample_learning["step"],
                        y=sample_learning["performance"].rolling(window).mean(),
                        name="Learning",
                        line=dict(color="#2ecc71"),
                    )
                )
                fig.update_layout(
                    title=f"Efficiency (rolling avg, window={window})", height=350
                )
                st.plotly_chart(fig, use_container_width=True)

            gap_df = build_gap_dataframe(sample_static, sample_learning, performance_window=50)
            budget_crossover = find_crossover_step(
                sample_static["budget"], sample_learning["budget"], sample_static["step"]
            )
            efficiency_crossover = find_crossover_step(
                sample_static["performance"].rolling(50, min_periods=1).mean(),
                sample_learning["performance"].rolling(50, min_periods=1).mean(),
                sample_static["step"],
            )

            st.markdown("### Crossover Indicators")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Budget Crossover",
                    "Not reached" if budget_crossover is None else f"Step {budget_crossover}",
                )
            with col2:
                st.metric(
                    "Efficiency Crossover",
                    "Not reached"
                    if efficiency_crossover is None
                    else f"Step {efficiency_crossover}",
                )
            with col3:
                st.metric(
                    "Adaptive Setting",
                    f"{thesis_learning_rate:.3f} / {thesis_lr_schedule}",
                )

            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(
                    gap_df,
                    x="step",
                    y="budget_gap",
                    title="Budget Gap (Learning - Static)",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(
                    gap_df,
                    x="step",
                    y="efficiency_gap",
                    title="Efficiency Gap (Learning - Static, rolling)",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

            # Learning dynamics
            if "learning_rate" in sample_learning.columns:
                st.markdown("### Learning Dynamics")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.line(
                        sample_learning,
                        x="step",
                        y="learning_rate",
                        title="Learning Rate Schedule",
                    )
                    fig.update_traces(line_color="#e74c3c")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if "model_divergence" in sample_learning.columns:
                        fig = px.line(
                            sample_learning,
                            x="step",
                            y="model_divergence",
                            title="Model Divergence from Initial",
                        )
                        fig.update_traces(line_color="#9b59b6")
                        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Export")
        st.download_button(
            "Download Thesis Comparison CSV",
            data=comparison_export.to_csv(index=False).encode("utf-8"),
            file_name=build_export_filename("thesis_comparison", base_seed),
            mime="text/csv",
        )

# --- TAB 5: Q-LEARNING VISUALIZATION ---
with tab5:
    st.header("Q-Learning Analysis")

    if agent_type not in ["qlearning", "double_qlearning"]:
        st.warning(
            "Select Q-Learning or Double Q-Learning agent in the sidebar to use this tab."
        )
    else:
        st.info(
            "Analyze Q-Learning agent behavior: Q-table, TD errors, and epsilon decay"
        )

        col1, col2 = st.columns(2)
        with col1:
            q_steps = st.number_input(
                "Training Steps", 500, 5000, 1000, 100, key="q_steps"
            )
        with col2:
            q_runs = st.slider("Number of Runs", 1, 5, 1, key="q_runs")

        if st.button("Train Q-Learning Agent", type="primary"):
            with st.spinner("Training Q-Learning agent..."):
                # Create and train agent
                if agent_type == "double_qlearning":
                    q_agent = DoubleQLearningAgent(
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        discount_factor=discount_factor,
                    )
                else:
                    q_agent = QLearningAgent(
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        discount_factor=discount_factor,
                    )

                # Run simulation
                df, _ = run_simulation(
                    n_steps=q_steps,
                    attack_prob=attack_prob,
                    cyber_defense_active=defense_active,
                    agent_type=agent_type,
                    learning_rate=learning_rate,
                    seed=resolve_run_seed(base_seed, 0),
                    agent=q_agent,
                )

                # Get learning history from agent
                q_history = (
                    q_agent.get_learning_history_df()
                    if hasattr(q_agent, "get_learning_history_df")
                    else None
                )

                st.session_state["q_agent"] = q_agent
                st.session_state["q_data"] = df

            st.success("Training complete!")

            # Display metrics
            stats = q_agent.get_learning_stats()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("States Visited", stats["n_visited_states"])
            with col2:
                st.metric("Total Updates", stats["total_updates"])
            with col3:
                st.metric("Final Epsilon", f"{stats['epsilon']:.4f}")
            with col4:
                st.metric("Avg TD Error", f"{stats['avg_td_error']:.4f}")

            # Q-Table Visualization
            st.markdown("### Q-Table Heatmap")

            if len(q_agent.Q) > 0:
                # Create Q-table visualization
                states = list(q_agent.Q.keys())
                q_values = np.array([q_agent.Q[s] for s in states])

                # Show top visited states
                state_labels = [f"S{s}" for s in states[: min(20, len(states))]]
                action_labels = [
                    "Cool-Dec-Wait",
                    "Cool-Dec-Ver",
                    "Cool-Main-Wait",
                    "Cool-Main-Ver",
                    "Cool-Inc-Wait",
                    "Cool-Inc-Ver",
                    "Main-Dec-Wait",
                    "Main-Dec-Ver",
                    "Main-Main-Wait",
                    "Main-Main-Ver",
                    "Main-Inc-Wait",
                    "Main-Inc-Ver",
                    "Heat-Dec-Wait",
                    "Heat-Dec-Ver",
                    "Heat-Main-Wait",
                    "Heat-Main-Ver",
                    "Heat-Inc-Wait",
                    "Heat-Inc-Ver",
                ]

                q_subset = q_values[: min(20, len(q_values)), :]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=q_subset,
                        x=action_labels,
                        y=state_labels,
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig.update_layout(
                    title="Q-Values for Top 20 Visited States",
                    xaxis_title="Action",
                    yaxis_title="State",
                    height=500,
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                # Best action per state
                st.markdown("### Best Actions per State")
                best_actions = np.argmax(q_subset, axis=1)
                best_q = np.max(q_subset, axis=1)

                fig2 = go.Figure()
                fig2.add_trace(
                    go.Bar(
                        x=state_labels,
                        y=best_q,
                        text=[action_labels[a][:8] for a in best_actions],
                        textposition="auto",
                    )
                )
                fig2.update_layout(
                    title="Maximum Q-Value per State",
                    xaxis_title="State",
                    yaxis_title="Max Q-Value",
                    height=350,
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Performance over time
            st.markdown("### Training Performance")
            col1, col2 = st.columns(2)

            with col1:
                # Budget over time
                fig = px.line(df, x="step", y="budget", title="Budget During Training")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Efficiency rolling average
                window = 50
                eff_smooth = (
                    df["performance"].rolling(window=window, min_periods=1).mean()
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["step"], y=eff_smooth, name="Efficiency"))
                fig.update_layout(
                    title=f"Efficiency (Rolling Avg, window={window})", height=350
                )
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 6: B MATRIX VISUALIZATION ---
with tab6:
    st.header("B Matrix Visualization")

    if not B_MATRIX_VIZ_AVAILABLE:
        st.error("B Matrix visualization module not available. Check imports.")
    elif agent_type not in ["intelligent_adaptive"]:
        st.warning(
            "Select 'Active Inference (Learning)' agent in the sidebar to visualize B matrix learning."
        )
    else:
        st.info("Visualize how the B matrix (transition model) evolves during learning")

        col1, col2 = st.columns(2)
        with col1:
            b_steps = st.number_input(
                "Training Steps", 500, 5000, 1000, 100, key="b_steps"
            )
        with col2:
            record_interval = st.slider(
                "Recording Interval", 10, 100, 50, key="rec_int"
            )

        factor_names = ["Temperature", "Motor", "Load", "Temp_Health", "Motor_Health"]
        selected_factor = st.selectbox("Factor to Visualize", factor_names)
        factor_idx = factor_names.index(selected_factor)

        if st.button("Train and Record B Matrix", type="primary"):
            with st.spinner("Training agent and recording B matrix evolution..."):
                if use_fixed_seed:
                    set_global_seed(base_seed)

                # Create adaptive agent
                agent = AdaptiveActiveInferenceAgent(
                    learning_rate=learning_rate,
                    lr_schedule=lr_schedule,
                    efe_mode=efe_mode,
                    precision=precision,
                )

                # Create recorder
                recorder = BMatrixRecorder(agent, interval=record_interval)

                # Run simulation step by step
                from simulation import Environment, Sensor, Actuator
                import torch

                env = Environment(
                    initial_budget=1000.0, cyber_defense_active=defense_active
                )
                sensors = {
                    "temperature": Sensor(
                        "temperature", noise_std_dev=0.5, sampling_interval=5
                    ),
                    "motor_temperature": Sensor(
                        "motor_temperature", noise_std_dev=0.5, sampling_interval=5
                    ),
                    "load": Sensor("load", noise_std_dev=0.2, sampling_interval=5),
                }

                progress = st.progress(0)
                for step in range(b_steps):
                    # Read sensors
                    temp = sensors["temperature"].read(env.temperature).item()
                    motor = (
                        sensors["motor_temperature"].read(env.motor_temperature).item()
                    )
                    load = sensors["load"].read(env.load).item()

                    # Random attacks
                    attack = np.random.random() < attack_prob

                    # Agent step
                    temp_cmd, load_cmd, verify = agent.step(temp, motor, load, attack)

                    # Environment step
                    env.step(torch.tensor([temp_cmd]), torch.tensor([load_cmd]))
                    env.budget += env.calculate_performance() * 1.5

                    # Record
                    recorder.record(step)

                    if step % 100 == 0:
                        progress.progress(step / b_steps)

                progress.progress(1.0)
                st.session_state["b_agent"] = agent
                st.session_state["b_recorder"] = recorder

            st.success("Training complete!")

            # Display B Matrix visualizations using matplotlib figures converted to Plotly
            st.markdown("### B Matrix Summary")

            # Get B matrix snapshot
            snapshot = agent.get_b_matrix_snapshot()

            # Display learning stats
            stats = agent.get_learning_stats()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Updates", stats["total_updates"])
            with col2:
                st.metric("Learning Rate", f"{stats['learning_rate']:.4f}")
            with col3:
                st.metric("Model Divergence", f"{stats['model_divergence']:.4f}")
            with col4:
                st.metric("Avg Pred Error", f"{stats['avg_prediction_error']:.4f}")

            # Show B matrix for selected factor
            st.markdown(f"### {selected_factor} Factor B Matrix")

            data = snapshot[selected_factor]
            B_init = data["initial_matrix"]
            B_learned = data["matrix"]
            B_diff = data["change"]
            actions = data["actions"]
            states = data["states"]

            # Create heatmaps for each action
            for a_idx, action in enumerate(actions):
                st.markdown(f"#### Action: {action}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=B_init[:, :, a_idx],
                            x=states,
                            y=states,
                            colorscale="Blues",
                            zmin=0,
                            zmax=1,
                            text=np.round(B_init[:, :, a_idx], 2),
                            texttemplate="%{text}",
                        )
                    )
                    fig.update_layout(
                        title="Initial",
                        height=300,
                        xaxis_title="Current",
                        yaxis_title="Next",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=B_learned[:, :, a_idx],
                            x=states,
                            y=states,
                            colorscale="Blues",
                            zmin=0,
                            zmax=1,
                            text=np.round(B_learned[:, :, a_idx], 2),
                            texttemplate="%{text}",
                        )
                    )
                    fig.update_layout(
                        title="Learned",
                        height=300,
                        xaxis_title="Current",
                        yaxis_title="Next",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    fig = go.Figure(
                        data=go.Heatmap(
                            z=B_diff[:, :, a_idx],
                            x=states,
                            y=states,
                            colorscale="RdBu",
                            zmid=0,
                            text=np.round(B_diff[:, :, a_idx], 3),
                            texttemplate="%{text:+.3f}",
                        )
                    )
                    fig.update_layout(
                        title="Change",
                        height=300,
                        xaxis_title="Current",
                        yaxis_title="Next",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Summary of changes across all factors
            st.markdown("### Learning Summary (All Factors)")

            change_summary = []
            for f_name in factor_names:
                f_data = snapshot[f_name]
                total_change = np.sum(np.abs(f_data["change"]))
                change_summary.append(
                    {"Factor": f_name, "Total |Change|": total_change}
                )

            change_df = pd.DataFrame(change_summary)
            fig = px.bar(
                change_df,
                x="Factor",
                y="Total |Change|",
                color="Factor",
                title="Total B Matrix Change by Factor",
            )
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 7: CURRICULUM LEARNING ---
with tab7:
    st.header("Curriculum Learning")

    if not CURRICULUM_AVAILABLE:
        st.error("Curriculum Learning module not available. Check imports.")
    else:
        st.info("Train agents with gradually increasing attack difficulty")

        # Curriculum configuration
        st.markdown("### Curriculum Configuration")

        col1, col2, col3 = st.columns(3)
        with col1:
            curriculum_steps = st.number_input(
                "Total Steps", 1000, 10000, 4000, 500, key="curr_steps"
            )
        with col2:
            curriculum_strategy = st.selectbox(
                "Progression Strategy",
                ["step_based", "performance_based", "loss_based", "survival_based"],
                help="How to decide when to advance difficulty",
            )
        with col3:
            steps_per_stage = curriculum_steps // 4
            st.metric("Steps per Stage", steps_per_stage)

        # Show curriculum stages
        st.markdown("### Curriculum Stages")
        stages_data = [
            {
                "Stage": "Easy",
                "Attack Prob": 0.005,
                "Types": "Bias only",
                "Duration": "5-20",
            },
            {
                "Stage": "Medium",
                "Attack Prob": 0.015,
                "Types": "Bias, Outlier",
                "Duration": "10-50",
            },
            {
                "Stage": "Hard",
                "Attack Prob": 0.025,
                "Types": "Bias, Outlier, Spoofing",
                "Duration": "20-80",
            },
            {
                "Stage": "Adversarial",
                "Attack Prob": 0.04,
                "Types": "All (including DoS)",
                "Duration": "30-100",
            },
        ]
        st.table(pd.DataFrame(stages_data))

        # Agent selection for curriculum
        curriculum_agent = st.selectbox(
            "Agent to Train",
            ["intelligent_adaptive", "qlearning"],
            format_func=lambda x: {
                "intelligent_adaptive": "AI Learning",
                "qlearning": "Q-Learning",
            }[x],
            key="curr_agent",
        )

        if st.button("Run Curriculum Training", type="primary"):
            with st.spinner("Running curriculum training..."):
                # Create agent
                if curriculum_agent == "intelligent_adaptive":
                    agent = AdaptiveActiveInferenceAgent(
                        learning_rate=learning_rate, lr_schedule=lr_schedule
                    )
                else:
                    agent = QLearningAgent(learning_rate=learning_rate, epsilon=epsilon)

                # Run curriculum simulation
                df, curriculum_hist = run_curriculum_simulation(
                    agent,
                    n_steps=curriculum_steps,
                    curriculum_strategy=curriculum_strategy,
                    steps_per_stage=steps_per_stage,
                )

                st.session_state["curriculum_df"] = df
                st.session_state["curriculum_hist"] = curriculum_hist

            st.success("Curriculum training complete!")

            # Results
            st.markdown("### Training Results")

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Budget", f"{df['budget'].iloc[-1]:.0f}")
            with col2:
                st.metric("Survived", "Yes" if df["budget"].iloc[-1] > 0 else "No")
            with col3:
                st.metric("Total Attacks", int(df["is_under_attack"].sum()))
            with col4:
                st.metric("Attacks Detected", int(df["attack_detected"].sum()))

            # Performance by stage
            st.markdown("### Performance by Curriculum Stage")

            stage_stats = (
                df.groupby("curriculum_stage")
                .agg(
                    {
                        "performance": "mean",
                        "budget": "last",
                        "is_under_attack": "sum",
                        "attack_detected": "sum",
                    }
                )
                .reindex(["easy", "medium", "hard", "adversarial"])
            )

            stage_stats.columns = [
                "Avg Efficiency",
                "End Budget",
                "Attacks",
                "Detected",
            ]
            st.dataframe(stage_stats.round(2))

            # Plots
            col1, col2 = st.columns(2)

            with col1:
                # Budget over time with stage markers
                fig = go.Figure()

                for stage in ["easy", "medium", "hard", "adversarial"]:
                    stage_data = df[df["curriculum_stage"] == stage]
                    if len(stage_data) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=stage_data["step"],
                                y=stage_data["budget"],
                                name=stage.capitalize(),
                                mode="lines",
                            )
                        )

                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(title="Budget by Curriculum Stage", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Efficiency by stage (boxplot)
                fig = px.box(
                    df,
                    x="curriculum_stage",
                    y="performance",
                    title="Efficiency Distribution by Stage",
                    category_orders={
                        "curriculum_stage": ["easy", "medium", "hard", "adversarial"]
                    },
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Attack probability over time
            st.markdown("### Difficulty Progression")
            fig = px.line(
                df,
                x="step",
                y="curriculum_attack_prob",
                title="Attack Probability Throughout Training",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 8: STATISTICAL ANALYSIS ---
with tab8:
    st.header("Statistical Analysis")

    if not STATS_AVAILABLE:
        st.warning("Statistical analysis module not available. Run `pip install scipy`")
    else:
        st.info("Comprehensive statistical analysis with export options")

        analysis_mode = st.radio(
            "Analysis Mode",
            ["Quick Demo", "Analyze Comparison Results", "Custom Data Upload"],
            horizontal=True,
        )

        if analysis_mode == "Quick Demo":
            if st.button("Run Statistical Demo"):
                # Generate sample data
                np.random.seed(42)
                static_data = np.random.normal(500, 300, 20)
                learning_data = np.random.normal(5000, 1500, 20)
                qlearning_data = np.random.normal(3000, 1000, 20)

                st.markdown("### Sample Data (Simulated)")
                st.write(
                    "Static Agent: N(500, 300) | Learning Agent: N(5000, 1500) | Q-Learning: N(3000, 1000)"
                )

                # T-tests
                st.markdown("### Pairwise T-Tests")

                comparisons = [
                    ("Static", "Learning", static_data, learning_data),
                    ("Static", "Q-Learning", static_data, qlearning_data),
                    ("Learning", "Q-Learning", learning_data, qlearning_data),
                ]

                for name1, name2, data1, data2 in comparisons:
                    result = independent_ttest(data1, data2)
                    sig = (
                        "**"
                        if result["significant_at_01"]
                        else ("*" if result["significant_at_05"] else "")
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            f"{name1} vs {name2}", f"t = {result['t_statistic']:.3f}"
                        )
                    with col2:
                        st.metric("p-value", f"{result['p_value']:.4f}{sig}")
                    with col3:
                        st.metric("Cohen's d", f"{result['cohens_d']:.3f}")
                    with col4:
                        st.metric("Effect", result["effect_size_interpretation"])

                # ANOVA
                st.markdown("### One-Way ANOVA")
                groups = {
                    "Static": static_data,
                    "Learning": learning_data,
                    "Q-Learning": qlearning_data,
                }
                anova_result = one_way_anova(groups)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F-statistic", f"{anova_result['f_statistic']:.3f}")
                with col2:
                    sig = (
                        "**"
                        if anova_result["p_value"] < 0.01
                        else ("*" if anova_result["p_value"] < 0.05 else "")
                    )
                    st.metric("p-value", f"{anova_result['p_value']:.6f}{sig}")
                with col3:
                    st.metric("Eta-squared", f"{anova_result['eta_squared']:.4f}")

                # Confidence intervals with Bootstrap
                st.markdown("### Bootstrap Confidence Intervals (95%)")

                for name, data in [
                    ("Static", static_data),
                    ("Learning", learning_data),
                    ("Q-Learning", qlearning_data),
                ]:
                    obs, low, high = bootstrap_ci(data, n_bootstrap=5000)
                    st.write(
                        f"**{name}**: Mean = {obs:.2f}, 95% CI = [{low:.2f}, {high:.2f}]"
                    )

                # Visualization
                fig = go.Figure()

                for name, data, color in [
                    ("Static", static_data, "#3498db"),
                    ("Learning", learning_data, "#2ecc71"),
                    ("Q-Learning", qlearning_data, "#e74c3c"),
                ]:
                    mean, ci_low, ci_high = compute_confidence_interval(data)
                    fig.add_trace(
                        go.Bar(
                            name=name,
                            x=[name],
                            y=[mean],
                            error_y=dict(
                                type="data",
                                symmetric=False,
                                array=[ci_high - mean],
                                arrayminus=[mean - ci_low],
                            ),
                            marker_color=color,
                        )
                    )

                fig.update_layout(
                    title="Mean with 95% Confidence Intervals",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # LaTeX Export
                st.markdown("### Export for Thesis")

                # Create analysis dict for LaTeX generation
                data_for_analysis = []
                for label, budget in [
                    ("Static", static_data),
                    ("Learning", learning_data),
                    ("Q-Learning", qlearning_data),
                ]:
                    for i, b in enumerate(budget):
                        data_for_analysis.append(
                            {
                                "config_label": label,
                                "run_id": i,
                                "final_budget": b,
                            }
                        )

                analysis_df = pd.DataFrame(data_for_analysis)
                analysis = analyze_experiment_results(
                    analysis_df,
                    group_column="config_label",
                    metric_columns=["final_budget"],
                )

                latex_table = generate_latex_stats_table(
                    analysis,
                    "final_budget",
                    caption="Budget Comparison Across Agent Types",
                    label="tab:budget_comparison",
                )

                st.code(latex_table, language="latex")

                st.download_button(
                    "Download LaTeX Table",
                    data=latex_table,
                    file_name="stats_table.tex",
                    mime="text/plain",
                )

        elif analysis_mode == "Analyze Comparison Results":
            if "comparison_results" in st.session_state:
                results_df = st.session_state["comparison_results"]

                st.markdown("### Analyzing Agent Comparison Results")

                # Perform full analysis
                analysis = analyze_experiment_results(
                    results_df,
                    group_column="agent",
                    metric_columns=["final_budget", "avg_efficiency"],
                )

                # Summary statistics
                st.markdown("#### Summary Statistics")
                for metric, groups in analysis["summary_statistics"].items():
                    st.markdown(f"**{metric}:**")
                    stats_table = []
                    for group, stats in groups.items():
                        ci = stats["ci_95"]
                        stats_table.append(
                            {
                                "Agent": group,
                                "Mean": f"{stats['mean']:.2f}",
                                "Std": f"{stats['std']:.2f}",
                                "95% CI": f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                                "N": stats["n"],
                            }
                        )
                    st.table(pd.DataFrame(stats_table))

                # Pairwise comparisons
                if analysis["pairwise_comparisons"]:
                    st.markdown("#### Pairwise Comparisons")
                    for metric, comparisons in analysis["pairwise_comparisons"].items():
                        st.markdown(f"**{metric}:**")
                        for comp_name, stats in comparisons.items():
                            sig = (
                                "**"
                                if stats["significant_at_01"]
                                else ("*" if stats["significant_at_05"] else "")
                            )
                            st.write(
                                f"- {comp_name}: t={stats['t_statistic']:.3f}, p={stats['p_value']:.4f}{sig}, d={stats['cohens_d']:.3f} ({stats['effect_size_interpretation']})"
                            )

                # ANOVA
                if analysis["anova_results"]:
                    st.markdown("#### ANOVA Results")
                    for metric, stats in analysis["anova_results"].items():
                        st.markdown(f"**{metric}:**")
                        sig = (
                            "**"
                            if stats["p_value"] < 0.01
                            else ("*" if stats["p_value"] < 0.05 else "")
                        )
                        st.write(
                            f"F = {stats['f_statistic']:.3f}, p = {stats['p_value']:.4f}{sig}, eta^2 = {stats['eta_squared']:.4f}"
                        )

                # LaTeX export
                st.markdown("#### Export LaTeX Tables")
                for metric in ["final_budget", "avg_efficiency"]:
                    if metric in analysis["summary_statistics"]:
                        latex = generate_latex_stats_table(
                            analysis,
                            metric,
                            caption=f"{metric.replace('_', ' ').title()} Comparison",
                            label=f"tab:{metric}",
                        )
                        with st.expander(f"LaTeX Table: {metric}"):
                            st.code(latex, language="latex")
                            st.download_button(
                                f"Download {metric}.tex",
                                data=latex,
                                file_name=f"stats_{metric}.tex",
                                mime="text/plain",
                                key=f"dl_{metric}",
                            )
            else:
                st.warning(
                    "No comparison results available. Run an Agent Comparison first (Tab 3)."
                )

        else:  # Custom Data Upload
            st.markdown("### Upload Experiment Results")

            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

            if uploaded_file is not None:
                results_df = pd.read_csv(uploaded_file)
                st.dataframe(results_df.head())

                # Select columns
                group_col = st.selectbox("Group Column", results_df.columns)
                metric_cols = st.multiselect(
                    "Metric Columns",
                    [
                        c
                        for c in results_df.columns
                        if results_df[c].dtype in ["float64", "int64"]
                    ],
                )

                if st.button("Analyze Data"):
                    analysis = analyze_experiment_results(
                        results_df, group_column=group_col, metric_columns=metric_cols
                    )

                    # Display results similar to above
                    for metric, groups in analysis["summary_statistics"].items():
                        st.markdown(f"**{metric}:**")
                        for group, stats in groups.items():
                            ci = stats["ci_95"]
                            st.write(
                                f"- {group}: Mean={stats['mean']:.2f}, SD={stats['std']:.2f}, 95% CI=[{ci[0]:.2f}, {ci[1]:.2f}]"
                            )

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("IIoT Security Testbed v3.0")
st.sidebar.caption("Active Inference vs Q-Learning")
st.sidebar.caption("For Thesis Research")
