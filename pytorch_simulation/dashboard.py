
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulation import run_simulation, run_batch_simulation
import wandb
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cyber-Physical Simulation Testbed",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d0d0d0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #444444 !important;
    }
        font-size: 0.9em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 1.8em;
        font-weight: bold;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #000000 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #28a745 !important; /* Green for positive delta */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
        fill: #28a745 !important;
    }
    /* General header styling */
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Simulation Parameters
n_steps = st.sidebar.slider("Simulation Steps", min_value=100, max_value=2000, value=500, step=100)
attack_prob = st.sidebar.slider("Attack Probability", min_value=0.0, max_value=0.1, value=0.02, step=0.01)
defense_active = st.sidebar.toggle("Cyber Defense Active", value=True)
agent_type = st.sidebar.selectbox("Agent Type", ["static", "intelligent"], index=0)

# EFE Mode (only for intelligent agent)
st.sidebar.markdown("---")
st.sidebar.markdown("**Active Inference Settings**")
efe_mode = st.sidebar.selectbox(
    "EFE Mode", 
    ["full", "epistemic_only", "pragmatic_only"],
    index=0,
    help="Expected Free Energy calculation mode: full (both terms), epistemic_only (curiosity/exploration), pragmatic_only (goal achievement)"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the 'Batch Experiment' tab to compare performance with/without defense.")
st.sidebar.info("🧠 **EFE Modes:**\n- **full**: Balanced behavior\n- **epistemic_only**: Exploration/verification focused\n- **pragmatic_only**: Goal-driven, higher risk")

# --- MAIN CONTENT ---
st.title("🛡️ Cyber-Physical System Testbed")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Simulation", "Batch Experiment", "Comparative Experiment", "📊 WandB History"])

# --- TAB 1: SINGLE RUN ---
with tab1:
    st.subheader("Real-time Simulation Monitor")
    
    # Initialize session state for run data if not present
    if 'single_run_data' not in st.session_state:
        st.session_state['single_run_data'] = None

    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Simulating..."):
            try:
                # Esegui simulazione
                df, _ = run_simulation(n_steps=n_steps, attack_prob=attack_prob, cyber_defense_active=defense_active, agent_type=agent_type.split()[0], efe_mode=efe_mode)
                st.session_state['single_run_data'] = df
                st.success(f"Simulation completed! Agent: {agent_type}, EFE Mode: {efe_mode}")
            except Exception as e:
                st.error(f"An error occurred during simulation: {e}")
                st.session_state['single_run_data'] = None

    # Display results if data exists
    if st.session_state['single_run_data'] is not None:
        df = st.session_state['single_run_data']
        
        # --- METRICS ROW ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Calculate bankruptcy step
            bankruptcy_step = df[df['budget'] <= 0]['step'].min()
            if pd.isna(bankruptcy_step):
                step_display = "Survived"
            else:
                step_display = f"Bankruptcy at Step {int(bankruptcy_step)}"
            
            st.metric(f"Final Budget ({step_display})", f"{df['budget'].iloc[-1]:.2f}", delta=f"{df['budget'].iloc[-1] - 1000:.2f}")
        with col2:
            st.metric("Avg Efficiency", f"{df['performance'].mean():.2f}")
        with col3:
            st.metric("Total Attacks", int(df['is_under_attack'].sum()))
        with col4:
            st.metric("Attacks Detected", int(df['attack_detected'].sum()))

        # --- INTERACTIVE PLOTS (PLOTLY) ---
        
        # 1. Temperature & Motor Temp
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=df['step'], y=df['true_temp'], name='True Temp', line=dict(color='blue')))
        fig_temp.add_trace(go.Scatter(x=df['step'], y=df['sensed_temp'], name='Sensed Temp', line=dict(color='lightblue', dash='dot')))
        fig_temp.add_trace(go.Scatter(x=df['step'], y=df['true_motor_temp'], name='True Motor Temp', line=dict(color='red')))
        fig_temp.add_trace(go.Scatter(x=df['step'], y=df['sensed_motor_temp'], name='Sensed Motor Temp', line=dict(color='pink', dash='dot')))
        
        fig_temp.update_layout(title="Temperature Dynamics", xaxis_title="Step", yaxis_title="Temperature (°C)", height=400)
        st.plotly_chart(fig_temp, use_container_width=True)

        # 2. Load & Efficiency
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            fig_load = go.Figure()
            fig_load.add_trace(go.Scatter(x=df['step'], y=df['true_load'], name='True Load', line=dict(color='green')))
            fig_load.add_trace(go.Scatter(x=df['step'], y=df['sensed_load'], name='Sensed Load', line=dict(color='lightgreen', dash='dot')))
            fig_load.update_layout(title="Load Dynamics", height=350)
            st.plotly_chart(fig_load, use_container_width=True)
        
        with col_plot2:
            fig_eff = px.line(df, x='step', y='performance', title="System Efficiency")
            fig_eff.update_traces(line_color='purple')
            fig_eff.update_layout(height=350)
            st.plotly_chart(fig_eff, use_container_width=True)

        # 3. Budget
        fig_budget = px.area(df, x='step', y='budget', title="Budget Trend")
        fig_budget.update_traces(line_color='gold')
        st.plotly_chart(fig_budget, use_container_width=True)

        # --- DATA EXPORT ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Run Data (CSV)", data=csv, file_name="simulation_run.csv", mime="text/csv")
            
        # Salvataggio Run for comparison
        st.divider()
        st.write("Save this run for comparison:")
        run_name_default = f"Run {len(st.session_state.get('saved_runs', {})) + 1}"
        run_name = st.text_input("Run Name", value=run_name_default)
        if st.button("Save Run"):
            if 'saved_runs' not in st.session_state:
                st.session_state['saved_runs'] = {}
            st.session_state['saved_runs'][run_name] = df # Save the current df from single_run_data
            st.success(f"Run '{run_name}' saved!")


# --- TAB 2: BATCH EXPERIMENT ---
with tab2:
    st.subheader("🧪 Batch Experiment & Statistics")
    
    # Mode Selection
    analysis_mode = st.radio("Analysis Mode", ["Compare Defense ON vs OFF", "Analyze Current Configuration"], horizontal=True)
    
    n_batch_runs = st.slider("Number of Runs", 5, 100, 20)
    
    if analysis_mode == "Compare Defense ON vs OFF":
        st.markdown("Run multiple simulations to statistically verify the impact of Cyber Defense.")
        
        if st.button("⚔️ Run Comparison"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Run Defense OFF
            status_text.text(f"Running {n_batch_runs} simulations with Defense OFF...")
            df_off = run_batch_simulation(n_runs=n_batch_runs, n_steps=n_steps, attack_prob=attack_prob, cyber_defense_active=False, agent_type=agent_type.split()[0], efe_mode=efe_mode)
            df_off['Configuration'] = 'Defense OFF'
            progress_bar.progress(50)
            
            # 2. Run Defense ON
            status_text.text(f"Running {n_batch_runs} simulations with Defense ON...")
            df_on = run_batch_simulation(n_runs=n_batch_runs, n_steps=n_steps, attack_prob=attack_prob, cyber_defense_active=True, agent_type=agent_type.split()[0], efe_mode=efe_mode)
            df_on['Configuration'] = 'Defense ON'
            progress_bar.progress(100)
            
            status_text.text("Analysis Complete!")
            
            # Combine results
            df_compare = pd.concat([df_off, df_on])
            
            # --- RESULTS METRICS ---
            avg_budget_off = df_off['final_budget'].mean()
            avg_budget_on = df_on['final_budget'].mean()
            budget_delta = avg_budget_on - avg_budget_off
            
            avg_eff_off = df_off['avg_efficiency'].mean()
            avg_eff_on = df_on['avg_efficiency'].mean()
            eff_delta = avg_eff_on - avg_eff_off
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Budget (OFF)", f"{avg_budget_off:.2f}")
            with col2:
                st.metric("Avg Budget (ON)", f"{avg_budget_on:.2f}", delta=f"{budget_delta:.2f}")
            with col3:
                st.metric("Avg Efficiency (OFF)", f"{avg_eff_off:.2f}")
            with col4:
                st.metric("Avg Efficiency (ON)", f"{avg_eff_on:.2f}", delta=f"{eff_delta:.2f}")
                
            # --- COMPARISON PLOTS ---
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                fig_box = px.box(df_compare, x="Configuration", y="final_budget", color="Configuration", 
                                 title="Final Budget Distribution", points="all")
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col_c2:
                fig_box_eff = px.box(df_compare, x="Configuration", y="avg_efficiency", color="Configuration", 
                                     title="Average Efficiency Distribution", points="all")
                st.plotly_chart(fig_box_eff, use_container_width=True)

    else: # Analyze Current Configuration
        st.markdown(f"Run {n_batch_runs} simulations with the **current sidebar settings**.")
        st.info(f"Current Settings: Defense={'ON' if defense_active else 'OFF'}, Attack Prob={attack_prob}, Steps={n_steps}")
        
        if st.button("🧪 Run Batch Analysis"):
            with st.spinner(f"Running {n_batch_runs} simulations..."):
                df_batch = run_batch_simulation(n_runs=n_batch_runs, n_steps=n_steps, attack_prob=attack_prob, cyber_defense_active=defense_active, agent_type=agent_type.split()[0], efe_mode=efe_mode)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Final Budget", f"{df_batch['final_budget'].mean():.2f}")
                col2.metric("Avg Efficiency", f"{df_batch['avg_efficiency'].mean():.2f}")
                col3.metric("Success Rate (>0 Budget)", f"{(df_batch['final_budget'] > 0).mean() * 100:.1f}%")
                
                # Plots
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    fig_hist = px.histogram(df_batch, x="final_budget", nbins=20, title="Final Budget Distribution", marginal="box")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col_p2:
                    fig_hist_eff = px.histogram(df_batch, x="avg_efficiency", nbins=20, title="Efficiency Distribution", marginal="box")
                    st.plotly_chart(fig_hist_eff, use_container_width=True)
                
                # Raw Data
                with st.expander("View Raw Batch Data"):
                    st.dataframe(df_batch)
    # --- Tab 3: Comparative Experiment ---
    with tab3:
        st.header("🔬 Comparative Experiment")
        st.info("Compare multiple agent configurations side-by-side.")

        col1, col2 = st.columns(2)
        with col1:
            n_runs_comp = st.number_input("Runs per Configuration", min_value=1, max_value=50, value=10, step=1)
        with col2:
            n_steps_comp = st.number_input("Simulation Steps (Comparison)", min_value=100, max_value=2000, value=500, step=100)

        st.subheader("Configurations")
        
        # Dynamic Configuration Adder
        if 'comp_configs' not in st.session_state:
            st.session_state.comp_configs = [{'agent': 'static', 'defense': True, 'efe_mode': 'full'}]

        def add_config():
            if len(st.session_state.comp_configs) < 4:
                st.session_state.comp_configs.append({'agent': 'intelligent', 'defense': True, 'efe_mode': 'full'})

        def remove_config():
            if len(st.session_state.comp_configs) > 1:
                st.session_state.comp_configs.pop()

        c1, c2 = st.columns([1, 1])
        with c1:
            st.button("➕ Add Config", on_click=add_config, disabled=len(st.session_state.comp_configs) >= 4)
        with c2:
            st.button("➖ Remove Config", on_click=remove_config, disabled=len(st.session_state.comp_configs) <= 1)

        # Display Config Editors
        configs_to_run = []
        for i, config in enumerate(st.session_state.comp_configs):
            st.markdown(f"**Config {i+1}**")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                agent_sel = st.selectbox(f"Agent Type #{i+1}", ["static", "intelligent"], index=0 if config['agent']=='static' else 1, key=f"c_agent_{i}")
            with cc2:
                def_sel = st.toggle(f"Cyber Defense #{i+1}", value=config['defense'], key=f"c_def_{i}")
            with cc3:
                efe_sel = st.selectbox(f"EFE Mode #{i+1}", ["full", "epistemic_only", "pragmatic_only"], 
                                       index=["full", "epistemic_only", "pragmatic_only"].index(config.get('efe_mode', 'full')),
                                       key=f"c_efe_{i}", disabled=(agent_sel == 'static'))
            
            # Update session state (indirectly via list reconstruction for run)
            configs_to_run.append({'agent': agent_sel, 'defense': def_sel, 'efe_mode': efe_sel})

        if st.button("🚀 Run Comparison"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_runs = len(configs_to_run) * n_runs_comp
            completed_runs = 0

            for i, cfg in enumerate(configs_to_run):
                efe_display = cfg['efe_mode'] if cfg['agent'] == 'intelligent' else 'N/A'
                status_text.text(f"Running Config {i+1}: Agent={cfg['agent']}, Defense={cfg['defense']}, EFE={efe_display}...")
                
                try:
                    # We can't easily track progress INSIDE run_batch_simulation without modifying it,
                    # so we just update per config block or modify run_batch to accept a callback?
                    # For simplicity, we just run the batch.
                    df = run_batch_simulation(n_runs=n_runs_comp, n_steps=n_steps_comp, 
                                              attack_prob=attack_prob, # Use global attack prob from sidebar
                                              cyber_defense_active=cfg['defense'], 
                                              agent_type=cfg['agent'],
                                              efe_mode=cfg['efe_mode'])
                    
                    avg_eff = df['avg_efficiency'].mean()
                    avg_budget = df['final_budget'].mean()
                    avg_attacks = df['total_attacks'].mean()
                    avg_detected = df['attacks_detected'].mean()
                    
                    results.append({
                        'Config': f"#{i+1}",
                        'Agent': cfg['agent'],
                        'Defense': 'ON' if cfg['defense'] else 'OFF',
                        'EFE Mode': efe_display,
                        'Avg Efficiency': f"{avg_eff:.2f}",
                        'Avg Final Budget': f"{avg_budget:.2f}",
                        'Avg Attacks': f"{avg_attacks:.1f}",
                        'Avg Detected': f"{avg_detected:.1f}"
                    })
                    
                    completed_runs += n_runs_comp
                    progress_bar.progress(min(completed_runs / total_runs, 1.0))
                    
                except Exception as e:
                    st.error(f"Error in Config {i+1}: {e}")

            status_text.text("Comparison Complete!")
            progress_bar.progress(1.0)
            
            st.subheader("Results")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            
            # Highlight Winner
            best_budget = res_df.loc[res_df['Avg Final Budget'].astype(float).idxmax()]
            st.success(f"🏆 Best Financial Performance: Config {best_budget['Config']} ({best_budget['Agent']}, Defense {best_budget['Defense']}) with Budget {best_budget['Avg Final Budget']}")

# --- TAB 4: WANDB HISTORY ---
with tab4:
    st.header("📊 WandB Run History")
    st.markdown("View and analyze past simulation runs logged to Weights & Biases.")
    
    # Check if WandB API key is configured
    wandb_api_key = os.getenv('WANDB_API_KEY')
    
    if not wandb_api_key:
        st.warning("⚠️ WandB API Key not found. Add `WANDB_API_KEY` to your `.env` file to enable this feature.")
        st.code("WANDB_API_KEY=your_api_key_here", language="bash")
    else:
        try:
            # Initialize WandB API
            api = wandb.Api()
            
            # Get project runs (default project from simulation.py)
            project_name = "pytorch-simulation"
            
            st.info(f"📁 Fetching runs from project: **{project_name}**")
            
            if st.button("🔄 Refresh Run List"):
                st.session_state['wandb_runs_cache'] = None
            
            # Fetch or use cached runs
            if 'wandb_runs_cache' not in st.session_state or st.session_state['wandb_runs_cache'] is None:
                with st.spinner("Fetching runs from WandB..."):
                    try:
                        runs = api.runs(f"{api.default_entity}/{project_name}", per_page=50)
                        runs_list = []
                        for run in runs:
                            runs_list.append({
                                'id': run.id,
                                'name': run.name,
                                'state': run.state,
                                'created_at': run.created_at,
                                'summary': run.summary._json_dict if hasattr(run.summary, '_json_dict') else {}
                            })
                        st.session_state['wandb_runs_cache'] = runs_list
                    except Exception as e:
                        st.error(f"Error fetching runs: {e}")
                        st.session_state['wandb_runs_cache'] = []
            
            runs_list = st.session_state.get('wandb_runs_cache', [])
            
            if not runs_list:
                st.info("No runs found in this project yet. Run a simulation first!")
            else:
                st.success(f"Found {len(runs_list)} runs")
                
                # Create runs dataframe for display
                runs_df = pd.DataFrame([
                    {
                        'Name': r['name'],
                        'State': r['state'],
                        'Created': r['created_at'][:19] if r['created_at'] else 'N/A',
                        'Final Budget': r['summary'].get('budget', 'N/A'),
                        'Performance': r['summary'].get('performance', 'N/A')
                    }
                    for r in runs_list
                ])
                
                st.dataframe(runs_df, use_container_width=True)
                
                # Run selector
                run_names = [r['name'] for r in runs_list]
                selected_run_name = st.selectbox("Select a run to view details:", run_names)
                
                if selected_run_name and st.button("📈 Load Run Data"):
                    selected_run = next((r for r in runs_list if r['name'] == selected_run_name), None)
                    
                    if selected_run:
                        with st.spinner(f"Loading data for run: {selected_run_name}..."):
                            try:
                                run = api.run(f"{api.default_entity}/{project_name}/{selected_run['id']}")
                                history = run.history()
                                
                                if not history.empty:
                                    st.subheader(f"Run: {selected_run_name}")
                                    
                                    # Show summary metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Final Budget", f"{history['budget'].iloc[-1]:.2f}" if 'budget' in history.columns else "N/A")
                                    with col2:
                                        st.metric("Avg Performance", f"{history['performance'].mean():.2f}" if 'performance' in history.columns else "N/A")
                                    with col3:
                                        st.metric("Total Steps", len(history))
                                    
                                    # Plot key metrics
                                    if 'budget' in history.columns:
                                        fig_budget = px.line(history, y='budget', title="Budget Over Time")
                                        st.plotly_chart(fig_budget, use_container_width=True)
                                    
                                    if 'performance' in history.columns:
                                        fig_perf = px.line(history, y='performance', title="Performance Over Time")
                                        st.plotly_chart(fig_perf, use_container_width=True)
                                    
                                    if 'true_motor_temp' in history.columns:
                                        fig_temp = go.Figure()
                                        fig_temp.add_trace(go.Scatter(y=history['true_motor_temp'], name='Motor Temp'))
                                        if 'true_temp' in history.columns:
                                            fig_temp.add_trace(go.Scatter(y=history['true_temp'], name='Ambient Temp'))
                                        fig_temp.update_layout(title="Temperature Dynamics")
                                        st.plotly_chart(fig_temp, use_container_width=True)
                                    
                                    # Show raw data expander
                                    with st.expander("📋 View Raw Run Data"):
                                        st.dataframe(history)
                                else:
                                    st.warning("No history data available for this run.")
                                    
                            except Exception as e:
                                st.error(f"Error loading run data: {e}")
                
                # Link to WandB dashboard
                st.divider()
                entity = api.default_entity
                st.markdown(f"🔗 [Open full WandB Dashboard](https://wandb.ai/{entity}/{project_name})")
                
        except Exception as e:
            st.error(f"WandB API Error: {e}")
            st.info("Make sure your WANDB_API_KEY is valid and you have access to the project.")

