import pandas as pd
import numpy as np
import os

try:
    df = pd.read_csv('simulation_log.csv')
    
    initial_budget = df['budget'].iloc[0]
    final_budget = df['budget'].iloc[-1]
    
    avg_performance = df['performance'].mean()
    avg_motor_temp = df['true_motor_temp'].mean()
    
    # Investment analysis (still relevant for context, even if commented out)
    investment_made = df['investment_done'].any() # Check if investment was made at any point
    investment_step = df[df['investment_done'] == True]['step'].min() if investment_made else -1
    
    # Verification analysis
    verification_triggered = df['is_verifying_sensor'].any()
    # Count starts of verification by looking for steps where verification_paused is at its max value
    num_verifications = len(df[df['verification_paused'] == df['verification_paused'].max()]['step']) if verification_triggered else 0
    
    print('--- Analisi dei Risultati dell\'ultima Simulazione ---')
    print(f'- Budget Iniziale: {initial_budget:.2f}')
    print(f'- Budget Finale: {final_budget:.2f}')
    print(f'- Variazione Budget: {final_budget - initial_budget:.2f}')
    print(f'- Performance Media Totale: {avg_performance:.2f}')
    print(f'- Temperatura Media Motore: {avg_motor_temp:.2f}°C')
    
    if investment_made:
        print(f'- Investimento Epistemico Effettuato allo Step: {investment_step}')
        downtime_duration_invest = df['production_paused'].max() 
        downtime_end_step_invest = investment_step + downtime_duration_invest
        df_pre_invest = df[df['step'] < investment_step]
        df_post_invest = df[df['step'] > downtime_end_step_invest]
        if not df_pre_invest.empty:
            avg_perf_pre = df_pre_invest['performance'].mean()
            print(f'  - Performance Media Pre-Investimento: {avg_perf_pre:.2f}')
        if not df_post_invest.empty:
            avg_perf_post = df_post_invest['performance'].mean()
            print(f'  - Performance Media Post-Investimento (dopo downtime): {avg_perf_post:.2f}')
            if not df_pre_invest.empty and not df_post_invest.empty:
                print(f'  - Miglioramento Performance: {avg_perf_post - avg_perf_pre:.2f}')
    else:
        print('- Nessun Investimento Epistemico Effettuato.')

    print('\n--- Analisi Azione Epistemica di Verifica ---')
    if verification_triggered:
        print(f'- Numero di Verifiche Avviate: {num_verifications}')
        # Assumiamo che verification_cost e verification_downtime siano costanti e presi dal primo step
        verification_cost_per_action = df['verification_cost'].iloc[0] if 'verification_cost' in df.columns else 50.0 # Default se non loggato
        verification_downtime_per_action = df['verification_downtime'].iloc[0] if 'verification_downtime' in df.columns else 10 # Default se non loggato
        
        total_verification_cost = num_verifications * verification_cost_per_action
        total_verification_downtime = num_verifications * verification_downtime_per_action
        print(f'- Costo Totale Verifiche (Budget): {total_verification_cost:.2f}')
        print(f'- Downtime Totale per Verifiche (Step): {total_verification_downtime}')
        
        # Check budget impact
        print(f'- Budget Iniziale: {initial_budget:.2f}, Budget Finale: {final_budget:.2f}')
        
        if df['budget'].min() <= 0:
            print(f'- ATTENZIONE: Il budget è sceso a {df["budget"].min():.2f} o meno durante la simulazione.')
    else:
        print('- Nessuna Verifica Sensore Avviata.')

    print('\n--- Analisi Costi Operativi ---')
    total_fixed_cost = df['fixed_operational_cost'].sum() if 'fixed_operational_cost' in df.columns else 0
    total_production_cost = df['cost_of_production'].sum() if 'cost_of_production' in df.columns else 0
    total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
    print(f'- Costo Operativo Fisso Totale: {total_fixed_cost:.2f}')
    print(f'- Costo di Produzione Totale: {total_production_cost:.2f}')
    print(f'- Ricavi Totali (Revenue): {total_revenue:.2f}')


except FileNotFoundError:
    print('File simulation_log.csv non trovato. Assicurati che la simulazione sia stata eseguita.')
except Exception as e:
    print(f"Si è verificato un errore durante l\'analisi: {e}")