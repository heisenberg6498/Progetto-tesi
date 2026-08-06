import os
import pandas as pd
import numpy as np
import time

from lstm_model import run_lstm_pipeline
from hybrid_model import run_hybrid_pipeline

def evaluate_seeds(seeds, output_dir="results/robustness"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    lstm_results = []
    hybrid_results = []
    
    print(f"==================================================")
    print(f"AVVIO VERIFICA DI ROBUSTEZZA (Multi-Seed)")
    print(f"Seeds: {seeds}")
    print(f"==================================================\n")
    
    for seed in seeds:
        print(f"\n**************************************************")
        print(f"                ESECUZIONE SEED {seed}            ")
        print(f"**************************************************")
        
        # 1. Eseguiamo l'LSTM Standalone
        lstm_metrics, _, _ = run_lstm_pipeline(output_dir=os.path.join(output_dir, f"lstm_seed_{seed}"), seed=seed)
        lstm_metrics['Seed'] = seed
        lstm_results.append(lstm_metrics)
        
        # 2. Eseguiamo il Modello Ibrido
        hybrid_metrics, _ = run_hybrid_pipeline(output_dir=os.path.join(output_dir, f"hybrid_seed_{seed}"), seed=seed)
        hybrid_metrics['Seed'] = seed
        hybrid_results.append(hybrid_metrics)
        
    # Aggreghiamo i risultati
    df_lstm = pd.concat(lstm_results, ignore_index=True)
    df_hybrid = pd.concat(hybrid_results, ignore_index=True)
    
    # Calcolo delle statistiche
    def aggregate_metrics(df, model_name):
        agg_list = []
        for proxy in ["Squared_Residual", "Parkinson_Var"]:
            subset = df[df['Proxy'] == proxy]
            metrics_summary = {"Model": model_name, "Proxy": proxy}
            for col in ['QLIKE', 'MSE_var', 'MAE_var', 'RMSE_vol', 'R2_OOS_HistoricalMean']:
                mean_val = subset[col].mean()
                std_val = subset[col].std()
                # Formattiamo come "Media +/- Std"
                if 'var' in col and col != 'R2_OOS_HistoricalMean':
                    # Notazione scientifica per varianze piccole
                    metrics_summary[col] = f"{mean_val:.2e} ± {std_val:.2e}"
                elif col == 'R2_OOS_HistoricalMean':
                    metrics_summary[col] = f"{(mean_val*100):.2f}% ± {(std_val*100):.2f}%"
                else:
                    metrics_summary[col] = f"{mean_val:.4f} ± {std_val:.4f}"
            agg_list.append(metrics_summary)
        return pd.DataFrame(agg_list)
        
    summary_lstm = aggregate_metrics(df_lstm, "LSTM_Standalone")
    summary_hybrid = aggregate_metrics(df_hybrid, "Hybrid_GARCH_LSTM")
    
    final_summary = pd.concat([summary_lstm, summary_hybrid], ignore_index=True)
    
    summary_path = os.path.join(output_dir, "multi_seed_robustness_summary.csv")
    final_summary.to_csv(summary_path, index=False)
    
    print(f"\n==================================================")
    print(f"RISULTATI FINALI ROBUSTEZZA ({len(seeds)} Seeds)")
    print(f"==================================================")
    print(final_summary.to_string(index=False))
    print(f"\nI dettagli completi sono stati salvati in: {summary_path}")

if __name__ == "__main__":
    seeds_to_test = [11, 22, 33, 42, 55]
    evaluate_seeds(seeds_to_test)
