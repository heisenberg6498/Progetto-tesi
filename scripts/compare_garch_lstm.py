import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_models(results_dir="results"):
    print("=== CONFRONTO FINALE: GARCH(1,1) vs LSTM ===")
    
    # 1. Caricamento Metriche
    garch_metrics_path = os.path.join(results_dir, "garch_metrics.csv")
    lstm_metrics_path = os.path.join(results_dir, "lstm_metrics.csv")
    
    if not os.path.exists(garch_metrics_path) or not os.path.exists(lstm_metrics_path):
        raise FileNotFoundError("File delle metriche mancanti. Assicurati di aver eseguito sia garch_model.py che lstm_model.py")
        
    garch_metrics = pd.read_csv(garch_metrics_path)
    lstm_metrics = pd.read_csv(lstm_metrics_path)
    
    garch_metrics['Modello'] = 'GARCH(1,1)'
    lstm_metrics['Modello'] = 'LSTM'
    
    # Uniamo i dataset
    combined_metrics = pd.concat([garch_metrics, lstm_metrics], ignore_index=True)
    
    # Riorganizziamo le colonne per chiarezza
    cols = ['Modello', 'Proxy', 'R2_OOS_HistoricalMean', 'MAE_var', 'RMSE_vol', 'QLIKE', 'MSE_var']
    combined_metrics = combined_metrics[cols]
    
    # Formattazione per la stampa
    print("\n--- TABELLA COMPARATIVA DELLE METRICHE ---")
    
    # Filtriamo per la proxy principale (Residuo Quadratico) per il commento testuale
    sq_metrics = combined_metrics[combined_metrics['Proxy'] == 'Squared_Residual']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(combined_metrics.to_string(index=False))
    
    # Salvataggio Tabella
    out_path = os.path.join(results_dir, "confronto_garch_lstm_metriche.csv")
    combined_metrics.to_csv(out_path, index=False)
    print(f"\nTabella comparativa salvata in: {out_path}")
    
    # 2. Generazione Grafico Comparativo
    garch_preds_path = os.path.join(results_dir, "garch_predictions.csv")
    lstm_preds_path = os.path.join(results_dir, "lstm_predictions.csv")
    
    if os.path.exists(garch_preds_path) and os.path.exists(lstm_preds_path):
        garch_preds = pd.read_csv(garch_preds_path)
        lstm_preds = pd.read_csv(lstm_preds_path)
        
        garch_preds['Date'] = pd.to_datetime(garch_preds['Date'])
        lstm_preds['Date'] = pd.to_datetime(lstm_preds['Date'])
        
        # Merge sulle date per allineamento perfetto
        merged_preds = pd.merge(garch_preds[['Date', 'Squared_Residual_Proxy', 'GARCH_Forecast_Var']], 
                                lstm_preds[['Date', 'LSTM_Forecast_Var']], 
                                on='Date', how='inner')
                                
        dates = merged_preds['Date']
        realized = merged_preds['Squared_Residual_Proxy']
        garch_var = merged_preds['GARCH_Forecast_Var']
        lstm_var = merged_preds['LSTM_Forecast_Var']
        
        plt.figure(figsize=(15, 8))
        
        # Subplot 1: GARCH
        plt.subplot(2, 1, 1)
        plt.plot(dates, realized, color='lightgray', label='Proxy Realizzata (Residuo Quadratico)', alpha=0.6)
        plt.plot(dates, garch_var, color='midnightblue', linewidth=1.5, label='Previsione GARCH(1,1)')
        plt.title('GARCH(1,1) - Previsione Out-of-Sample', fontweight='bold')
        plt.ylabel('Varianza Decimale')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Subplot 2: LSTM
        plt.subplot(2, 1, 2)
        plt.plot(dates, realized, color='lightgray', label='Proxy Realizzata (Residuo Quadratico)', alpha=0.6)
        plt.plot(dates, lstm_var, color='darkred', linewidth=1.5, label='Previsione LSTM')
        plt.title('LSTM - Previsione Out-of-Sample', fontweight='bold')
        plt.ylabel('Varianza Decimale')
        plt.xlabel('Data')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "confronto_grafico_garch_lstm.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Grafico comparativo salvato in: {plot_path}")

if __name__ == "__main__":
    compare_models()
