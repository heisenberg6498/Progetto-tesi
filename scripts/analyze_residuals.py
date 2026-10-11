import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_residuals(results_dir="results"):
    print("=== AVVIO ANALISI DEI RESIDUI E DELLE DISTANZE (ACCURATEZZA) ===")
    
    # 1. Caricamento Dati Previsioni
    garch_path = os.path.join(results_dir, "garch_predictions.csv")
    lstm_path = os.path.join(results_dir, "lstm_predictions.csv")
    
    if not os.path.exists(garch_path) or not os.path.exists(lstm_path):
        raise FileNotFoundError("File di previsioni mancanti. Assicurati che garch_model.py e lstm_model.py siano stati eseguiti.")
        
    df_garch = pd.read_csv(garch_path)
    df_lstm = pd.read_csv(lstm_path)
    
    df_garch['Date'] = pd.to_datetime(df_garch['Date'])
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
    
    # Unione sulle date per perfetto allineamento temporale
    df = pd.merge(df_garch[['Date', 'Squared_Residual_Proxy', 'GARCH_Forecast_Var']],
                  df_lstm[['Date', 'LSTM_Forecast_Var']],
                  on='Date', how='inner').sort_values('Date').reset_index(drop=True)
                  
    # 2. Conversione su scala Volatilità Giornaliera (Deviazione Standard)
    # Volatilità osservata reale (proxy: deviazione assoluta dal rendimento medio)
    vol_realized = np.sqrt(np.maximum(df['Squared_Residual_Proxy'], 0))
    vol_garch = np.sqrt(np.maximum(df['GARCH_Forecast_Var'], 0))
    vol_lstm = np.sqrt(np.maximum(df['LSTM_Forecast_Var'], 0))
    
    # 3. Calcolo degli Errori di Previsione (Residui: Reale - Previsto)
    err_garch = vol_realized - vol_garch
    err_lstm = vol_realized - vol_lstm
    
    # 4. Tabella Riassuntiva dell'Accuratezza in Percentuale
    stats = []
    for nome, err in [("GARCH(1,1)", err_garch), ("LSTM", err_lstm)]:
        bias = np.mean(err) * 100.0                       # Bias medio (se positivo: sottostima; se negativo: sovrastima)
        mae = np.mean(np.abs(err)) * 100.0                # Distanza media assoluta in punti percentuali
        median_ae = np.median(np.abs(err)) * 100.0        # Mediana della distanza assoluta
        rmse = np.sqrt(np.mean(err**2)) * 100.0           # RMSE in punti percentuali
        max_under = np.max(err) * 100.0                   # Errore massimo (picco inaspettato non previsto)
        max_over = np.min(err) * 100.0                    # Massima sovrastima del rischio
        std_err = np.std(err) * 100.0                     # Dispersione degli errori attorno alla media
        
        stats.append({
            "Modello": nome,
            "Distanza Media (MAE %)": f"{mae:.3f}%",
            "Mediana Errore Assoluto (%)": f"{median_ae:.3f}%",
            "RMSE (%)": f"{rmse:.3f}%",
            "Bias Medio (%)": f"{bias:.3f}%",
            "Deviazione Standard Errore (%)": f"{std_err:.3f}%",
            "Max Sottostima (%)": f"{max_under:.2f}%",
            "Max Sovrastima (%)": f"{max_over:.2f}%"
        })
        
    df_stats = pd.DataFrame(stats)
    out_table_path = os.path.join(results_dir, "analisi_accuratezza_residui.csv")
    df_stats.to_csv(out_table_path, index=False)
    
    print("\n--- METRICHE DI ACCURATEZZA E DISTANZA (SCALA VOLATILITÀ GIORNALIERA) ---")
    print(df_stats.to_string(index=False))
    print(f"\nTabella salvata con successo in: {out_table_path}")
    
    # 5. Generazione del Grafico Diagnostico a Due Pannelli
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2, 1.2]})
    
    # Definizione colori ad alto contrasto (Blue vs Orange)
    col_garch = '#1f77b4'  # Royal Blue
    col_lstm = '#ff7f0e'   # Vivid Orange
    
    # Pannello Superiore: Serie Storica dei Residui nel Tempo
    ax1.plot(df['Date'], err_garch * 100, label='Residui GARCH(1,1)', color=col_garch, alpha=0.7, linewidth=1.1)
    ax1.plot(df['Date'], err_lstm * 100, label='Residui LSTM', color=col_lstm, alpha=0.85, linewidth=1.1)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label='Errore Nullo (Previsione Perfetta)')
    ax1.set_title('Serie Storica degli Errori di Previsione Out-of-Sample ($e_t = \\text{Volatilità Reale} - \\text{Prevista}$)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Errore Giornaliero (%)', fontsize=11)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Pannello Inferiore: Istogramma della Distribuzione degli Errori
    bins = np.linspace(-4, 4, 60)
    ax2.hist(err_garch * 100, bins=bins, color=col_garch, alpha=0.5, label='Distribuzione Errori GARCH', density=True)
    ax2.hist(err_lstm * 100, bins=bins, color=col_lstm, alpha=0.5, label='Distribuzione Errori LSTM', density=True)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    ax2.axvline(np.mean(err_garch)*100, color=col_garch, linestyle=':', linewidth=1.8, label=f'Media GARCH ({np.mean(err_garch)*100:.2f}%)')
    ax2.axvline(np.mean(err_lstm)*100, color=col_lstm, linestyle=':', linewidth=1.8, label=f'Media LSTM ({np.mean(err_lstm)*100:.2f}%)')
    
    ax2.set_title('Distribuzione di Densità degli Errori (Confronto Centratura e Dispersione)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Errore di Previsione (%)', fontsize=11)
    ax2.set_ylabel('Densità', fontsize=11)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_plot_path = os.path.join(results_dir, "analisi_residui_garch_lstm.png")
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    
    print(f"Grafico diagnostico dei residui salvato in: {out_plot_path}")
    return df_stats

if __name__ == "__main__":
    analyze_residuals()
