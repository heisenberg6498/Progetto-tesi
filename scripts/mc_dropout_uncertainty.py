import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# Assumiamo di lanciare lo script dalla directory 'scripts' o root, aggiustiamo i path
import sys
# Aggiungiamo la directory scripts al path se lanciato dalla root
if os.path.exists("scripts/lstm_model.py"):
    sys.path.append("scripts")
    from lstm_model import qlike_loss_tf, create_oos_sequences
else:
    from lstm_model import qlike_loss_tf, create_oos_sequences

def run_mc_dropout(data_path="data/ndx_processed_data.csv", output_dir="results", n_iterations=100):
    print("=== AVVIO STIMA INCERTEZZA VIA MONTE CARLO DROPOUT ===")
    
    # 1. Caricamento Dati
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Log_Return", "Log_Range"]).sort_values("Date").reset_index(drop=True)
    
    total_len = len(df)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    
    mu_proxy_final = fit_df['Log_Return'].mean()
    scale_factor = 100.0
    variance_scale = scale_factor ** 2
    
    def process_features(dataset, mu_proxy):
        ds = dataset.copy()
        ds['Scaled_Return'] = ds['Log_Return'] * scale_factor
        ds['Squared_Residual_Proxy'] = ((ds['Log_Return'] - mu_proxy) * scale_factor) ** 2
        return ds
        
    fit_df_proc = process_features(fit_df, mu_proxy_final)
    test_df_proc = process_features(test_df, mu_proxy_final)
    
    # 2. Caricamento Scaler e Modello
    scaler_path = os.path.join(output_dir, "lstm_scaler.joblib")
    model_path = os.path.join(output_dir, "lstm_final_model.keras")
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Modello o Scaler non trovati. Esegui prima lstm_model.py per generarli.")
        
    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(model_path, custom_objects={"qlike_loss_tf": qlike_loss_tf})
    
    feature_cols = ['Scaled_Return', 'Squared_Residual_Proxy']
    target_col = 'Squared_Residual_Proxy'
    time_steps = 60
    
    fit_feat_scaled = scaler.transform(fit_df_proc[feature_cols])
    test_feat_scaled = scaler.transform(test_df_proc[feature_cols])
    
    X_test, _ = create_oos_sequences(fit_feat_scaled, test_feat_scaled, test_df_proc[target_col].values, time_steps)
    
    # 3. MC Dropout Inference
    print(f"\nGenerazione di {n_iterations} previsioni stocastiche con training=True (Dropout attivo)...")
    predictions = []
    for i in range(n_iterations):
        # Il flag training=True forza l'attivazione del Dropout layer anche in fase di inferenza
        y_pred_scaled = model(X_test, training=True).numpy().flatten()
        y_pred_dec = y_pred_scaled / variance_scale
        predictions.append(y_pred_dec)
        if (i+1) % 20 == 0:
            print(f"Iterazione {i+1}/{n_iterations} completata.")
            
    predictions = np.array(predictions) # shape: (100, N_test)
    
    # 4. Calcolo delle statistiche e dei percentili
    mean_pred = np.mean(predictions, axis=0)
    median_pred = np.median(predictions, axis=0)
    
    # 2.5% e 97.5% -> Intervallo di Confidenza al 95%
    p_025 = np.percentile(predictions, 2.5, axis=0)
    p_975 = np.percentile(predictions, 97.5, axis=0)
    
    # 5% e 95% -> Intervallo di Confidenza al 90%
    p_050 = np.percentile(predictions, 5.0, axis=0)
    p_950 = np.percentile(predictions, 95.0, axis=0)
    
    dates = pd.to_datetime(test_df['Date'])
    realized_var = ((test_df["Log_Return"].values - mu_proxy_final)) ** 2
    
    # 5. Visualizzazione Grafica
    plt.figure(figsize=(14, 7))
    plt.plot(dates, realized_var, label='Proxy Realizzata', color='lightgray', alpha=0.6)
    
    # Bande di confidenza
    plt.fill_between(dates, p_025, p_975, color='orange', alpha=0.2, label='Banda 95% (2.5% - 97.5%)')
    plt.fill_between(dates, p_050, p_950, color='red', alpha=0.3, label='Banda 90% (5.0% - 95.0%)')
    
    plt.plot(dates, mean_pred, label='Previsione Media (MC Dropout)', color='darkred', linewidth=1.5)
    
    plt.title('Stima dell\'Incertezza Epistemica via MC Dropout (LSTM)', fontsize=14, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Varianza Decimale', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "lstm_mc_dropout_bands.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nGrafico Varianza salvato in: {plot_path}")
    
    # 5.bis Visualizzazione sulla scala della Volatilità (Radice Quadrata)
    realized_vol = np.sqrt(np.maximum(realized_var, 0))
    mean_vol = np.sqrt(np.maximum(mean_pred, 0))
    p_025_vol = np.sqrt(np.maximum(p_025, 0))
    p_975_vol = np.sqrt(np.maximum(p_975, 0))
    p_050_vol = np.sqrt(np.maximum(p_050, 0))
    p_950_vol = np.sqrt(np.maximum(p_950, 0))

    plt.figure(figsize=(15, 7))
    plt.plot(dates, realized_vol, label='Proxy Volatilità Osservata (|r_t|)', color='lightgray', alpha=0.55, linewidth=1.0)
    
    # Banda 95% (Esterna) in Azzurro con bordi tratteggiati
    plt.fill_between(dates, p_025_vol, p_975_vol, color='lightblue', alpha=0.5, label='Banda 95% (2.5% - 97.5%)')
    plt.plot(dates, p_025_vol, color='deepskyblue', linestyle='--', linewidth=0.7, alpha=0.8)
    plt.plot(dates, p_975_vol, color='deepskyblue', linestyle='--', linewidth=0.7, alpha=0.8)
    
    # Banda 90% (Interna) in Arancione
    plt.fill_between(dates, p_050_vol, p_950_vol, color='orange', alpha=0.6, label='Banda 90% (5.0% - 95.0%)')
    
    # Previsione Media
    plt.plot(dates, mean_vol, label='Previsione Media Volatilità (MC Dropout)', color='darkred', linewidth=1.2)

    plt.title('Stima dell\'Incertezza Epistemica sulla Volatilità Giornaliera (LSTM - MC Dropout)', fontsize=14, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Volatilità Giornaliera (Scala Deviazione Standard)', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    vol_plot_path = os.path.join(output_dir, "lstm_mc_dropout_volatility_bands.png")
    plt.savefig(vol_plot_path, dpi=300)
    plt.close()
    print(f"Grafico Volatilità salvato in: {vol_plot_path}")
    
    # 6. Salvataggio CSV
    df_out = pd.DataFrame({
        "Date": dates,
        "Realized_Proxy": realized_var,
        "Mean_Pred": mean_pred,
        "Median_Pred": median_pred,
        "P_2.5": p_025,
        "P_5.0": p_050,
        "P_95.0": p_950,
        "P_97.5": p_975
    })
    csv_path = os.path.join(output_dir, "lstm_mc_dropout_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Risultati CSV salvati in: {csv_path}")

if __name__ == "__main__":
    run_mc_dropout()
