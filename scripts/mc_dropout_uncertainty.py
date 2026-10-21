import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib

# Impostazione seed NumPy per la riproducibilità
np.random.seed(42)

def run_mc_dropout(data_path="data/ndx_processed_data.csv", output_dir="results", n_iterations=100):
    import tensorflow as tf
    tf.random.set_seed(42)

    if os.path.exists("scripts/lstm_model.py"):
        if "scripts" not in sys.path:
            sys.path.append("scripts")
        from lstm_model import qlike_loss_tf, create_oos_sequences
    else:
        from lstm_model import qlike_loss_tf, create_oos_sequences

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
    
    # 5. Salvataggio CSV
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

    # 6. Visualizzazione sulla scala della Volatilità (Globale + Zoom Comparativo)
    plot_mc_dropout_volatility_full(df_out, output_dir=output_dir)
    plot_mc_dropout_regime_zoom(df_out, output_dir=output_dir)


def plot_mc_dropout_volatility_full(df_out, output_dir="results"):
    """
    Genera il grafico globale sulla scala della volatilità giornaliera (deviazione standard)
    con elevato contrasto cromatico tra la banda al 95% (blu cobalto/navy) e la banda al 90% (arancione).
    """
    dates = pd.to_datetime(df_out["Date"])
    realized_vol = np.sqrt(np.maximum(df_out["Realized_Proxy"].values, 0))
    mean_vol = np.sqrt(np.maximum(df_out["Mean_Pred"].values, 0))
    p_025_vol = np.sqrt(np.maximum(df_out["P_2.5"].values, 0))
    p_975_vol = np.sqrt(np.maximum(df_out["P_97.5"].values, 0))
    p_050_vol = np.sqrt(np.maximum(df_out["P_5.0"].values, 0))
    p_950_vol = np.sqrt(np.maximum(df_out["P_95.0"].values, 0))

    plt.figure(figsize=(15, 7))
    plt.plot(dates, realized_vol, label='Proxy Volatilità Osservata (|r_t|)', color='lightgray', alpha=0.55, linewidth=1.0, zorder=1)

    # Banda 95% (Code esterne 2.5%-5.0% e 95.0%-97.5%) in Blu Reale ad alta visibilità
    plt.fill_between(dates, p_025_vol, p_050_vol, color='#1d4ed8', alpha=0.80, label='Banda 95% (2.5% - 97.5%)', zorder=3)
    plt.fill_between(dates, p_950_vol, p_975_vol, color='#1d4ed8', alpha=0.80, zorder=3)
    plt.plot(dates, p_025_vol, color='#0f172a', linestyle='--', linewidth=0.95, alpha=0.95, zorder=4)
    plt.plot(dates, p_975_vol, color='#0f172a', linestyle='--', linewidth=0.95, alpha=0.95, zorder=4)

    # Banda 90% (Interna 5.0%-95.0%) in Arancione Ambra puro (senza sovrapposizione cromatica)
    plt.fill_between(dates, p_050_vol, p_950_vol, color='#f59e0b', alpha=0.65, label='Banda 90% (5.0% - 95.0%)', zorder=2)

    # Previsione Media
    plt.plot(dates, mean_vol, label='Previsione Media Volatilità (MC Dropout)', color='#991b1b', linewidth=1.4, zorder=5)

    plt.title('Stima dell\'Incertezza Epistemica sulla Volatilità Giornaliera (LSTM - MC Dropout)', fontsize=14, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Volatilità Giornaliera (Scala Deviazione Standard)', fontsize=12)
    plt.legend(loc='upper right', framealpha=0.92)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    vol_plot_path = os.path.join(output_dir, "lstm_mc_dropout_volatility_bands.png")
    plt.savefig(vol_plot_path, dpi=300)
    plt.close()
    print(f"Grafico Volatilità salvato in: {vol_plot_path}")


def plot_mc_dropout_regime_zoom(df_out, output_dir="results"):
    """
    Genera una figura comparativa a due pannelli affiancati (1x2) con asse Y condiviso
    e calibrato sulle bande predittive per confrontare:
      - Pannello A: Regime di Shock di Mercato (15 Marzo 2025 - 15 Giugno 2025, n=63 giorni di trading)
      - Pannello B: Regime di Calma Estrema (15 Luglio 2025 - 10 Ottobre 2025, n=63 giorni di trading)
    """
    df_z = df_out.copy()
    df_z["Date"] = pd.to_datetime(df_z["Date"])
    df_z["realized_vol"] = np.sqrt(np.maximum(df_z["Realized_Proxy"], 0))
    df_z["mean_vol"] = np.sqrt(np.maximum(df_z["Mean_Pred"], 0))
    df_z["p_025_vol"] = np.sqrt(np.maximum(df_z["P_2.5"], 0))
    df_z["p_975_vol"] = np.sqrt(np.maximum(df_z["P_97.5"], 0))
    df_z["p_050_vol"] = np.sqrt(np.maximum(df_z["P_5.0"], 0))
    df_z["p_950_vol"] = np.sqrt(np.maximum(df_z["P_95.0"], 0))
    df_z["width_95"] = df_z["p_975_vol"] - df_z["p_025_vol"]
    df_z["width_90"] = df_z["p_950_vol"] - df_z["p_050_vol"]

    regimes = [
        {
            "title": "(A) Regime di Shock — Aprile 2025 (Mar–Giu 2025)",
            "start": "2025-03-15",
            "end": "2025-06-15",
            "box_pos": (0.03, 0.95),
            "ha": "left"
        },
        {
            "title": "(B) Regime di Calma Estrema — Estate 2025 (Lug–Ott 2025)",
            "start": "2025-07-15",
            "end": "2025-10-10",
            "box_pos": (0.03, 0.95),
            "ha": "left"
        }
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    for ax, reg in zip(axes, regimes):
        sub = df_z[(df_z["Date"] >= reg["start"]) & (df_z["Date"] <= reg["end"])].copy()

        # Proxy osservata sullo sfondo (troncata visivamente dall'ylim focalizzato sulle bande)
        ax.plot(
            sub["Date"], sub["realized_vol"],
            label="Proxy Volatilità Osservata ($|r_t|$)",
            color="lightgray", alpha=0.65, linewidth=1.0, zorder=1
        )

        # Banda 95% (Code esterne 2.5%-5.0% e 95.0%-97.5%) in Blu Reale ad alta visibilità
        ax.fill_between(
            sub["Date"], sub["p_025_vol"], sub["p_050_vol"],
            color="#1d4ed8", alpha=0.80, label="Banda 95% (2.5% - 97.5%)", zorder=3
        )
        ax.fill_between(
            sub["Date"], sub["p_950_vol"], sub["p_975_vol"],
            color="#1d4ed8", alpha=0.80, zorder=3
        )
        ax.plot(sub["Date"], sub["p_025_vol"], color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)
        ax.plot(sub["Date"], sub["p_975_vol"], color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.95, zorder=4)

        # Banda 90% (Interna 5.0%-95.0%) in Arancione Ambra puro (senza sovrapposizione cromatica)
        ax.fill_between(
            sub["Date"], sub["p_050_vol"], sub["p_950_vol"],
            color="#f59e0b", alpha=0.65, label="Banda 90% (5.0% - 95.0%)", zorder=2
        )

        # Previsione Media
        ax.plot(
            sub["Date"], sub["mean_vol"],
            label="Previsione Media Volatilità (MC Dropout)",
            color="#991b1b", linewidth=1.5, zorder=5
        )

        # Statistiche sintetiche del regime
        mean_v = sub["mean_vol"].mean()
        mean_w95 = sub["width_95"].mean()
        max_w95 = sub["width_95"].max()
        mean_w90 = sub["width_90"].mean()

        stats_text = (
            f"Giorni di trading: $N = {len(sub)}$\n"
            f"Media $\\bar{{\\sigma}}_t$: ${mean_v:.4f}$ (${mean_v*100:.2f}\\%$)\n"
            f"Ampiezza media $\\Delta_{{90\\%}}$: ${mean_w90:.5f}$\n"
            f"Ampiezza media $\\Delta_{{95\\%}}$: ${mean_w95:.5f}$\n"
            f"Ampiezza max $\\Delta_{{95\\%}}$: ${max_w95:.5f}$"
        )
        ax.text(
            reg["box_pos"][0], reg["box_pos"][1], stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment=reg["ha"],
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="gray", alpha=0.92),
            zorder=6
        )

        ax.set_title(reg["title"], fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("Data", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Asse Y condiviso e calibrato sulle bande predittive per massimizzare la leggibilità
    axes[0].set_ylim(0.005, 0.0265)
    axes[0].set_ylabel("Volatilità Giornaliera (Scala Deviazione Standard $\\sigma_t$)", fontsize=11.5)
    axes[1].legend(loc="upper right", framealpha=0.92, fontsize=9.5)

    fig.suptitle(
        "Confronto Dinamica dell'Incertezza Epistemica (LSTM - MC Dropout): Regime di Shock vs Regime di Calma",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    zoom_plot_path = os.path.join(output_dir, "lstm_mc_dropout_volatility_zoom.png")
    plt.savefig(zoom_plot_path, dpi=300)
    plt.close()
    print(f"Grafico Zoom Volatilità salvato in: {zoom_plot_path}")


if __name__ == "__main__":
    run_mc_dropout()
