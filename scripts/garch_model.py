import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Impostazione del seed casuale per garantire la riproducibilità deterministica
np.random.seed(42)

def qlike_loss(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """
    Calcola la Quasi-Likelihood Loss (Patton, 2011) sulla scala della varianza.
    Formula: QLIKE = mean(log(h_t) + y_t / h_t)
    
    Regole econometriche e interpretative:
    - Valori inferiori indicano stime migliori.
    - Il valore della loss può risultare negativo quando la varianza è in unità decimali;
      ciò che conta è il confronto relativo tra modelli a parità di proxy e campione.
    - Non applica mascheramenti numerici (clip) per garantire l'integrità dei controlli.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    
    if not np.all(np.isfinite(realized_var)):
        raise ValueError("La proxy di varianza realizzata contiene valori NaN o infiniti.")
        
    if not np.all(np.isfinite(forecast_var)):
        raise ValueError("Le previsioni della varianza contengono valori NaN o infiniti.")
        
    if np.any(realized_var < 0):
        raise ValueError("La proxy di varianza contiene valori negativi non validi.")
        
    if np.any(forecast_var <= 0):
        raise ValueError("La varianza prevista dal modello deve essere strettamente positiva.")
        
    return float(np.mean(np.log(forecast_var) + (realized_var / forecast_var)))

def predictive_r2_oos(realized_var: np.ndarray, forecast_var: np.ndarray, benchmark_var: np.ndarray) -> float:
    """
    Calcola l'R-squared Out-of-Sample (R2_OOS_HistoricalMean) rispetto alla media storica ex-ante.
    Formula: R2_OOS = 1 - (SSE_modello / SSE_benchmark)
    
    Un valore negativo indica che il modello ha generato un errore quadratico superiore
    rispetto all'utilizzo della media storica ex-ante come previsione costante.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    benchmark_var = np.asarray(benchmark_var, dtype=float)
    
    if not (len(realized_var) == len(forecast_var) == len(benchmark_var)):
        raise ValueError("Le serie di input per R2_OOS devono avere la stessa lunghezza.")
        
    sse_model = np.sum((realized_var - forecast_var) ** 2)
    sse_benchmark = np.sum((realized_var - benchmark_var) ** 2)
    
    if sse_benchmark <= np.finfo(float).eps:
        return np.nan
        
    return float(1.0 - (sse_model / sse_benchmark))

def evaluate_predictions(realized_var: np.ndarray, forecast_var: np.ndarray, benchmark_var: np.ndarray) -> dict:
    """
    Calcola il set completo di metriche di valutazione sulla scala della varianza.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    benchmark_var = np.asarray(benchmark_var, dtype=float)

    if not (len(realized_var) == len(forecast_var) == len(benchmark_var)):
        raise ValueError("Proxy, previsioni e benchmark devono avere la stessa lunghezza.")

    if not np.all(np.isfinite(realized_var)):
        raise ValueError("La proxy contiene valori NaN o infiniti.")

    if not np.all(np.isfinite(forecast_var)):
        raise ValueError("Le previsioni contengono valori NaN o infiniti.")

    if not np.all(np.isfinite(benchmark_var)):
        raise ValueError("Il benchmark contiene valori NaN o infiniti.")

    if np.any(realized_var < 0):
        raise ValueError("La proxy di varianza contiene valori negativi.")

    if np.any(forecast_var <= 0):
        raise ValueError("La varianza prevista deve essere positiva.")

    if np.any(benchmark_var <= 0):
        raise ValueError("Il benchmark di varianza deve essere positivo.")

    realized_vol = np.sqrt(realized_var)
    forecast_vol = np.sqrt(forecast_var)

    return {
        "QLIKE": qlike_loss(realized_var, forecast_var),
        "MSE_var": mean_squared_error(realized_var, forecast_var),
        "MAE_var": mean_absolute_error(realized_var, forecast_var),
        "RMSE_vol": np.sqrt(mean_squared_error(realized_vol, forecast_vol)),
        "R2_OOS_HistoricalMean": predictive_r2_oos(realized_var, forecast_var, benchmark_var)
    }

def run_garch_pipeline(data_path="data/ndx_processed_data.csv", output_dir="results"):
    """
    Esegue la pipeline del modello GARCH(1,1) allineata sulla scala della varianza condizionale.
    La pipeline è progettata per evitare look-ahead bias nella valutazione sul test set
    mediante previsioni ricorsive one-step-ahead calcolate solo con informazioni ex-ante.
    """
    print("=== AVVIO PIPELINE GARCH(1,1) (RIGOROUS & REPRODUCIBLE) ===")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Impossibile trovare il dataset in {data_path}. Eseguire prima fetch_data.py.")
        
    df = pd.read_csv(data_path)
    
    # 1. Validazione preliminare e pulizia trasparente del dataset
    required_columns = ["Date", "Log_Return", "Log_Range"]
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel dataset: {sorted(missing_columns)}")
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.replace([np.inf, -np.inf], np.nan)
    
    missing_mask = df[required_columns].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        print(f"Attenzione: rimozione di {n_missing} righe contenenti valori mancanti (NaN/Inf).")
        
    df = df.loc[~missing_mask].sort_values('Date').reset_index(drop=True)
    
    if df['Date'].duplicated().any():
        duplicated_dates = df.loc[df['Date'].duplicated(), 'Date'].dt.strftime('%Y-%m-%d').tolist()
        raise ValueError(f"Il dataset contiene date duplicate nell'indice: {duplicated_dates}")
        
    if (df["Log_Range"] < 0).any():
        raise ValueError("Log_Range contiene valori negativi non validi. Verificare che sia log(High / Low).")
        
    total_len = len(df)
    if total_len < 100:
        raise ValueError("Numero di osservazioni insufficiente per la pipeline.")

    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    # 2. Suddivisione temporale del dataset (70% Train, 15% Validation, 15% Test)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Uno o più segmenti temporali risultano vuoti.")

    # Insieme combinato Train + Validation per il re-fit dei parametri definitivi del benchmark
    fit_df = pd.concat([train_df, val_df]).copy()
    
    print(f"Dataset caricato e validato correttamente:")
    print(f"  - Righe totali utili        : {total_len}")
    print(f"  - Train + Validation (Stima) : {len(fit_df)} osservazioni ({fit_df['Date'].min().strftime('%Y-%m-%d')} -> {fit_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  - Test Set (Valutazione)     : {len(test_df)} osservazioni ({test_df['Date'].min().strftime('%Y-%m-%d')} -> {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    # Media comune empirica usata per la proxy di valutazione (indipendente dai parametri del modello)
    mu_proxy_decimal = float(fit_df['Log_Return'].mean())
    print(f"Media empirica dei rendimenti (Train+Val proxy center): {mu_proxy_decimal:.8f}")
    
    # Scaliamo i rendimenti per 100 per la stima numerica ottimale in `arch`
    scale_factor = 100.0
    returns_fit = fit_df['Log_Return'] * scale_factor
    returns_test = test_df['Log_Return'] * scale_factor
    
    # 3. Fitting del modello GARCH(1,1) sul campione Train + Validation
    garch_spec = arch_model(returns_fit, mean='Constant', vol='Garch', p=1, q=1, dist='normal', rescale=False)
    garch_fit = garch_spec.fit(disp='off')
    
    if garch_fit.convergence_flag != 0:
        raise RuntimeError(f"La stima GARCH non è convergente. Convergence flag: {garch_fit.convergence_flag}")
        
    print("\n--- Summary del Modello GARCH(1,1) (Parametri Stimati su Train + Val) ---")
    print(garch_fit.summary())
    
    mu_garch_scaled = float(garch_fit.params['mu'])
    omega_scaled = float(garch_fit.params['omega'])
    alpha = float(garch_fit.params['alpha[1]'])
    beta = float(garch_fit.params['beta[1]'])
    
    if omega_scaled <= 0:
        raise ValueError(f"Parametro omega non positivo: {omega_scaled}")
    if alpha < 0 or beta < 0:
        raise ValueError(f"Parametri alpha ({alpha}) o beta ({beta}) negativi.")
        
    persistence = alpha + beta
    print(f"\nParametri Stimati: omega_scaled={omega_scaled:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")
    print(f"Persistenza della volatilità (alpha + beta): {persistence:.6f}")
    if persistence >= 1.0:
        print("Attenzione: alpha + beta >= 1. La volatilità stimata non è strettamente mean-reverting.")
        
    mu_garch_decimal = mu_garch_scaled / scale_factor
    
    # 4. Previsione ricorsiva 1-step-ahead sul Test Set (parametri fissi)
    last_variance_scaled = float(garch_fit.conditional_volatility.iloc[-1] ** 2)
    last_residual_scaled = float(returns_fit.iloc[-1] - mu_garch_scaled)
    
    expected_first_var_scaled = omega_scaled + alpha * (last_residual_scaled ** 2) + beta * last_variance_scaled
    expected_first_var_decimal = expected_first_var_scaled / (scale_factor ** 2)
    
    forecast_variances_decimal = []
    
    for observed_return_scaled in returns_test:
        next_variance_scaled = omega_scaled + alpha * (last_residual_scaled ** 2) + beta * last_variance_scaled
        
        # Conversione da percentuale^2 a decimale^2
        forecast_variances_decimal.append(next_variance_scaled / (scale_factor ** 2))
        
        # Aggiornamento dello stato per il giorno t+1 dopo aver osservato il rendimento al giorno t
        last_variance_scaled = next_variance_scaled
        last_residual_scaled = float(observed_return_scaled - mu_garch_scaled)
        
    # Verifica d'integrità e di allineamento temporale sulla prima previsione
    if not np.isclose(forecast_variances_decimal[0], expected_first_var_decimal, rtol=1e-10, atol=1e-14):
        raise RuntimeError(
            f"Errore di allineamento temporale nella prima previsione: {forecast_variances_decimal[0]} != {expected_first_var_decimal}"
        )
        
    if len(forecast_variances_decimal) != len(test_df):
        raise RuntimeError("Il numero di previsioni non coincide con il numero di osservazioni del test set.")

    test_results = test_df.copy()
    test_results['GARCH_Forecast_Var'] = forecast_variances_decimal
    test_results['GARCH_Forecast_Vol'] = np.sqrt(test_results['GARCH_Forecast_Var'])
    
    if not np.all(np.isfinite(test_results["GARCH_Forecast_Var"])):
        raise ValueError("Trovati valori NaN o infiniti nelle previsioni.")

    if not (test_results["GARCH_Forecast_Var"] > 0).all():
        raise ValueError("Trovate varianze previste non strettamente positive.")
    
    # 5. Costruzione delle Proxy di Varianza Osservabili (usando mu_proxy_decimal comune)
    test_results['Squared_Residual_Proxy'] = (test_results['Log_Return'] - mu_proxy_decimal) ** 2
    test_results['Parkinson_Var_Proxy'] = (test_results['Log_Range'] ** 2) / (4.0 * np.log(2.0))
    
    if not (test_results["Squared_Residual_Proxy"] >= 0).all():
        raise ValueError("La proxy dei residui quadratici contiene valori negativi.")

    if not (test_results["Parkinson_Var_Proxy"] >= 0).all():
        raise ValueError("La proxy di Parkinson contiene valori negativi.")
    
    # Benchmark storico ex-ante (media delle proxy calcolata su Train + Val)
    fit_sq_residuals = (fit_df['Log_Return'] - mu_proxy_decimal) ** 2
    fit_parkinson_var = (fit_df['Log_Range'] ** 2) / (4.0 * np.log(2.0))
    
    sq_benchmark = np.full(len(test_results), fit_sq_residuals.mean())
    parkinson_benchmark = np.full(len(test_results), fit_parkinson_var.mean())
    
    # 6. Calcolo delle metriche di Valutazione Out-of-Sample
    sq_metrics = evaluate_predictions(
        test_results['Squared_Residual_Proxy'],
        test_results['GARCH_Forecast_Var'],
        sq_benchmark
    )
    
    parkinson_metrics = evaluate_predictions(
        test_results['Parkinson_Var_Proxy'],
        test_results['GARCH_Forecast_Var'],
        parkinson_benchmark
    )

    sq_qlike_benchmark = qlike_loss(
        test_results["Squared_Residual_Proxy"],
        sq_benchmark
    )
    parkinson_qlike_benchmark = qlike_loss(
        test_results["Parkinson_Var_Proxy"],
        parkinson_benchmark
    )

    sq_metrics["QLIKE_Benchmark"] = sq_qlike_benchmark
    sq_metrics["Delta_QLIKE_vs_Benchmark"] = (
        sq_metrics["QLIKE"] - sq_qlike_benchmark
    )

    parkinson_metrics["QLIKE_Benchmark"] = (
        parkinson_qlike_benchmark
    )
    parkinson_metrics["Delta_QLIKE_vs_Benchmark"] = (
        parkinson_metrics["QLIKE"] - parkinson_qlike_benchmark
    )
    
    print("\n==================================================")
    print("--- RISULTATI VALUTAZIONE GARCH(1,1) (TEST SET) ---")
    print("Proxy primaria: residui quadratici")
    for key, value in sq_metrics.items():
        print(f"  - {key:<28}: {value:.10e}")
        
    print("\nProxy secondaria: varianza di Parkinson")
    for key, value in parkinson_metrics.items():
        print(f"  - {key:<28}: {value:.10e}")
    print("==================================================")
    
    # 7. Salvataggio dei Risultati e dei Metadati dell'Esperimento
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics_df = pd.DataFrame([
        {"Proxy": "Squared_Residual", **sq_metrics},
        {"Proxy": "Parkinson_Var", **parkinson_metrics}
    ])
    
    metrics_path = os.path.join(output_dir, "garch_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    predictions_path = os.path.join(output_dir, "garch_predictions.csv")
    test_results.to_csv(predictions_path, index=False)
    
    metadata_df = pd.DataFrame([{
        "Model": "GARCH(1,1)",
        "Mean_Specification": "Constant",
        "Volatility_Specification": "GARCH(1,1)",
        "Distribution": "Normal",
        "Scale_Factor": scale_factor,
        "Total_Observations": total_len,
        "Fit_Observations": len(fit_df),
        "Train_Observations": len(train_df),
        "Validation_Observations": len(val_df),
        "Test_Observations": len(test_df),
        "Benchmark_Type": "Fixed Train+Validation Historical Mean",
        "Primary_Proxy": "Squared centered return",
        "Secondary_Proxy": "Parkinson variance",
        "Train_Start": train_df['Date'].min().strftime('%Y-%m-%d'),
        "Train_End": train_df['Date'].max().strftime('%Y-%m-%d'),
        "Validation_Start": val_df['Date'].min().strftime('%Y-%m-%d'),
        "Validation_End": val_df['Date'].max().strftime('%Y-%m-%d'),
        "Test_Start": test_df['Date'].min().strftime('%Y-%m-%d'),
        "Test_End": test_df['Date'].max().strftime('%Y-%m-%d'),
        "Mu_GARCH_Scaled": mu_garch_scaled,
        "Mu_GARCH_Decimal": mu_garch_decimal,
        "Mu_Proxy_Decimal": mu_proxy_decimal,
        "Omega_Scaled": omega_scaled,
        "Alpha": alpha,
        "Beta": beta,
        "Persistence": persistence,
        "Execution_Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    
    metadata_path = os.path.join(output_dir, "garch_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    # 8. Grafico delle Previsioni con Terminologia Corretta
    plt.figure(figsize=(14, 6))
    plt.plot(test_results['Date'], test_results['Squared_Residual_Proxy'], label='Proxy primaria: residuo quadratico (r_t - mu_proxy)^2', color='lightgray', alpha=0.6)
    plt.plot(test_results['Date'], test_results['Parkinson_Var_Proxy'], label='Proxy secondaria: varianza di Parkinson (High-Low)', color='orange', alpha=0.5)
    plt.plot(test_results['Date'], test_results['GARCH_Forecast_Var'], label='Previsione GARCH(1,1) della varianza', color='navy', linewidth=1.5)
    
    plt.title('Previsione Out-of-Sample della Varianza NASDAQ 100 - Modello GARCH(1,1)', fontsize=13, fontweight='bold')
    plt.xlabel('Data', fontsize=11)
    plt.ylabel('Varianza Giornaliera', fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "garch_variance_forecast.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nMetriche salvate in: {metrics_path}")
    print(f"Previsioni salvate in: {predictions_path}")
    print(f"Metadati salvati in: {metadata_path}")
    print(f"Grafico salvato in: {plot_path}")
    
    return metrics_df, test_results

if __name__ == "__main__":
    run_garch_pipeline()