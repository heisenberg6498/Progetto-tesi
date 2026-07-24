import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Impostazione del seed casuale per garantire la riproducibilità deterministica
np.random.seed(42)

def run_garch_pipeline(data_path="data/ndx_processed_data.csv", output_dir="results"):
    """
    Esegue la pipeline completa per il modello GARCH(1,1) sul NASDAQ 100:
    1. Caricamento e suddivisione temporale dei dati (70% Train, 15% Validation, 15% Test).
    2. Stima del modello GARCH(1,1) sul Training set tramite Massima Verosimiglianza.
    3. Previsione out-of-sample in modalita Walk-Forward (1-step-ahead).
    4. Valutazione delle performance (R^2, MAPE, RMSE) e salvataggio dei risultati/grafici.
    """
    print("=== AVVIO PIPELINE MODELLO GARCH(1,1) ===")
    
    # 1. Caricamento del dataset processato
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Impossibile trovare il dataset in {data_path}. Eseguire prima fetch_data.py.")
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    total_len = len(df)
    print(f"Dataset caricato correttamente. Righe totali: {total_len}")
    
    # 2. Suddivisione temporale del dataset (70% Train, 15% Validation, 15% Test)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    test_size = total_len - train_size - val_size
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    print(f"Suddivisione temporale completata:")
    print(f"  - Training Set   : {len(train_df)} osservazioni ({train_df['Date'].min().strftime('%Y-%m-%d')} -> {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  - Validation Set : {len(val_df)} osservazioni ({val_df['Date'].min().strftime('%Y-%m-%d')} -> {val_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"  - Test Set       : {len(test_df)} osservazioni ({test_df['Date'].min().strftime('%Y-%m-%d')} -> {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    # Scaliamo i rendimenti per 100 per stabilizzare l'ottimizzatore numerico MLE della libreria `arch`
    scale_factor = 100.0
    returns_train = train_df['Log_Return'] * scale_factor
    returns_full = df['Log_Return'] * scale_factor
    
    # 3. Definizioni e Fitting del Modello GARCH(1,1) sul Training Set
    # Specifica standard GARCH(1,1): media costante, varianza condizionale GARCH, distribuzione Normale
    print("\nAdattamento del modello GARCH(1,1) sul Training set...")
    garch_spec = arch_model(returns_train, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
    garch_fit = garch_spec.fit(disp='off')
    
    print("\n--- Summary del Modello GARCH(1,1) ---")
    print(garch_fit.summary())
    
    # Extract parameters for reference
    omega = garch_fit.params['omega']
    alpha = garch_fit.params['alpha[1]']
    beta = garch_fit.params['beta[1]']
    print(f"\nParametri Stimati: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")
    print(f"Persistenza della volatilità (alpha + beta): {alpha + beta:.6f}")
    
    # 4. Previsione Walk-Forward Out-of-Sample sul Test Set
    # Inizializziamo il vettore per memorizzare la volatilità condizionale prevista (sigma_t)
    print("\nCalcolo delle previsioni Walk-Forward out-of-sample...")
    
    # Utilizziamo la funzione fix=True per mantenere i parametri stimati sul train
    # ricalcolando la varianza condizionale man mano che arrivano le nuove osservazioni
    full_garch = arch_model(returns_full, mean='Constant', vol='Garch', p=1, q=1, dist='normal')
    full_fit = full_garch.fix(garch_fit.params)
    
    # Estraiamo la serie completa della varianza condizionale stimata
    cond_volatility_scaled = np.sqrt(full_fit.conditional_volatility ** 2)
    # Riportiamo la volatilità condizionale sulla scala originale del rendimento logaritmico (dividendo per 100)
    cond_volatility = cond_volatility_scaled / scale_factor
    
    # Assegniamo la volatilità stimata al dataframe originale
    df['GARCH_Cond_Vol'] = cond_volatility
    df['GARCH_Cond_Var'] = (cond_volatility) ** 2
    
    # Definiamo la proxy di volatilità reale osservata:
    # Usiamo sia il valore assoluto dei rendimenti logaritmici |r_t| sia la Parkinson Volatility
    df['Actual_Vol_Proxy'] = np.abs(df['Log_Return'])
    df['Parkinson_Vol_Proxy'] = df['Log_Range'] / np.sqrt(4 * np.log(2))
    
    # Separiamo il test set per la valutazione delle metriche
    test_results = df.iloc[train_size + val_size:].copy()
    
    y_true = test_results['Actual_Vol_Proxy'].values
    y_pred = test_results['GARCH_Cond_Vol'].values
    
    # 5. Calcolo delle Metriche di Valutazione (R-squared, MAPE, RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Calcolo aggiuntivo rispetto alla Parkinson Volatility per robustezza accademica
    y_true_parkinson = test_results['Parkinson_Vol_Proxy'].values
    rmse_parkinson = np.sqrt(mean_squared_error(y_true_parkinson, y_pred))
    mape_parkinson = mean_absolute_percentage_error(y_true_parkinson, y_pred) * 100
    r2_parkinson = r2_score(y_true_parkinson, y_pred)
    
    print("\n==================================================")
    print("--- RISULTATI VALUTAZIONE MODELLO GARCH(1,1) (TEST SET) ---")
    print(f"Target Proxy: Rendimento Assoluto |r_t|")
    print(f"  - RMSE : {rmse:.6f}")
    print(f"  - MAPE : {mape:.2f}%")
    print(f"  - R^2  : {r2:.6f}")
    print("\nTarget Proxy: Volatilità di Parkinson (High-Low Range)")
    print(f"  - RMSE : {rmse_parkinson:.6f}")
    print(f"  - MAPE : {mape_parkinson:.2f}%")
    print(f"  - R^2  : {r2_parkinson:.6f}")
    print("==================================================")
    
    # 6. Salvataggio dei Risultati e Metriche
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics_df = pd.DataFrame({
        'Model': ['GARCH(1,1) (|r_t| proxy)', 'GARCH(1,1) (Parkinson proxy)'],
        'RMSE': [rmse, rmse_parkinson],
        'MAPE (%)': [mape, mape_parkinson],
        'R2': [r2, r2_parkinson]
    })
    
    metrics_path = os.path.join(output_dir, "garch_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    predictions_path = os.path.join(output_dir, "garch_predictions.csv")
    df.to_csv(predictions_path, index=False)
    
    print(f"\nMetriche salvate in: {metrics_path}")
    print(f"Previsioni salvate in: {predictions_path}")
    
    # 7. Generazione del Grafico delle Previsioni di Volatilità
    plt.figure(figsize=(14, 7))
    plt.plot(test_results['Date'], test_results['Actual_Vol_Proxy'], label='Volatilità Osservata (|r_t|)', color='lightgray', alpha=0.7)
    plt.plot(test_results['Date'], test_results['Parkinson_Vol_Proxy'], label='Volatilità Parkinson (Range High-Low)', color='lightblue', alpha=0.7)
    plt.plot(test_results['Date'], test_results['GARCH_Cond_Vol'], label='Stima Volatilità Condizionale GARCH(1,1)', color='navy', linewidth=1.5)
    
    plt.title('Previsione Out-of-Sample della Volatilità NASDAQ 100 - Modello GARCH(1,1)', fontsize=14, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Volatilità Giornaliera', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "garch_volatility_forecast.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Grafico delle previsioni salvato in: {plot_path}")
    
    return metrics_df, test_results

if __name__ == "__main__":
    run_garch_pipeline()
