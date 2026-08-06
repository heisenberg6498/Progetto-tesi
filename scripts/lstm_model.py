import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib

# ==============================================================================
# CONFIGURAZIONE INIZIALE E RIPRODUCIBILITA'
# ==============================================================================
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

def qlike_loss_tf(y_true, y_pred):
    """QLIKE loss sulla varianza scalata, con protezione forme."""
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    
    eps = tf.cast(tf.keras.backend.epsilon(), y_pred.dtype)
    y_pred_safe = tf.maximum(y_pred, eps)
    
    loss = tf.math.log(y_pred_safe) + y_true / y_pred_safe
    return tf.reduce_mean(loss, axis=-1)

def qlike_loss_np(realized_var, forecast_var):
    realized_var = np.asarray(realized_var, dtype=float).flatten()
    forecast_var = np.asarray(forecast_var, dtype=float).flatten()
    
    if not np.all(np.isfinite(realized_var)) or not np.all(np.isfinite(forecast_var)):
        raise ValueError("NaN o Inf presenti in proxy o forecast.")
    if np.any(realized_var < 0):
        raise ValueError("La proxy di varianza non può essere negativa.")
    if np.any(forecast_var <= 0):
        raise ValueError("La varianza prevista deve essere positiva.")
        
    return float(np.mean(np.log(forecast_var) + realized_var / forecast_var))

def create_oos_sequences(history_features, current_features, current_target, time_steps):
    if len(history_features) < time_steps:
        raise ValueError("Lo storico precedente è più corto del look-back.")
        
    combined_features = np.vstack([history_features[-time_steps:], current_features])
    X = np.stack([combined_features[i:i + time_steps] for i in range(len(current_features))])
    y = np.asarray(current_target, dtype=np.float32).reshape(-1, 1)
    return X, y

def build_model(time_steps, n_features):
    model = Sequential([
        LSTM(32, input_shape=(time_steps, n_features), return_sequences=False, name="LSTM_Layer"),
        Dropout(0.20, name="Dropout_Regularization"),
        Dense(1, activation='softplus', name="Output_Variance")
    ])
    model.compile(optimizer='adam', loss=qlike_loss_tf, metrics=['mse', 'mae'])
    return model

def calculate_metrics(realized_dec, forecast_dec, benchmark_history_dec):
    """
    Calcola le metriche rispetto a un benchmark costante pari
    alla media storica della proxy su Train + Validation.
    """
    realized_dec = np.asarray(realized_dec, dtype=float).flatten()
    forecast_dec = np.asarray(forecast_dec, dtype=float).flatten()
    benchmark_history_dec = np.asarray(benchmark_history_dec, dtype=float).flatten()

    if len(realized_dec) != len(forecast_dec):
        raise ValueError("Proxy realizzata e previsioni devono avere la stessa lunghezza.")
    if not np.all(np.isfinite(realized_dec)):
        raise ValueError("La proxy realizzata contiene NaN o Inf.")
    if not np.all(np.isfinite(forecast_dec)):
        raise ValueError("Le previsioni contengono NaN o Inf.")
    if not np.all(np.isfinite(benchmark_history_dec)):
        raise ValueError("Lo storico del benchmark contiene NaN o Inf.")
    if np.any(realized_dec < 0):
        raise ValueError("La proxy di varianza non può essere negativa.")
    if np.any(forecast_dec <= 0):
        raise ValueError("La varianza prevista deve essere positiva.")
    if np.any(benchmark_history_dec < 0):
        raise ValueError("Lo storico della proxy contiene valori negativi.")

    benchmark_mean = float(np.mean(benchmark_history_dec))
    if benchmark_mean <= 0:
        raise ValueError("La media storica del benchmark deve essere positiva.")

    benchmark_array = np.full(realized_dec.shape, benchmark_mean, dtype=float)

    mse_val = np.mean((realized_dec - forecast_dec) ** 2)
    mae_val = np.mean(np.abs(realized_dec - forecast_dec))
    qlike_val = qlike_loss_np(realized_dec, forecast_dec)
    rmse_vol = np.sqrt(np.mean((np.sqrt(realized_dec) - np.sqrt(forecast_dec)) ** 2))

    sse_model = np.sum((realized_dec - forecast_dec) ** 2)
    sse_bench = np.sum((realized_dec - benchmark_array) ** 2)

    if sse_bench <= np.finfo(float).eps:
        r2_oos = np.nan
    else:
        r2_oos = 1.0 - sse_model / sse_bench

    qlike_bench = qlike_loss_np(realized_dec, benchmark_array)
    delta_qlike = qlike_val - qlike_bench

    return qlike_val, mse_val, mae_val, rmse_vol, r2_oos, qlike_bench, delta_qlike

def run_lstm_pipeline(data_path="data/ndx_processed_data.csv", output_dir="results", seed=42):
    print(f"=== AVVIO PIPELINE LSTM (Seed: {seed}) ===")
    
    # ==============================================================================
    # CARICAMENTO DATI E CONTROLLI ROBUSTI
    # ==============================================================================
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset non trovato: {data_path}")
        
    df = pd.read_csv(data_path)
    required_columns = ["Date", "Log_Return", "Log_Range"]
    missing_columns = set(required_columns).difference(df.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti: {sorted(missing_columns)}")
        
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns).sort_values("Date").reset_index(drop=True)
    if df["Date"].duplicated().any():
        raise ValueError("Il dataset contiene date duplicate.")
        
    if (df["Log_Range"] < 0).any():
        raise ValueError("Log_Range contiene valori negativi. Verificare che sia definito come log(High / Low).")
    if len(df) < 200:
        raise ValueError("Il numero di osservazioni è insufficiente.")
        
    total_len = len(df)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    train_df_original = df.iloc[:train_size].copy()
    val_df_original = df.iloc[train_size:train_size + val_size].copy()
    test_df_original = df.iloc[train_size + val_size:].copy()
    
    if train_df_original.empty or val_df_original.empty or test_df_original.empty:
        raise ValueError("Uno o più segmenti temporali sono vuoti.")
        
    time_steps = 60
    if len(train_df_original) <= time_steps:
        raise ValueError("Il training set è più corto del look-back.")
            
    scale_factor = 100.0
    variance_scale = scale_factor ** 2
    feature_cols = ['Scaled_Return', 'Squared_Residual_Proxy']
    target_col = 'Squared_Residual_Proxy'
    
    def process_features(dataset, mu_proxy):
        dataset = dataset.copy()
        dataset['Scaled_Return'] = dataset['Log_Return'] * scale_factor
        dataset['Squared_Residual_Proxy'] = ((dataset['Log_Return'] - mu_proxy) * scale_factor) ** 2
        return dataset
        
    # ==============================================================================
    # FASE 1: Selezione Best Epoch su Validation Set
    # ==============================================================================
    mu_proxy_train = train_df_original['Log_Return'].mean()
    train_df = process_features(train_df_original, mu_proxy_train)
    val_df = process_features(val_df_original, mu_proxy_train)
    
    scaler1 = StandardScaler()
    train_feat_scaled = scaler1.fit_transform(train_df[feature_cols])
    val_feat_scaled = scaler1.transform(val_df[feature_cols])
    
    X_train, y_train = [], []
    for i in range(time_steps, len(train_feat_scaled)):
        X_train.append(train_feat_scaled[i - time_steps:i])
        y_train.append(train_df[target_col].values[i])
    X_train = np.array(X_train)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    
    X_val, y_val = create_oos_sequences(train_feat_scaled, val_feat_scaled, val_df[target_col].values, time_steps)
    
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model1 = build_model(time_steps, len(feature_cols))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    print("\n--- FASE 1: Addestramento e Selezione Best Epoch ---")
    history = model1.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], shuffle=False, verbose=1)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"\n=> Best epoch trovata: {best_epoch}")
    
    # ==============================================================================
    # FASE 2: Refit Finale su Train+Validation
    # ==============================================================================
    print("\n--- FASE 2: Refit finale per il Test Set ---")
    fit_df_original = pd.concat([train_df_original, val_df_original], ignore_index=True)
    mu_proxy_final = fit_df_original['Log_Return'].mean()
    
    fit_df = process_features(fit_df_original, mu_proxy_final)
    test_df = process_features(test_df_original, mu_proxy_final)
    
    scaler2 = StandardScaler()
    fit_feat_scaled = scaler2.fit_transform(fit_df[feature_cols])
    test_feat_scaled = scaler2.transform(test_df[feature_cols])
    
    X_fit, y_fit = [], []
    for i in range(time_steps, len(fit_feat_scaled)):
        X_fit.append(fit_feat_scaled[i - time_steps:i])
        y_fit.append(fit_df[target_col].values[i])
    X_fit = np.array(X_fit)
    y_fit = np.asarray(y_fit, dtype=np.float32).reshape(-1, 1)
    
    X_test, y_test = create_oos_sequences(fit_feat_scaled, test_feat_scaled, test_df[target_col].values, time_steps)
    
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    final_model = build_model(time_steps, len(feature_cols))
    final_model.fit(X_fit, y_fit, epochs=best_epoch, batch_size=32, shuffle=False, verbose=1)
    
    print("\nGenerazione delle previsioni sul Test Set...")
    y_pred_scaled = final_model.predict(X_test, verbose=0).flatten()
    
    # ==============================================================================
    # FASE 3: Valutazione su Scala Decimale e Proxy Multiple
    # ==============================================================================
    forecast_dec = y_pred_scaled / variance_scale
    
    # 1. Squared Residual Proxy in float64
    realized_sq_dec = (test_df_original["Log_Return"].to_numpy(dtype=np.float64) - float(mu_proxy_final)) ** 2
    benchmark_sq_dec = (fit_df_original["Log_Return"].to_numpy(dtype=np.float64) - float(mu_proxy_final)) ** 2
    
    # Verifica Proxy e Date col GARCH
    garch_predictions_path = os.path.join(output_dir, "garch_predictions.csv")
    if os.path.exists(garch_predictions_path):
        garch_results = pd.read_csv(garch_predictions_path, parse_dates=["Date"])
        lstm_dates = pd.to_datetime(test_df_original["Date"]).reset_index(drop=True)
        garch_dates = pd.to_datetime(garch_results["Date"]).reset_index(drop=True)
        if not lstm_dates.equals(garch_dates):
            raise RuntimeError("Le date di valutazione LSTM e GARCH non coincidono.")
        if "Squared_Residual_Proxy" not in garch_results.columns:
            raise ValueError("Nel file GARCH manca Squared_Residual_Proxy.")
            
        garch_proxy = garch_results["Squared_Residual_Proxy"].to_numpy(dtype=float)
        if not np.allclose(realized_sq_dec, garch_proxy, rtol=1e-10, atol=1e-14):
            raise RuntimeError("La proxy primaria LSTM non coincide con quella utilizzata dal GARCH.")
            
    metrics_sq = calculate_metrics(realized_sq_dec, forecast_dec, benchmark_sq_dec)
    
    # 2. Parkinson Proxy
    parkinson_test_dec = (test_df_original["Log_Range"].values ** 2) / (4.0 * np.log(2.0))
    parkinson_fit_dec = (fit_df_original["Log_Range"].values ** 2) / (4.0 * np.log(2.0))
    metrics_pk = calculate_metrics(parkinson_test_dec, forecast_dec, parkinson_fit_dec)
    
    metrics_df = pd.DataFrame([
        {
            "Proxy": "Squared_Residual",
            "QLIKE": metrics_sq[0],
            "MSE_var": metrics_sq[1],
            "MAE_var": metrics_sq[2],
            "RMSE_vol": metrics_sq[3],
            "R2_OOS_HistoricalMean": metrics_sq[4],
            "QLIKE_Benchmark": metrics_sq[5],
            "Delta_QLIKE_vs_Benchmark": metrics_sq[6]
        },
        {
            "Proxy": "Parkinson_Var",
            "QLIKE": metrics_pk[0],
            "MSE_var": metrics_pk[1],
            "MAE_var": metrics_pk[2],
            "RMSE_vol": metrics_pk[3],
            "R2_OOS_HistoricalMean": metrics_pk[4],
            "QLIKE_Benchmark": metrics_pk[5],
            "Delta_QLIKE_vs_Benchmark": metrics_pk[6]
        }
    ])
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics_path = os.path.join(output_dir, "lstm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    
    predictions_df = pd.DataFrame({
        "Date": test_df_original['Date'].values,
        "Squared_Residual_Proxy": realized_sq_dec,
        "Parkinson_Var_Proxy": parkinson_test_dec,
        "LSTM_Forecast_Var": forecast_dec
    })
    predictions_path = os.path.join(output_dir, "lstm_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    
    # Salvataggio Metadati, Modello e Scaler
    final_model.save(os.path.join(output_dir, "lstm_final_model.keras"))
    joblib.dump(scaler2, os.path.join(output_dir, "lstm_scaler.joblib"))
    
    metadata_df = pd.DataFrame([{
        "Model": "LSTM",
        "Lookback": time_steps,
        "LSTM_Units": 32,
        "Dropout": 0.20,
        "Loss": "QLIKE",
        "Optimizer": "Adam",
        "Batch_Size": 32,
        "Best_Epoch": best_epoch,
        "Features": ", ".join(feature_cols),
        "Scale_Factor": scale_factor,
        "Mu_Proxy_Final": mu_proxy_final,
        "Train_Observations": len(train_df_original),
        "Validation_Observations": len(val_df_original),
        "Test_Observations": len(test_df_original),
        "Seed": 42
    }])
    metadata_df.to_csv(os.path.join(output_dir, "lstm_metadata.csv"), index=False)
    
    # Ripristino Grafico (Opzionale ma Utile)
    plt.figure(figsize=(14, 6))
    plt.plot(pd.to_datetime(test_df_original['Date']), realized_sq_dec, label='Proxy: Residuo Quadratico', color='lightgray', alpha=0.6)
    plt.plot(pd.to_datetime(test_df_original['Date']), forecast_dec, label='Previsione LSTM (32 units)', color='darkred', linewidth=1.5)
    plt.title('Previsione Out-of-Sample della Varianza - Modello LSTM Definitivo', fontsize=13, fontweight='bold')
    plt.xlabel('Data', fontsize=11)
    plt.ylabel('Varianza Decimale', fontsize=11)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lstm_variance_forecast.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # ESEMPIO DI CARICAMENTO FUTURO DEL MODELLO
    # ---------------------------------------------------------
    # Per ricaricare il modello e lo scaler in un altro script:
    #
    # import joblib
    # import tensorflow as tf
    # from lstm_model import qlike_loss_tf
    #
    # loaded_model = tf.keras.models.load_model(
    #     os.path.join(output_dir, "lstm_final_model.keras"),
    #     custom_objects={"qlike_loss_tf": qlike_loss_tf}
    # )
    # # Oppure compile=False se serve solo per predire
    #
    # scaler = joblib.load(os.path.join(output_dir, "lstm_scaler.joblib"))
    # ---------------------------------------------------------

    print("\n--- RISULTATI LSTM ---")
    print(metrics_df.to_string(index=False))
    return metrics_df, predictions_df, final_model

if __name__ == "__main__":
    run_lstm_pipeline()
