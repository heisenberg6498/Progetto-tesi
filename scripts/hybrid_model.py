import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
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
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, tf.shape(y_pred))
    eps = tf.cast(tf.keras.backend.epsilon(), y_pred.dtype)
    y_pred_safe = tf.maximum(y_pred, eps)
    loss = tf.math.log(y_pred_safe) + y_true / y_pred_safe
    return tf.reduce_mean(loss, axis=-1)

def qlike_loss_np(realized_var, forecast_var):
    realized_var = np.asarray(realized_var, dtype=float).flatten()
    forecast_var = np.asarray(forecast_var, dtype=float).flatten()
    return float(np.mean(np.log(forecast_var) + realized_var / forecast_var))

def calculate_metrics(realized_dec, forecast_dec, benchmark_history_dec):
    realized_dec = np.asarray(realized_dec, dtype=float).flatten()
    forecast_dec = np.asarray(forecast_dec, dtype=float).flatten()
    benchmark_history_dec = np.asarray(benchmark_history_dec, dtype=float).flatten()
    benchmark_mean = float(np.mean(benchmark_history_dec))
    benchmark_array = np.full(realized_dec.shape, benchmark_mean, dtype=float)

    mse_val = np.mean((realized_dec - forecast_dec) ** 2)
    mae_val = np.mean(np.abs(realized_dec - forecast_dec))
    qlike_val = qlike_loss_np(realized_dec, forecast_dec)
    rmse_vol = np.sqrt(np.mean((np.sqrt(realized_dec) - np.sqrt(forecast_dec)) ** 2))

    sse_model = np.sum((realized_dec - forecast_dec) ** 2)
    sse_bench = np.sum((realized_dec - benchmark_array) ** 2)
    r2_oos = np.nan if sse_bench <= np.finfo(float).eps else 1.0 - sse_model / sse_bench

    qlike_bench = qlike_loss_np(realized_dec, benchmark_array)
    delta_qlike = qlike_val - qlike_bench

    return qlike_val, mse_val, mae_val, rmse_vol, r2_oos, qlike_bench, delta_qlike

def build_hybrid_model(time_steps, lstm_features):
    # Branch 1: L'LSTM elabora la sequenza storica (rendimenti + proxy)
    lstm_input = Input(shape=(time_steps, lstm_features), name="Historical_Sequence")
    lstm_out = LSTM(32, return_sequences=False, name="LSTM_Layer")(lstm_input)
    lstm_out = Dropout(0.20, name="Dropout_Regularization")(lstm_out)
    
    # Branch 2: Il GARCH ci fornisce la singola previsione esatta al tempo T per il tempo T+1
    garch_input = Input(shape=(1,), name="GARCH_Forecast_t_plus_1")
    
    # Uniamo le estrazioni non-lineari dell'LSTM alla previsione robusta del GARCH
    concat = Concatenate(name="Concat_LSTM_GARCH")([lstm_out, garch_input])
    
    # Lo strato denso le pesa e produce una previsione congiunta sempre positiva
    output_var = Dense(1, activation='softplus', name="Output_Variance")(concat)
    
    model = Model(inputs=[lstm_input, garch_input], outputs=output_var, name="Hybrid_GARCH_LSTM")
    model.compile(optimizer='adam', loss=qlike_loss_tf, metrics=['mse', 'mae'])
    return model

def generate_pseudo_oos_garch(df_returns, mu_garch_scaled, omega_scaled, alpha, beta, scale_factor):
    """Genera la serie in-sample delle varianze condizionate del GARCH per addestrare l'ibrido senza leakage."""
    var_scaled = []
    # Inizializziamo con la varianza non condizionata del modello (omega / (1 - alpha - beta))
    initial_var = omega_scaled / (1.0 - alpha - beta) if (1.0 - alpha - beta) > 0 else np.var(df_returns - mu_garch_scaled)
    last_var = initial_var
    last_res_sq = initial_var
    
    for ret in df_returns:
        # Previsione per il giorno odierno
        curr_var = omega_scaled + alpha * last_res_sq + beta * last_var
        var_scaled.append(curr_var)
        
        # Aggiornamento per il loop successivo avendo "osservato" il giorno odierno
        last_res_sq = (ret - mu_garch_scaled) ** 2
        last_var = curr_var
        
    return np.array(var_scaled) / (scale_factor ** 2)

def run_hybrid_pipeline(data_path="data/ndx_processed_data.csv", output_dir="results", seed=42):
    print(f"=== AVVIO PIPELINE IBRIDA (GARCH + LSTM) (Seed: {seed}) ===")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. CARICAMENTO DATI
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.dropna(subset=["Date", "Log_Return", "Log_Range"]).sort_values("Date").reset_index(drop=True)
    
    total_len = len(df)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    scale_factor = 100.0
    variance_scale = scale_factor ** 2
    time_steps = 60
    
    # 2. IMPORTAZIONE PARAMETRI GARCH DA FILE (Dalla cartella principale results)
    garch_meta = pd.read_csv("results/garch_metadata.csv").iloc[0]
    garch_preds = pd.read_csv("results/garch_predictions.csv", parse_dates=["Date"])
    
    # 3. GENERAZIONE PREVISIONI EX-ANTE DEL GARCH PER TUTTA LA STORIA
    all_returns_scaled = df['Log_Return'].values * scale_factor
    garch_full_series = generate_pseudo_oos_garch(
        all_returns_scaled, 
        garch_meta['Mu_GARCH_Scaled'], 
        garch_meta['Omega_Scaled'], 
        garch_meta['Alpha'], 
        garch_meta['Beta'],
        scale_factor
    )
    # Scaliamo le varianze del GARCH riportandole alla scala x10.000 su cui lavora la rete
    df['GARCH_Variance_Scaled'] = garch_full_series * variance_scale
    
    # Adesso splittiamo gli originali
    train_df_original = df.iloc[:train_size].copy()
    val_df_original = df.iloc[train_size:train_size + val_size].copy()
    test_df_original = df.iloc[train_size + val_size:].copy()
    
    # 4. PREPARAZIONE DEL DATASET PER L'IBRIDO
    lstm_features = ['Scaled_Return', 'Squared_Residual_Proxy']
    target_col = 'Squared_Residual_Proxy'
    
    def process_features(dataset, mu_proxy):
        dataset = dataset.copy()
        dataset['Scaled_Return'] = dataset['Log_Return'] * scale_factor
        dataset['Squared_Residual_Proxy'] = ((dataset['Log_Return'] - mu_proxy) * scale_factor) ** 2
        return dataset
        
    mu_proxy_train = train_df_original['Log_Return'].mean()
    train_df = process_features(df.iloc[:train_size], mu_proxy_train)
    val_df = process_features(df.iloc[train_size:train_size + val_size], mu_proxy_train)
    
    scaler_hybrid = StandardScaler()
    train_feat_scaled = scaler_hybrid.fit_transform(train_df[lstm_features])
    val_feat_scaled = scaler_hybrid.transform(val_df[lstm_features])
    
    # Scaler dedicato solo alla feature numerica del GARCH
    garch_scaler1 = StandardScaler()
    train_garch_scaled = garch_scaler1.fit_transform(train_df[['GARCH_Variance_Scaled']])
    val_garch_scaled = garch_scaler1.transform(val_df[['GARCH_Variance_Scaled']])
    
    # Creazione Dati Train
    X_train_seq, X_train_garch, y_train = [], [], []
    for i in range(time_steps, len(train_df)):
        X_train_seq.append(train_feat_scaled[i - time_steps:i])
        X_train_garch.append(train_garch_scaled[i])
        y_train.append(train_df[target_col].values[i])
        
    X_train_seq = np.array(X_train_seq)
    X_train_garch = np.array(X_train_garch)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    
    # Creazione Dati Validation (senza perdere record)
    combined_val_seq = np.vstack([train_feat_scaled[-time_steps:], val_feat_scaled])
    X_val_seq = np.stack([combined_val_seq[i:i + time_steps] for i in range(len(val_feat_scaled))])
    X_val_garch = np.array(val_garch_scaled)
    y_val = np.asarray(val_df[target_col].values, dtype=np.float32).reshape(-1, 1)
    
    print("\n--- FASE 1: Addestramento e Selezione Best Epoch (Ibrido) ---")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model_val = build_hybrid_model(time_steps, len(lstm_features))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    
    history = model_val.fit(
        [X_train_seq, X_train_garch], y_train, 
        epochs=200, batch_size=32, 
        validation_data=([X_val_seq, X_val_garch], y_val), 
        callbacks=[early_stop], shuffle=False, verbose=1
    )
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"\n=> Best epoch trovata: {best_epoch}")
    
    # 5. REFIT FINALE SU TRAIN + VAL
    print("\n--- FASE 2: Refit finale per il Test Set (Ibrido) ---")
    fit_df_original = df.iloc[:train_size + val_size].copy()
    mu_proxy_final = fit_df_original['Log_Return'].mean()
    
    fit_df = process_features(fit_df_original, mu_proxy_final)
    test_df = process_features(test_df_original, mu_proxy_final)
    
    scaler_hybrid_final = StandardScaler()
    fit_feat_scaled = scaler_hybrid_final.fit_transform(fit_df[lstm_features])
    test_feat_scaled = scaler_hybrid_final.transform(test_df[lstm_features])
    
    garch_scaler_final = StandardScaler()
    fit_garch_scaled = garch_scaler_final.fit_transform(fit_df[['GARCH_Variance_Scaled']])
    test_garch_scaled = garch_scaler_final.transform(test_df[['GARCH_Variance_Scaled']])
    
    # Dati per fit finale
    X_fit_seq, X_fit_garch, y_fit = [], [], []
    for i in range(time_steps, len(fit_df)):
        X_fit_seq.append(fit_feat_scaled[i - time_steps:i])
        X_fit_garch.append(fit_garch_scaled[i])
        y_fit.append(fit_df[target_col].values[i])
        
    X_fit_seq = np.array(X_fit_seq)
    X_fit_garch = np.array(X_fit_garch)
    y_fit = np.asarray(y_fit, dtype=np.float32).reshape(-1, 1)
    
    # Dati Test
    combined_test_seq = np.vstack([fit_feat_scaled[-time_steps:], test_feat_scaled])
    X_test_seq = np.stack([combined_test_seq[i:i + time_steps] for i in range(len(test_feat_scaled))])
    X_test_garch = np.array(test_garch_scaled)
    y_test = np.asarray(test_df[target_col].values, dtype=np.float32).reshape(-1, 1)
    
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    final_model = build_hybrid_model(time_steps, len(lstm_features))
    final_model.fit([X_fit_seq, X_fit_garch], y_fit, epochs=best_epoch, batch_size=32, shuffle=False, verbose=1)
    
    print("\nGenerazione delle previsioni ibride sul Test Set...")
    y_pred_scaled = final_model.predict([X_test_seq, X_test_garch], verbose=0).flatten()
    forecast_dec = y_pred_scaled / variance_scale
    
    # 6. VALUTAZIONE MULTIPROXY E EXPORT
    realized_sq_dec = (test_df_original["Log_Return"].to_numpy(dtype=np.float64) - float(mu_proxy_final)) ** 2
    benchmark_sq_dec = (fit_df_original["Log_Return"].to_numpy(dtype=np.float64) - float(mu_proxy_final)) ** 2
    
    parkinson_test_dec = (test_df_original["Log_Range"].to_numpy(dtype=np.float64) ** 2) / (4.0 * np.log(2.0))
    parkinson_fit_dec = (fit_df_original["Log_Range"].to_numpy(dtype=np.float64) ** 2) / (4.0 * np.log(2.0))
    
    metrics_sq = calculate_metrics(realized_sq_dec, forecast_dec, benchmark_sq_dec)
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
    
    metrics_df.to_csv(os.path.join(output_dir, "hybrid_metrics.csv"), index=False)
    final_model.save(os.path.join(output_dir, "hybrid_final_model.keras"))
    joblib.dump({"seq_scaler": scaler_hybrid_final, "garch_scaler": garch_scaler_final}, os.path.join(output_dir, "hybrid_scalers.joblib"))
    
    # ---------------------------------------------------------
    # ESEMPIO DI CARICAMENTO FUTURO DEL MODELLO
    # ---------------------------------------------------------
    # Per ricaricare il modello ibrido e i suoi due scaler:
    #
    # import joblib
    # import tensorflow as tf
    # from hybrid_model import qlike_loss_tf
    #
    # loaded_model = tf.keras.models.load_model(
    #     os.path.join(output_dir, "hybrid_final_model.keras"),
    #     custom_objects={"qlike_loss_tf": qlike_loss_tf}
    # )
    # # Oppure compile=False se serve solo per predire
    #
    # scalers = joblib.load(os.path.join(output_dir, "hybrid_scalers.joblib"))
    # seq_scaler = scalers["seq_scaler"]
    # garch_scaler = scalers["garch_scaler"]
    # ---------------------------------------------------------

    print("\n--- RISULTATI MODELLO IBRIDO (GARCH+LSTM) ---")
    print(metrics_df.to_string(index=False))
    
    return metrics_df, final_model

if __name__ == "__main__":
    run_hybrid_pipeline()
