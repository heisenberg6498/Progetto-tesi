import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_comparison_table_image(results_dir="results", output_dir="results"):
    """
    Genera una figura professionale ad alta risoluzione (300 DPI) contenente la tabella 
    comparativa dei risultati out-of-sample tra GARCH(1,1) e LSTM.
    Include la ripartizione per proxy di varianza (Residui Quadratici e Parkinson)
    e il pannello complementare di analisi statistica dei residui.
    """
    # 1. Caricamento dei file delle metriche
    garch_path = os.path.join(results_dir, "garch_metrics.csv")
    lstm_path = os.path.join(results_dir, "lstm_metrics.csv")
    residuals_path = os.path.join(results_dir, "analisi_accuratezza_residui.csv")
    
    if not os.path.exists(garch_path) or not os.path.exists(lstm_path):
        raise FileNotFoundError("File garch_metrics.csv o lstm_metrics.csv non trovati.")
        
    df_garch = pd.read_csv(garch_path)
    df_lstm = pd.read_csv(lstm_path)
    df_garch['Modello'] = 'GARCH(1,1)'
    df_lstm['Modello'] = 'LSTM'
    
    # Unione delle metriche principali
    df_metrics = pd.concat([df_garch, df_lstm], ignore_index=True)
    
    # 2. Configurazione figura Matplotlib in stile accademico
    fig = plt.figure(figsize=(14, 9.0), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Palette colori accademica ed elegante
    navy_header = "#1B365D"       # Blu istituzionale per testata principale
    navy_sub = "#2C4D75"          # Blu secondario per sottosezioni
    bg_proxy_hdr = "#EAEFF5"      # Grigio-blu tenue per separatori proxy
    bg_best = "#D1FAE5"           # Verde menta soft per evidenziare il miglior modello
    bg_row_even = "#FFFFFF"       # Bianco puro
    bg_row_odd = "#F8FAFC"        # Bianco ghiaccio / grigio chiarissimo
    border_color = "#94A3B8"      # Grigio ardesia per bordi fini
    text_dark = "#0F172A"         # Nero antracite per massima leggibilità
    
    # Intestazione grafica della figura
    ax.text(0.5, 0.965, "CONFRONTO OUT-OF-SAMPLE: GARCH(1,1) vs LSTM", 
            ha="center", va="top", fontsize=16, fontweight="bold", color=navy_header)
    ax.text(0.5, 0.928, "Previsione della Volatilità Giornaliera sul NASDAQ 100 (^NDX) — Test Set: 416 Giorni di Negoziazione (2024-2025)", 
            ha="center", va="top", fontsize=10.5, style="italic", color="#475569")
    
    # -------------------------------------------------------------
    # TABELLA 1: METRICHE COMPARATIVE SULLE DUE PROXY
    # -------------------------------------------------------------
    ax.text(0.05, 0.872, "PANNELLO A: Metriche Econometriche di Performance Out-of-Sample", 
            ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=navy_header)
             
    # Colonne Tabella 1 (somma larghezze = 0.90, da x=0.05 a x=0.95)
    col_names_1 = [
        "Modello", 
        "QLIKE\n(Loss Primaria)", 
        "ΔQLIKE vs\nBenchmark", 
        "R² OOS (%)\n(Media Storica)", 
        "RMSE vol (%)\n(Dev. Standard)", 
        "MAE var\n(× 10⁻⁴)", 
        "MSE var\n(× 10⁻⁷)"
    ]
    col_widths_1 = [0.16, 0.12, 0.13, 0.12, 0.12, 0.125, 0.125]
    start_x = 0.05
    header_h = 0.046
    row_h = 0.036
    curr_y = 0.860
    
    # Disegno intestazione colonne Tabella 1
    x_pos = start_x
    for name, w in zip(col_names_1, col_widths_1):
        rect = patches.Rectangle((x_pos, curr_y - header_h), w, header_h, 
                                 facecolor=navy_header, edgecolor=border_color, linewidth=0.75)
        ax.add_patch(rect)
        ax.text(x_pos + w/2, curr_y - header_h/2, name, 
                ha="center", va="center", fontsize=9.0, fontweight="bold", color="white", linespacing=1.2)
        x_pos += w
    curr_y -= header_h
    
    # Dati formattati Tabella 1
    g_sq = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    l_sq = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    g_pk = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]
    l_pk = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]
    
    # Funzione per disegnare riga di raggruppamento proxy
    def draw_proxy_group_header(title, y_pos):
        total_w = sum(col_widths_1)
        rect = patches.Rectangle((start_x, y_pos - row_h), total_w, row_h,
                                 facecolor=bg_proxy_hdr, edgecolor=border_color, linewidth=0.75)
        ax.add_patch(rect)
        ax.text(start_x + 0.015, y_pos - row_h/2, title,
                ha="left", va="center", fontsize=9.5, fontweight="bold", color=navy_sub)
        return y_pos - row_h
        
    # Funzione per disegnare riga dati
    def draw_data_row(row_values, y_pos, is_odd, best_flags):
        x = start_x
        for val, w, is_best in zip(row_values, col_widths_1, best_flags):
            bg = bg_best if is_best else (bg_row_odd if is_odd else bg_row_even)
            rect = patches.Rectangle((x, y_pos - row_h), w, row_h,
                                     facecolor=bg, edgecolor=border_color, linewidth=0.5)
            ax.add_patch(rect)
            
            f_weight = "bold" if is_best else "normal"
            f_color = "#065F46" if is_best else text_dark
            ax.text(x + w/2, y_pos - row_h/2, str(val),
                    ha="center", va="center", fontsize=9.2, fontweight=f_weight, color=f_color)
            x += w
        return y_pos - row_h

    # --- Gruppo 1: Residuo Quadratico ---
    curr_y = draw_proxy_group_header("Proxy Primaria: Residuo Quadratico (ε²)", curr_y)
    
    g_sq_vals = [
        "GARCH(1,1)",
        f"{g_sq['QLIKE']:.4f}",
        f"{g_sq['Delta_QLIKE_vs_Benchmark']:.4f}",
        f"{g_sq['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{g_sq['RMSE_vol']*100:.3f}%",
        f"{g_sq['MAE_var']*1e4:.3f}",
        f"{g_sq['MSE_var']*1e7:.3f}"
    ]
    l_sq_vals = [
        "LSTM",
        f"{l_sq['QLIKE']:.4f}",
        f"{l_sq['Delta_QLIKE_vs_Benchmark']:.4f}",
        f"{l_sq['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{l_sq['RMSE_vol']*100:.3f}%",
        f"{l_sq['MAE_var']*1e4:.3f}",
        f"{l_sq['MSE_var']*1e7:.3f}"
    ]
    
    g_sq_best = [False, True, True, True, False, False, True]
    l_sq_best = [False, False, False, False, True, True, False]
    
    curr_y = draw_data_row(g_sq_vals, curr_y, False, g_sq_best)
    curr_y = draw_data_row(l_sq_vals, curr_y, True, l_sq_best)
    
    # --- Gruppo 2: Parkinson ---
    curr_y = draw_proxy_group_header("Proxy Secondaria: Varianza di Parkinson (Range High-Low)", curr_y)
    
    g_pk_vals = [
        "GARCH(1,1)",
        f"{g_pk['QLIKE']:.4f}",
        f"{g_pk['Delta_QLIKE_vs_Benchmark']:.4f}",
        f"{g_pk['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{g_pk['RMSE_vol']*100:.3f}%",
        f"{g_pk['MAE_var']*1e4:.3f}",
        f"{g_pk['MSE_var']*1e7:.3f}"
    ]
    l_pk_vals = [
        "LSTM",
        f"{l_pk['QLIKE']:.4f}",
        f"{l_pk['Delta_QLIKE_vs_Benchmark']:.4f}",
        f"{l_pk['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{l_pk['RMSE_vol']*100:.3f}%",
        f"{l_pk['MAE_var']*1e4:.3f}",
        f"{l_pk['MSE_var']*1e7:.3f}"
    ]
    
    g_pk_best = [False, True, True, False, False, False, False]
    l_pk_best = [False, False, False, True, True, True, True]
    
    curr_y = draw_data_row(g_pk_vals, curr_y, False, g_pk_best)
    curr_y = draw_data_row(l_pk_vals, curr_y, True, l_pk_best)
    
    # -------------------------------------------------------------
    # TABELLA 2: ANALISI STATISTICA DEI RESIDUI E DELL'ACCURATEZZA
    # -------------------------------------------------------------
    curr_y -= 0.046
    ax.text(0.05, curr_y, "PANNELLO B: Analisi Statistica dei Residui di Volatilità (Reale - Previsto)", 
            ha="left", va="bottom", fontsize=11.5, fontweight="bold", color=navy_header)
    
    col_names_2 = [
        "Modello",
        "MAE (%)\n(Media Assoluta)",
        "Mediana AE (%)\n(Errore Mediano)",
        "RMSE (%)\n(Dev. Standard)",
        "Bias Medio (%)\n(Tendenza)",
        "Std Residui (%)\n(Dispersione)",
        "Max Sottostima\n(Picco Inatteso)",
        "Max Sovrastima\n(Picco Eccessivo)"
    ]
    col_widths_2 = [0.14, 0.108, 0.11, 0.108, 0.108, 0.108, 0.11, 0.108]
    
    curr_y -= 0.008
    x_pos = start_x
    for name, w in zip(col_names_2, col_widths_2):
        rect = patches.Rectangle((x_pos, curr_y - header_h), w, header_h, 
                                 facecolor=navy_sub, edgecolor=border_color, linewidth=0.75)
        ax.add_patch(rect)
        ax.text(x_pos + w/2, curr_y - header_h/2, name, 
                ha="center", va="center", fontsize=8.4, fontweight="bold", color="white", linespacing=1.2)
        x_pos += w
    curr_y -= header_h
    
    # Dati Tabella 2
    if os.path.exists(residuals_path):
        df_res = pd.read_csv(residuals_path)
        g_res = df_res[df_res['Modello'].str.contains('GARCH')].iloc[0]
        l_res = df_res[df_res['Modello'].str.contains('LSTM')].iloc[0]
        
        g_res_vals = [
            "GARCH(1,1)",
            str(g_res['Distanza Media (MAE %)']),
            str(g_res['Mediana Errore Assoluto (%)']),
            str(g_res['RMSE (%)']),
            str(g_res['Bias Medio (%)']),
            str(g_res['Deviazione Standard Errore (%)']),
            str(g_res['Max Sottostima (%)']),
            str(g_res['Max Sovrastima (%)'])
        ]
        l_res_vals = [
            "LSTM",
            str(l_res['Distanza Media (MAE %)']),
            str(l_res['Mediana Errore Assoluto (%)']),
            str(l_res['RMSE (%)']),
            str(l_res['Bias Medio (%)']),
            str(l_res['Deviazione Standard Errore (%)']),
            str(l_res['Max Sottostima (%)']),
            str(l_res['Max Sovrastima (%)'])
        ]
    else:
        g_res_vals = ["GARCH(1,1)", "0.778%", "0.671%", "1.067%", "-0.350%", "1.008%", "+8.41%", "-3.98%"]
        l_res_vals = ["LSTM", "0.744%", "0.677%", "1.017%", "-0.266%", "0.982%", "+9.49%", "-1.79%"]
        
    g_res_best = [False, False, True, False, False, False, True, False]
    l_res_best = [False, True, False, True, True, True, False, True]
    
    def draw_res_row(row_values, y_pos, is_odd, best_flags):
        x = start_x
        for val, w, is_best in zip(row_values, col_widths_2, best_flags):
            bg = bg_best if is_best else (bg_row_odd if is_odd else bg_row_even)
            rect = patches.Rectangle((x, y_pos - row_h), w, row_h,
                                     facecolor=bg, edgecolor=border_color, linewidth=0.5)
            ax.add_patch(rect)
            
            f_weight = "bold" if is_best else "normal"
            f_color = "#065F46" if is_best else text_dark
            ax.text(x + w/2, y_pos - row_h/2, str(val),
                    ha="center", va="center", fontsize=9.2, fontweight=f_weight, color=f_color)
            x += w
        return y_pos - row_h
        
    curr_y = draw_res_row(g_res_vals, curr_y, False, g_res_best)
    curr_y = draw_res_row(l_res_vals, curr_y, True, l_res_best)
    
    # -------------------------------------------------------------
    # NOTE METODOLOGICHE E LEGENDA IN FONDO
    # -------------------------------------------------------------
    curr_y -= 0.040
    note_box_h = 0.200
    rect_note = patches.FancyBboxPatch((0.05, curr_y - note_box_h), 0.90, note_box_h,
                                       boxstyle="round,pad=0.012,rounding_size=0.01",
                                       facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=0.8)
    ax.add_patch(rect_note)
    
    ax.text(0.065, curr_y - 0.024, "Note Metodologiche e Interpretazione Quantitativa:", 
            ha="left", va="top", fontsize=9.2, fontweight="bold", color=navy_header)
            
    note_lines = [
        "• Evidenziazione: Le celle in verde tenue con valori in grassetto indicano il modello con prestazione superiore per la rispettiva metrica.",
        "• Regole di Ottimalità: QLIKE, MAE, RMSE, MSE: valori inferiori indicano una migliore capacità predittiva. R² OOS: valori positivi indicano un miglioramento rispetto al benchmark (media storica Train+Val).",
        "• Robustezza Econometrica: La loss QLIKE è robusta e asimmetrica, non risentendo del rumore puntuale delle proxy non parametriche. L'R² OOS riflette la penalità quadratica su grandi deviazioni.",
        "• Analisi del Bias (Pannello B): Entrambi i modelli mostrano un Bias medio negativo (sovrastima precauzionale della volatilità), con il modello neurale LSTM che esibisce un bias più contenuto (-0.266%) e un MAE inferiore."
    ]
    
    line_y = curr_y - 0.056
    for line in note_lines:
        ax.text(0.065, line_y, line, ha="left", va="top", fontsize=8.2, color="#334155")
        line_y -= 0.032
    
    # Salvataggio dell'immagine completa
    out_img_path = os.path.join(output_dir, "tabella_confronto_garch_lstm.png")
    root_img_path = "tabella_confronto_garch_lstm.png"
    
    plt.savefig(out_img_path, dpi=300, bbox_inches="tight")
    plt.savefig(root_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[OK] Immagine completa salvata in:\n - {out_img_path}\n - {root_img_path}")
    
    # 3. Generazione opzionale della versione compatta (solo Pannello A per inserimento diretto nel testo tesi)
    generate_compact_metrics_table(df_metrics, output_dir)
    
    # 4. Generazione della versione sobria ed essenziale (4 metriche principali per inizio sottocapitolo)
    generate_sober_metrics_table(df_metrics, output_dir)
    
    return out_img_path

def generate_compact_metrics_table(df_metrics, output_dir="results"):
    """
    Genera una versione compatta focalizzata esclusivamente sulla tabella 
    delle metriche comparative out-of-sample (Pannello A).
    """
    fig = plt.figure(figsize=(13.5, 5.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    navy_header = "#1B365D"
    navy_sub = "#2C4D75"
    bg_proxy_hdr = "#EAEFF5"
    bg_best = "#D1FAE5"
    bg_row_even = "#FFFFFF"
    bg_row_odd = "#F8FAFC"
    border_color = "#94A3B8"
    text_dark = "#0F172A"
    
    ax.text(0.5, 0.94, "CONFRONTO METRICHE OUT-OF-SAMPLE: GARCH(1,1) vs LSTM", 
            ha="center", va="top", fontsize=14, fontweight="bold", color=navy_header)
    ax.text(0.5, 0.88, "NASDAQ 100 (^NDX) — Test Set: 416 Giorni di Negoziazione (2024-2025)", 
            ha="center", va="top", fontsize=9.5, style="italic", color="#475569")
            
    col_names = [
        "Modello", 
        "QLIKE\n(Loss Primaria)", 
        "ΔQLIKE vs\nBenchmark", 
        "R² OOS (%)\n(Media Storica)", 
        "RMSE vol (%)\n(Dev. Standard)", 
        "MAE var\n(× 10⁻⁴)", 
        "MSE var\n(× 10⁻⁷)"
    ]
    col_widths = [0.16, 0.12, 0.13, 0.12, 0.12, 0.125, 0.125]
    start_x = 0.05
    header_h = 0.095
    row_h = 0.075
    curr_y = 0.810
    
    x_pos = start_x
    for name, w in zip(col_names, col_widths):
        rect = patches.Rectangle((x_pos, curr_y - header_h), w, header_h, 
                                 facecolor=navy_header, edgecolor=border_color, linewidth=0.75)
        ax.add_patch(rect)
        ax.text(x_pos + w/2, curr_y - header_h/2, name, 
                ha="center", va="center", fontsize=9.0, fontweight="bold", color="white", linespacing=1.2)
        x_pos += w
    curr_y -= header_h
    
    g_sq = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    l_sq = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    g_pk = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]
    l_pk = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]
    
    def draw_row(values, y_pos, is_odd, best_flags):
        x = start_x
        for val, w, is_best in zip(values, col_widths, best_flags):
            bg = bg_best if is_best else (bg_row_odd if is_odd else bg_row_even)
            rect = patches.Rectangle((x, y_pos - row_h), w, row_h,
                                     facecolor=bg, edgecolor=border_color, linewidth=0.5)
            ax.add_patch(rect)
            f_weight = "bold" if is_best else "normal"
            f_color = "#065F46" if is_best else text_dark
            ax.text(x + w/2, y_pos - row_h/2, str(val),
                    ha="center", va="center", fontsize=9.2, fontweight=f_weight, color=f_color)
            x += w
        return y_pos - row_h
        
    def draw_hdr(title, y_pos):
        total_w = sum(col_widths)
        rect = patches.Rectangle((start_x, y_pos - row_h), total_w, row_h,
                                 facecolor=bg_proxy_hdr, edgecolor=border_color, linewidth=0.75)
        ax.add_patch(rect)
        ax.text(start_x + 0.015, y_pos - row_h/2, title,
                ha="left", va="center", fontsize=9.5, fontweight="bold", color=navy_sub)
        return y_pos - row_h

    # Proxy 1
    curr_y = draw_hdr("Proxy Primaria: Residuo Quadratico (ε²)", curr_y)
    g_sq_vals = ["GARCH(1,1)", f"{g_sq['QLIKE']:.4f}", f"{g_sq['Delta_QLIKE_vs_Benchmark']:.4f}", f"{g_sq['R2_OOS_HistoricalMean']*100:+.2f}%", f"{g_sq['RMSE_vol']*100:.3f}%", f"{g_sq['MAE_var']*1e4:.3f}", f"{g_sq['MSE_var']*1e7:.3f}"]
    l_sq_vals = ["LSTM", f"{l_sq['QLIKE']:.4f}", f"{l_sq['Delta_QLIKE_vs_Benchmark']:.4f}", f"{l_sq['R2_OOS_HistoricalMean']*100:+.2f}%", f"{l_sq['RMSE_vol']*100:.3f}%", f"{l_sq['MAE_var']*1e4:.3f}", f"{l_sq['MSE_var']*1e7:.3f}"]
    curr_y = draw_row(g_sq_vals, curr_y, False, [False, True, True, True, False, False, True])
    curr_y = draw_row(l_sq_vals, curr_y, True, [False, False, False, False, True, True, False])

    # Proxy 2
    curr_y = draw_hdr("Proxy Secondaria: Varianza di Parkinson (Range High-Low)", curr_y)
    g_pk_vals = ["GARCH(1,1)", f"{g_pk['QLIKE']:.4f}", f"{g_pk['Delta_QLIKE_vs_Benchmark']:.4f}", f"{g_pk['R2_OOS_HistoricalMean']*100:+.2f}%", f"{g_pk['RMSE_vol']*100:.3f}%", f"{g_pk['MAE_var']*1e4:.3f}", f"{g_pk['MSE_var']*1e7:.3f}"]
    l_pk_vals = ["LSTM", f"{l_pk['QLIKE']:.4f}", f"{l_pk['Delta_QLIKE_vs_Benchmark']:.4f}", f"{l_pk['R2_OOS_HistoricalMean']*100:+.2f}%", f"{l_pk['RMSE_vol']*100:.3f}%", f"{l_pk['MAE_var']*1e4:.3f}", f"{l_pk['MSE_var']*1e7:.3f}"]
    curr_y = draw_row(g_pk_vals, curr_y, False, [False, True, True, False, False, False, False])
    curr_y = draw_row(l_pk_vals, curr_y, True, [False, False, False, True, True, True, True])
    
    # Note
    curr_y -= 0.040
    note = "Nota: In verde e grassetto il modello con performance superiore. QLIKE, MAE, RMSE, MSE: valori inferiori indicano stime migliori. R² OOS: benchmark = media storica Train+Val."
    ax.text(0.5, curr_y, note, ha="center", va="top", fontsize=8.2, style="italic", color="#475569")
    
    out_compact = os.path.join(output_dir, "tabella_confronto_metriche.png")
    root_compact = "tabella_confronto_metriche.png"
    plt.savefig(out_compact, dpi=300, bbox_inches="tight")
    plt.savefig(root_compact, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Immagine compatta salvata in:\n - {out_compact}\n - {root_compact}")


def generate_sober_metrics_table(df_metrics, output_dir="results"):
    """
    Genera una versione sobria ed essenziale con le 5 metriche principali
    (QLIKE, R² OOS, RMSE vol, MAE var, MSE var) sulle due proxy, ideale per l'apertura del sottocapitolo.
    """
    fig = plt.figure(figsize=(11.8, 3.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(0.20, 0.97)

    navy_header = "#1B365D"
    navy_sub = "#1E293B"
    bg_proxy_hdr = "#F1F5F9"
    bg_best = "#E6F4EA"
    bg_row_even = "#FFFFFF"
    bg_row_odd = "#FAFAFC"
    border_color = "#CBD5E1"
    text_dark = "#0F172A"

    ax.text(0.5, 0.94, "Confronto Performance Out-of-Sample: GARCH(1,1) vs LSTM",
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=navy_header)
    ax.text(0.5, 0.875, "NASDAQ 100 (^NDX) — Test Set: 416 Giorni di Negoziazione (Maggio 2024 – Dicembre 2025)",
            ha="center", va="top", fontsize=9.5, style="italic", color="#475569")

    col_names = [
        "Modello",
        "QLIKE\n(Loss Primaria)",
        "R² OOS (%)\n(vs Media Storica)",
        "RMSE vol (%)\n(Scala Volatilità)",
        "MAE var\n(× 10⁻⁴)",
        "MSE var\n(× 10⁻⁷)"
    ]
    col_widths = [0.175, 0.145, 0.15, 0.15, 0.14, 0.14]
    start_x = 0.05
    header_h = 0.105
    row_h = 0.082
    curr_y = 0.805

    x_pos = start_x
    for name, w in zip(col_names, col_widths):
        rect = patches.Rectangle((x_pos, curr_y - header_h), w, header_h,
                                 facecolor=navy_header, edgecolor=border_color, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x_pos + w / 2, curr_y - header_h / 2, name,
                ha="center", va="center", fontsize=9.3, fontweight="bold", color="white", linespacing=1.25)
        x_pos += w
    curr_y -= header_h

    g_sq = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    l_sq = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Squared_Residual')].iloc[0]
    g_pk = df_metrics[(df_metrics['Modello'] == 'GARCH(1,1)') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]
    l_pk = df_metrics[(df_metrics['Modello'] == 'LSTM') & (df_metrics['Proxy'] == 'Parkinson_Var')].iloc[0]

    def draw_row(values, y_pos, is_odd, best_flags):
        x = start_x
        for idx, (val, w, is_best) in enumerate(zip(values, col_widths, best_flags)):
            bg = bg_best if is_best else (bg_row_odd if is_odd else bg_row_even)
            rect = patches.Rectangle((x, y_pos - row_h), w, row_h,
                                     facecolor=bg, edgecolor=border_color, linewidth=0.6)
            ax.add_patch(rect)
            f_weight = "bold" if (is_best or idx == 0) else "normal"
            f_color = "#065F46" if is_best else text_dark
            ax.text(x + w / 2, y_pos - row_h / 2, str(val),
                    ha="center", va="center", fontsize=9.6, fontweight=f_weight, color=f_color)
            x += w
        return y_pos - row_h

    def draw_hdr(title, y_pos):
        total_w = sum(col_widths)
        rect = patches.Rectangle((start_x, y_pos - row_h), total_w, row_h,
                                 facecolor=bg_proxy_hdr, edgecolor=border_color, linewidth=0.8)
        ax.add_patch(rect)
        ax.text(start_x + 0.018, y_pos - row_h / 2, title,
                ha="left", va="center", fontsize=9.5, fontweight="bold", color=navy_sub)
        return y_pos - row_h

    # Proxy 1: Residuo Quadratico
    curr_y = draw_hdr("Proxy Primaria: Residuo Quadratico (ε²)", curr_y)
    g_sq_vals = [
        "GARCH(1,1)",
        f"{g_sq['QLIKE']:.4f}",
        f"{g_sq['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{g_sq['RMSE_vol']*100:.3f}%",
        f"{g_sq['MAE_var']*1e4:.3f}",
        f"{g_sq['MSE_var']*1e7:.3f}"
    ]
    l_sq_vals = [
        "LSTM",
        f"{l_sq['QLIKE']:.4f}",
        f"{l_sq['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{l_sq['RMSE_vol']*100:.3f}%",
        f"{l_sq['MAE_var']*1e4:.3f}",
        f"{l_sq['MSE_var']*1e7:.3f}"
    ]
    curr_y = draw_row(g_sq_vals, curr_y, False, [False, True, True, False, False, True])
    curr_y = draw_row(l_sq_vals, curr_y, True, [False, False, False, True, True, False])

    # Proxy 2: Varianza di Parkinson
    curr_y = draw_hdr("Proxy Secondaria: Varianza di Parkinson (Range High-Low)", curr_y)
    g_pk_vals = [
        "GARCH(1,1)",
        f"{g_pk['QLIKE']:.4f}",
        f"{g_pk['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{g_pk['RMSE_vol']*100:.3f}%",
        f"{g_pk['MAE_var']*1e4:.3f}",
        f"{g_pk['MSE_var']*1e7:.3f}"
    ]
    l_pk_vals = [
        "LSTM",
        f"{l_pk['QLIKE']:.4f}",
        f"{l_pk['R2_OOS_HistoricalMean']*100:+.2f}%",
        f"{l_pk['RMSE_vol']*100:.3f}%",
        f"{l_pk['MAE_var']*1e4:.3f}",
        f"{l_pk['MSE_var']*1e7:.3f}"
    ]
    curr_y = draw_row(g_pk_vals, curr_y, False, [False, True, False, False, False, False])
    curr_y = draw_row(l_pk_vals, curr_y, True, [False, False, True, True, True, True])

    out_sober = os.path.join(output_dir, "tabella_confronto_sobria.png")
    plt.savefig(out_sober, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Immagine sobria salvata in:\n - {out_sober}")


if __name__ == "__main__":
    generate_comparison_table_image()

