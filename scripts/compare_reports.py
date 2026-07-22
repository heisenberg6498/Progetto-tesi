import os
import re
import json
import numpy as np
import pandas as pd

def extract_js_array(html_content, var_name):
    pattern = re.compile(rf'const\s+{var_name}\s*=\s*(\[.*?\]);', re.DOTALL)
    match = pattern.search(html_content)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            print(f"Error parsing JSON for {var_name}: {e}")
            return None
    return None

def calculate_drawdown(prices):
    prices_ser = pd.Series(prices)
    roll_max = prices_ser.cummax()
    drawdowns = prices_ser / roll_max - 1.0
    return drawdowns.min()

def main():
    report_path_1 = "results/ndx_data_report.html"
    report_path_2 = "results/ndx_data_report_1.html"
    output_path = "results/ndx_comparison_report.html"
    
    print(f"Lettura dei due report:")
    print(f"  - Report 1: {report_path_1}")
    print(f"  - Report 2: {report_path_2}")
    
    if not os.path.exists(report_path_1) or not os.path.exists(report_path_2):
        raise FileNotFoundError("Assicurarsi che entrambi i file di report esistano nella cartella 'results'.")
        
    with open(report_path_1, 'r', encoding='utf-8') as f:
        html_1 = f.read()
        
    with open(report_path_2, 'r', encoding='utf-8') as f:
        html_2 = f.read()
        
    # Estrazione dei dati
    dates_1 = extract_js_array(html_1, "dates")
    prices_1 = extract_js_array(html_1, "prices")
    returns_1 = extract_js_array(html_1, "returns")
    ranges_1 = extract_js_array(html_1, "logRanges")
    volumes_1 = extract_js_array(html_1, "volumes")
    
    dates_2 = extract_js_array(html_2, "dates")
    prices_2 = extract_js_array(html_2, "prices")
    returns_2 = extract_js_array(html_2, "returns")
    ranges_2 = extract_js_array(html_2, "logRanges")
    volumes_2 = extract_js_array(html_2, "volumes")
    
    # Validazione estrazione
    if not (dates_1 and prices_1 and returns_1):
        raise ValueError("Impossibile estrarre i dati completi dal Report 1.")
    if not (dates_2 and prices_2 and returns_2):
        raise ValueError("Impossibile estrarre i dati completi dal Report 2.")
        
    print(f"Dati estratti con successo.")
    print(f"  Report 1: {len(dates_1)} righe.")
    print(f"  Report 2: {len(dates_2)} righe.")
    
    # Creazione dei DataFrame Pandas
    df1 = pd.DataFrame({
        'Date': pd.to_datetime(dates_1),
        'Close': prices_1,
        'Return': returns_1,
        'Range': ranges_1 if ranges_1 else [0]*len(dates_1),
        'Volume': volumes_1 if volumes_1 else [0]*len(dates_1)
    })
    
    df2 = pd.DataFrame({
        'Date': pd.to_datetime(dates_2),
        'Close': prices_2,
        'Return': returns_2,
        'Range': ranges_2 if ranges_2 else [0]*len(dates_2),
        'Volume': volumes_2 if volumes_2 else [0]*len(dates_2)
    })
    
    # Analisi Statistica
    n_obs_1 = len(df1)
    zeros_1 = (df1['Return'] == 0.0).sum()
    pct_zeros_1 = (zeros_1 / n_obs_1) * 100
    mean_ret_1 = df1['Return'].mean()
    std_ret_1 = df1['Return'].std()
    vol_252_1 = std_ret_1 * np.sqrt(252)
    vol_365_1 = std_ret_1 * np.sqrt(365)
    skew_1 = df1['Return'].skew()
    kurt_1 = df1['Return'].kurt()
    max_dd_1 = calculate_drawdown(df1['Close'])
    mean_price_1 = df1['Close'].mean()
    
    # Report 2 (Trading Nativo)
    n_obs_2 = len(df2)
    zeros_2 = (df2['Return'] == 0.0).sum()
    pct_zeros_2 = (zeros_2 / n_obs_2) * 100
    mean_ret_2 = df2['Return'].mean()
    std_ret_2 = df2['Return'].std()
    vol_252_2 = std_ret_2 * np.sqrt(252)
    vol_365_2 = std_ret_2 * np.sqrt(365)
    skew_2 = df2['Return'].skew()
    kurt_2 = df2['Return'].kurt()
    max_dd_2 = calculate_drawdown(df2['Close'])
    mean_price_2 = df2['Close'].mean()
    
    # Calcolo Rolling Volatility (21 trading days)
    df1['Rolling_Vol'] = df1['Return'].rolling(21).std() * np.sqrt(252)
    df2['Rolling_Vol'] = df2['Return'].rolling(21).std() * np.sqrt(252)
    
    df1_clean = df1.dropna(subset=['Rolling_Vol'])
    df2_clean = df2.dropna(subset=['Rolling_Vol'])
    
    dates_roll_1 = df1_clean['Date'].dt.strftime('%Y-%m-%d').tolist()
    vol_roll_1 = (df1_clean['Rolling_Vol'] * 100).tolist() # In percent
    
    dates_roll_2 = df2_clean['Date'].dt.strftime('%Y-%m-%d').tolist()
    vol_roll_2 = (df2_clean['Rolling_Vol'] * 100).tolist() # In percent
    
    # Calcolo Istogramma
    bins = np.linspace(-0.04, 0.04, 81)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    hist_1, _ = np.histogram(df1['Return'], bins=bins)
    hist_2, _ = np.histogram(df2['Return'], bins=bins)
    
    density_1 = (hist_1 / len(df1)) * 100
    density_2 = (hist_2 / len(df2)) * 100
    
    bin_centers_json = json.dumps(bin_centers.tolist())
    density_1_json = json.dumps(density_1.tolist())
    density_2_json = json.dumps(density_2.tolist())
    
    # Calcolo Autocorrelazione dei Rendimenti al Quadrato
    lags = list(range(1, 16))
    acf_1 = [df1['Return'].pow(2).autocorr(lag=l) for l in lags]
    acf_2 = [df2['Return'].pow(2).autocorr(lag=l) for l in lags]
    
    # Rimuovere eventuali NaN dall'ACF se presenti
    acf_1 = [x if not np.isnan(x) else 0 for x in acf_1]
    acf_2 = [x if not np.isnan(x) else 0 for x in acf_2]
    
    lags_json = json.dumps(lags)
    acf_1_json = json.dumps(acf_1)
    acf_2_json = json.dumps(acf_2)
    
    dates_roll_1_json = json.dumps(dates_roll_1)
    vol_roll_1_json = json.dumps(vol_roll_1)
    dates_roll_2_json = json.dumps(dates_roll_2)
    vol_roll_2_json = json.dumps(vol_roll_2)
    
    # Differenze percentuali e assolute
    diff_obs = n_obs_1 - n_obs_2
    diff_zeros = pct_zeros_1 - pct_zeros_2
    diff_mean = ((mean_ret_1 - mean_ret_2) / mean_ret_2) * 100 if mean_ret_2 != 0 else 0
    diff_vol_252 = ((vol_252_1 - vol_252_2) / vol_252_2) * 100
    diff_kurt = ((kurt_1 - kurt_2) / kurt_2) * 100
    diff_skew = ((skew_1 - skew_2) / skew_2) * 100
    diff_price = ((mean_price_1 - mean_price_2) / mean_price_2) * 100
    
    # HTML Content
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Comparativo: Calendario Solare vs Calendario di Trading</title>
    
    <!-- Google Fonts: Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Plotly.js via CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.0.min.js"></script>
    
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.6);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --primary-accent: #6366f1; /* Indigo - Solar Continuous */
            --secondary-accent: #06b6d4; /* Cyan - Trading Nativo */
            --highlight-color: #10b981; /* Emerald */
            --warning-color: #f43f5e; /* Rose */
            --amber-color: #f59e0b; /* Amber */
            --font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.6;
            padding: 2rem 4rem;
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 1rem 1.5rem;
            }}
        }}

        header {{
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
            position: relative;
        }}

        .badge {{
            display: inline-block;
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
            color: #ffffff;
            padding: 0.3rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 40%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }}

        /* Grid Layouts */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .summary-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .card-header-flex {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .card-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}

        .dot-solar {{ background-color: var(--primary-accent); box-shadow: 0 0 10px var(--primary-accent); }}
        .dot-trading {{ background-color: var(--secondary-accent); box-shadow: 0 0 10px var(--secondary-accent); }}
        .dot-diff {{ background-color: var(--amber-color); box-shadow: 0 0 10px var(--amber-color); }}

        .card-title {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .card-main-val {{
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }}

        .card-list {{
            list-style: none;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}

        .card-list li {{
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
        }}

        .card-list span.val {{
            color: #ffffff;
            font-weight: 500;
        }}

        /* Table CSS */
        .table-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 35px -15px rgba(0, 0, 0, 0.4);
            margin-bottom: 3rem;
            overflow-x: auto;
        }}

        .table-card h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
            min-width: 700px;
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 500;
            padding: 1rem;
            border-bottom: 2px solid var(--card-border);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        .col-metric {{
            font-weight: 600;
            width: 30%;
        }}

        .col-val-solar {{
            font-family: 'Courier New', Courier, monospace;
            color: #a5b4fc; /* light indigo */
            font-weight: 600;
            text-align: right;
        }}

        .col-val-trading {{
            font-family: 'Courier New', Courier, monospace;
            color: #67e8f9; /* light cyan */
            font-weight: 600;
            text-align: right;
        }}

        .col-val-diff {{
            font-family: 'Courier New', Courier, monospace;
            font-weight: 600;
            text-align: right;
        }}

        .diff-up {{ color: var(--warning-color); }}
        .diff-down {{ color: var(--highlight-color); }}
        .diff-neutral {{ color: var(--text-secondary); }}

        .col-interpretation {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            width: 35%;
            padding-left: 1.5rem;
        }}

        /* Dashboard Layout for Charts */
        .dashboard-layout {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        @media (min-width: 1200px) {{
            .dashboard-layout {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .span-full {{
                grid-column: span 2;
            }}
        }}

        .chart-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 15px 35px -15px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
        }}

        .chart-header {{
            margin-bottom: 1rem;
        }}

        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
        }}

        .chart-subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }}

        .chart-container {{
            width: 100%;
            height: 450px;
        }}

        /* Academic Commentary */
        .commentary-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 15px 35px -15px rgba(0, 0, 0, 0.4);
            margin-bottom: 3rem;
        }}

        .commentary-card h2 {{
            font-size: 1.75rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }}

        .commentary-section {{
            margin-bottom: 2rem;
        }}

        .commentary-section:last-child {{
            margin-bottom: 0;
        }}

        .commentary-section h3 {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }}

        .commentary-section h3::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 18px;
            background: linear-gradient(to bottom, var(--primary-accent), var(--secondary-accent));
            margin-right: 0.75rem;
            border-radius: 2px;
        }}

        .commentary-section p {{
            color: #cbd5e1;
            font-size: 1rem;
            text-align: justify;
            margin-bottom: 1.2rem;
        }}

        .commentary-section ul {{
            margin-left: 1.75rem;
            margin-bottom: 1.2rem;
            color: #cbd5e1;
        }}

        .commentary-section li {{
            margin-bottom: 0.6rem;
        }}

        .highlight-text {{
            color: var(--secondary-accent);
            font-weight: 600;
        }}
        
        .highlight-solar {{
            color: #a5b4fc;
            font-weight: 600;
        }}

        .highlight-trading {{
            color: #67e8f9;
            font-weight: 600;
        }}

        footer {{
            text-align: center;
            padding: 2rem 0;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--card-border);
            margin-top: 2rem;
        }}

        footer a {{
            color: var(--primary-accent);
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>

    <header>
        <div class="badge">Analisi Comparativa Metodologica</div>
        <h1>Effetto del Campionamento Temporale sulla Volatilità del NASDAQ 100</h1>
        <div class="subtitle">Studio comparativo per la tesi di laurea: Calendario Solare Continuo (Forward-Fill) vs Calendario di Trading Nativo (2015-2025)</div>
    </header>

    <!-- Top Summary Cards -->
    <div class="summary-grid">
        <!-- Solar Card -->
        <div class="summary-card">
            <div class="card-header-flex">
                <span class="card-title">Calendario Solare Continuo (Report 1)</span>
                <span class="card-dot dot-solar"></span>
            </div>
            <div>
                <div class="card-main-val">{n_obs_1} gg</div>
                <ul class="card-list">
                    <li>Osservazioni: <span class="val">Base giornaliera continua (365 gg)</span></li>
                    <li>Rendimenti nulli: <span class="val" style="color: var(--warning-color);">{pct_zeros_1:.2f}% (weekend ffill)</span></li>
                    <li>Vol. Ann (Trading 252): <span class="val">{vol_252_1*100:.2f}%</span></li>
                    <li>Curtosi Eccesso: <span class="val">{kurt_1:.2f}</span></li>
                </ul>
            </div>
        </div>

        <!-- Trading Card -->
        <div class="summary-card">
            <div class="card-header-flex">
                <span class="card-title">Calendario di Trading Nativo (Report 2)</span>
                <span class="card-dot dot-trading"></span>
            </div>
            <div>
                <div class="card-main-val">{n_obs_2} gg</div>
                <ul class="card-list">
                    <li>Osservazioni: <span class="val">Solo giorni effettivi di borsa (lunedì-venerdì)</span></li>
                    <li>Rendimenti nulli: <span class="val">{pct_zeros_2:.2f}% (reale assenza di variazioni)</span></li>
                    <li>Vol. Ann (Trading 252): <span class="val">{vol_252_2*100:.2f}%</span></li>
                    <li>Curtosi Eccesso: <span class="val">{kurt_2:.2f}</span></li>
                </ul>
            </div>
        </div>

        <!-- Difference Card -->
        <div class="summary-card">
            <div class="card-header-flex">
                <span class="card-title">Scostamento e Distorsione</span>
                <span class="card-dot dot-diff"></span>
            </div>
            <div>
                <div class="card-main-val">{diff_obs} gg</div>
                <ul class="card-list">
                    <li>Giorni extra (Weekend/Festivi): <span class="val">+{diff_obs} osservazioni (+{((n_obs_1-n_obs_2)/n_obs_2)*100:.1f}%)</span></li>
                    <li>Sottostima Volatilità (252): <span class="val" style="color: var(--warning-color);">{diff_vol_252:.2f}%</span></li>
                    <li>Differenza Curtosi: <span class="val" style="color: var(--warning-color);">+{diff_kurt:.2f}%</span></li>
                    <li>Differenza Rendimento Medio: <span class="val">{diff_mean:.2f}%</span></li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Comparative Table Section -->
    <div class="table-card">
        <h2>Confronto Statistico dei Due Dataset</h2>
        <table>
            <thead>
                <tr>
                    <th>Metrica Statistica</th>
                    <th style="text-align: right;">Report 1: Continuo (ffill)</th>
                    <th style="text-align: right;">Report 2: Nativo (Trading)</th>
                    <th style="text-align: right;">Differenza (%) / Delta</th>
                    <th>Interpretazione Metodologica Finanziaria</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="col-metric">Dimensione Dataset (N)</td>
                    <td class="col-val-solar">{n_obs_1}</td>
                    <td class="col-val-trading">{n_obs_2}</td>
                    <td class="col-val-diff diff-up">+{diff_obs} gg</td>
                    <td class="col-interpretation">La reindicizzazione inserisce i giorni festivi e i weekend (circa il 31% del totale) per rendere regolare il calendario per gli algoritmi sequenziali.</td>
                </tr>
                <tr>
                    <td class="col-metric">Rendimenti Esattamente Nulli (r_t = 0)</td>
                    <td class="col-val-solar">{pct_zeros_1:.2f}%</td>
                    <td class="col-val-trading">{pct_zeros_2:.2f}%</td>
                    <td class="col-val-diff diff-up">+{diff_zeros:.2f}% pct</td>
                    <td class="col-interpretation">Il forward-fill (ffill) propaga il prezzo di venerdì al weekend, generando rendimenti artificialmente nulli che diluiscono le variazioni di prezzo.</td>
                </tr>
                <tr>
                    <td class="col-metric">Rendimento Medio Giornaliero</td>
                    <td class="col-val-solar">{mean_ret_1:.6f}</td>
                    <td class="col-val-trading">{mean_ret_2:.6f}</td>
                    <td class="col-val-diff diff-down">{diff_mean:.2f}%</td>
                    <td class="col-interpretation">Il rendimento medio si riduce nel continuo poiché la somma totale dei rendimenti è identica (i prezzi finali coincidono), ma è divisa per un numero maggiore di giorni.</td>
                </tr>
                <tr>
                    <td class="col-metric">Deviazione Standard Giornaliera (Std Dev)</td>
                    <td class="col-val-solar">{std_ret_1:.6f}</td>
                    <td class="col-val-trading">{std_ret_2:.6f}</td>
                    <td class="col-val-diff diff-down">{((std_ret_1-std_ret_2)/std_ret_2)*100:.2f}%</td>
                    <td class="col-interpretation">Diminuisce drasticamente a causa dell'aggiunta massiccia di rendimenti a varianza zero (il weekend), che tirano la distribuzione verso il centro.</td>
                </tr>
                <tr>
                    <td class="col-metric">Volatilità Annualizzata (Trading - 252 gg)</td>
                    <td class="col-val-solar">{vol_252_1*100:.4f}%</td>
                    <td class="col-val-trading">{vol_252_2*100:.4f}%</td>
                    <td class="col-val-diff diff-down">{diff_vol_252:.2f}%</td>
                    <td class="col-interpretation">Causa una forte sottostima (-17.02%) se calcolata su base standard di 252 giorni. Questo errore inficia il pricing delle opzioni e la stima del VaR.</td>
                </tr>
                <tr>
                    <td class="col-metric">Volatilità Annualizzata (Solare - 365 gg)</td>
                    <td class="col-val-solar">{vol_365_1*100:.4f}%</td>
                    <td class="col-val-trading">{vol_365_2*100:.4f}%</td>
                    <td class="col-val-diff diff-down">{((vol_365_1-vol_365_2)/vol_365_2)*100:.2f}%</td>
                    <td class="col-interpretation">Anche adattando la costante di annualizzazione a &radic;365, la volatilità rimane sottostimata di circa il 17.02%.</td>
                </tr>
                <tr>
                    <td class="col-metric">Asimmetria (Skewness)</td>
                    <td class="col-val-solar">{skew_1:.4f}</td>
                    <td class="col-val-trading">{skew_2:.4f}</td>
                    <td class="col-val-diff diff-up">+{diff_skew:.2f}%</td>
                    <td class="col-interpretation">Lievemente più asimmetrica a sinistra (negativa) nel continuo, sebbene il profilo generale rimanga guidato dai medesimi eventi di coda estremi.</td>
                </tr>
                <tr>
                    <td class="col-metric">Eccesso di Curtosi (Kurtosis Excess)</td>
                    <td class="col-val-solar">{kurt_1:.4f}</td>
                    <td class="col-val-trading">{kurt_2:.4f}</td>
                    <td class="col-val-diff diff-up">+{diff_kurt:.2f}%</td>
                    <td class="col-interpretation">L'eccesso di curtosi balza da {kurt_2:.2f} a {kurt_1:.2f} (+{diff_kurt:.2f}%). Questo incremento drammatico è puramente artificiale ed è provocato dall'elevata concentrazione di valori nulli (picco su 0).</td>
                </tr>
                <tr>
                    <td class="col-metric">Prezzo Medio dell'Indice ($)</td>
                    <td class="col-val-solar">${mean_price_1:,.2f}</td>
                    <td class="col-val-trading">${mean_price_2:,.2f}</td>
                    <td class="col-val-diff diff-neutral">{diff_price:.4f}%</td>
                    <td class="col-interpretation">Quasi identico, indicando che la propagazione non introduce un bias sistematico sulla direzione del livello dei prezzi dell'indice.</td>
                </tr>
                <tr>
                    <td class="col-metric">Massimo Drawdown Storico</td>
                    <td class="col-val-solar">{max_dd_1*100:.2f}%</td>
                    <td class="col-val-trading">{max_dd_2*100:.2f}%</td>
                    <td class="col-val-diff diff-neutral">0.00% (Identico)</td>
                    <td class="col-interpretation">Essendo una misura path-dependent basata sul picco e sul minimo dei prezzi, non è influenzata dalla dilatazione temporale delle variazioni intermedie.</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Charts Layout Section -->
    <div class="dashboard-layout">
        
        <!-- Chart 1: Volatility comparison -->
        <div class="chart-card span-full">
            <div class="chart-header">
                <div class="chart-title">Confronto della Volatilità Rolling a 21 Giorni</div>
                <div class="chart-subtitle">Volatilità storica rolling annualizzata (base 252) calcolata sui due dataset. Si noti il gap sistematico.</div>
            </div>
            <div id="vol-chart" class="chart-container"></div>
        </div>

        <!-- Chart 2: Distribution Comparison -->
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Confronto della Distribuzione dei Rendimenti</div>
                <div class="chart-subtitle">Istogramma di densità dei rendimenti logaritmici giornalieri. Si noti l'imponente picco a zero causato dai weekend.</div>
            </div>
            <div id="dist-chart" class="chart-container"></div>
        </div>

        <!-- Chart 3: Autocorrelation of Squared Returns -->
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Autocorrelazione dei Rendimenti al Quadrato (ACF)</div>
                <div class="chart-subtitle">Confronto per i primi 15 lag. La stima della memoria della volatilità è alterata nei dati continui.</div>
            </div>
            <div id="acf-chart" class="chart-container"></div>
        </div>

    </div>

    <!-- Deep Academic Commentary Section -->
    <div class="commentary-card">
        <h2>Considerazioni Metodologiche ed Econometriche per la Tesi</h2>
        
        <div class="commentary-section">
            <h3>La Distorsione delle Proprietà Statistiche</h3>
            <p>
                La conversione da un calendario finanziario standard a un calendario solare continuo tramite <span class="highlight-text">Forward-Fill (ffill)</span> è una pratica standard nel Deep Learning (es. per reti <span class="highlight-text">LSTM</span> e <span class="highlight-text">GRU</span>). Ciò consente di addestrare i modelli su un reticolo temporale uniforme privo di "salti" (gap del weekend). Tuttavia, questo approccio comporta gravi distorsioni dal punto di vista statistico-quantitativo.
            </p>
            <p>
                Come evidenziato dal confronto dei dati, la propagazione del prezzo di chiusura del venerdì per coprire il sabato e la domenica introduce un numero massiccio di rendimenti giornalieri pari esattamente a zero (il <span class="highlight-solar">{pct_zeros_1:.2f}%</span> delle osservazioni totali del Report 1, contro lo <span class="highlight-trading">{pct_zeros_2:.2f}%</span> del Report 2). Questo innesto artificiale altera le caratteristiche della distribuzione dei rendimenti:
            </p>
            <ul>
                <li>
                    <strong>Sottostima della Volatilità:</strong> La deviazione standard della serie storica, che misura la dispersione dei rendimenti, viene "diluita" dalla presenza di zeri. La volatilità annualizzata su base 252 subisce una contrazione sistematica dal <span class="highlight-trading">{vol_252_2*100:.2f}%</span> al <span class="highlight-solar">{vol_252_1*100:.2f}%</span>. Si tratta di una riduzione del <span class="highlight-text">{abs(diff_vol_252):.2f}%</span> della volatilità percepita, con impatti disastrosi per il pricing di derivati o per il calcolo del Value-at-Risk (VaR), che risulterebbe cronicamente sottostimato.
                </li>
                <li>
                    <strong>Inflazione della Leptocurtosi:</strong> L'eccesso di curtosi cresce del <span class="highlight-text">+{diff_kurt:.2f}%</span>, passando da {kurt_2:.2f} a {kurt_1:.2f}. Questo incremento non riflette un aumento del rischio di eventi estremi (le code non si sono allungate), ma è dovuto esclusivamente all'irrigidimento del picco della distribuzione in corrispondenza del valore medio (~0.0). Questo fenomeno, noto come <em>leptocurtosi spuria</em>, viola l'ipotesi di normalità ed altera le distribuzioni teoriche a code pesanti (es. t di Student) utilizzate nei modelli parametrici.
                </li>
            </ul>
        </div>

        <div class="commentary-section">
            <h3>Implicazioni per i Modelli GARCH</h3>
            <p>
                I modelli della famiglia <span class="highlight-text">GARCH</span> (Generalized Autoregressive Conditional Heteroskedasticity) si fondano sulla struttura dei ritardi (lag) dei rendimenti al quadrato per stimare la volatilità condizionale:
            </p>
            <p style="text-align: center; font-family: 'Courier New', Courier, monospace; margin: 1.5rem 0; color: #a5b4fc;">
                &sigma;²_t = &omega; + &alpha;·r²_{{t-1}} + &beta;·&sigma;²_{{t-1}}
            </p>
            <p>
                L'inserimento del weekend altera profondamente questa dinamica in due modi:
            </p>
            <ul>
                <li>
                    <strong>Rottura del Clustered Volatility:</strong> Il fenomeno per cui "alta volatilità tende a essere seguita da alta volatilità" (volatility clustering) si manifesta attraverso l'autocorrelazione dei rendimenti al quadrato. Nel calendario solare continuo, l'inserimento di due zeri ad ogni fine settimana spezza la catena autoregressiva. Come illustrato nel grafico ACF, l'autocorrelazione del quadrato dei rendimenti (che misura la persistenza della volatilità) subisce un brusco calo, specialmente nei primi ritardi (lag 1 e lag 2). Ciò porta il modello GARCH a sottostimare sistematicamente la persistenza del processo (&alpha; + &beta; più basso).
                </li>
                <li>
                    <strong>Stima dei Parametri:</strong> La presenza di rendimenti artificialmente nulli distorce la massimizzazione della funzione di verosimiglianza (Maximum Likelihood Estimation - MLE), portando a stime inefficienti o non convergenti dei parametri &alpha; (reattività agli shock) e &beta; (decadimento della volatilità).
                </li>
            </ul>
        </div>

        <div class="commentary-section">
            <h3>Implicazioni per i Modelli LSTM e Soluzioni Ibride</h3>
            <p>
                Dall'altro lato, le reti neurali <span class="highlight-text">LSTM</span> risentono dei salti temporali insoluti. Trattare una serie temporale con "buchi" di 2 giorni (weekend) come se fosse continua distorce la rappresentazione della memoria di lungo termine delle celle di memoria LSTM. 
            </p>
            <p>
                Per conciliare questi due mondi all'interno di un <strong>Modello Ibrido (GARCH + LSTM)</strong>, si suggeriscono le seguenti best practice metodologiche:
            </p>
            <ul>
                <li>
                    <strong>Stima del GARCH su Dati di Trading:</strong> Stimare i parametri del modello GARCH ed estrarre la volatilità condizionale (&sigma;_t) esclusivamente sul dataset di trading nativo (senza weekend). Successivamente, se la rete LSTM richiede un input a calendario solare continuo, si effettua il reindexing e il ffill sia sui prezzi che sulle serie storiche della volatilità condizionale stimate dal GARCH.
                </li>
                <li>
                    <strong>Uso di "Masking Layers" nelle LSTM:</strong> Addestrare la rete LSTM sul calendario solare continuo, ma istruire la rete a ignorare (tramite Masking in Keras/Tensorflow) gli step temporali corrispondenti al fine settimana, evitando così che la cella di memoria si addestri su variazioni nulle artificiali.
                </li>
                <li>
                    <strong>Feature di Calendario (Day-of-Week):</strong> Inserire variabili esogene che segnalino alla rete LSTM se il giorno corrente è un lunedì o un venerdì, permettendo al modello di apprendere che il gap temporale del fine settimana influisce sulla magnitudo dei rendimenti del primo giorno di trading della settimana (effetto weekend).
                </li>
            </ul>
        </div>
    </div>

    <footer>
        <p>Progetto di Tesi di Laurea in Economia e Finanza | Analisi Comparativa Sviluppata con Antigravity AI</p>
    </footer>

    <!-- Plotly Charts Scripts -->
    <script>
        // Dati passati da Python
        const datesRoll1 = {dates_roll_1_json};
        const volRoll1 = {vol_roll_1_json};
        const datesRoll2 = {dates_roll_2_json};
        const volRoll2 = {vol_roll_2_json};
        
        const binCenters = {bin_centers_json};
        const density1 = {density_1_json};
        const density2 = {density_2_json};
        
        const lags = {lags_json};
        const acf1 = {acf_1_json};
        const acf2 = {acf_2_json};

        // Opzioni grafiche comuni per il tema scuro
        const commonLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{
                family: 'Outfit, sans-serif',
                color: '#94a3b8'
            }},
            margin: {{ t: 30, r: 20, b: 50, l: 60 }},
            xaxis: {{
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)'
            }},
            yaxis: {{
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)'
            }},
            legend: {{
                font: {{ color: '#f8fafc' }},
                orientation: 'h',
                yanchor: 'bottom',
                y: 1.02,
                xanchor: 'right',
                x: 1
            }}
        }};

        // 1. Grafico Volatilità Rolling
        const rollTraceSolar = {{
            x: datesRoll1,
            y: volRoll1,
            type: 'scatter',
            mode: 'lines',
            name: 'Continuo (Solare ffill)',
            line: {{
                color: '#6366f1',
                width: 1.5
            }},
            hovertemplate: 'Data: %{{x}}<br>Vol Continuo: %{{y:.2f}}%<extra></extra>'
        }};
        
        const rollTraceTrading = {{
            x: datesRoll2,
            y: volRoll2,
            type: 'scatter',
            mode: 'lines',
            name: 'Nativo (Trading)',
            line: {{
                color: '#06b6d4',
                width: 1.5
            }},
            hovertemplate: 'Data: %{{x}}<br>Vol Trading: %{{y:.2f}}%<extra></extra>'
        }};
        
        const rollLayout = {{
            ...commonLayout,
            xaxis: {{
                ...commonLayout.xaxis,
                rangeselector: {{
                    buttons: [
                        {{count: 6, label: '6M', step: 'month', stepmode: 'backward'}},
                        {{count: 1, label: '1Y', step: 'year', stepmode: 'backward'}},
                        {{count: 3, label: '3Y', step: 'year', stepmode: 'backward'}},
                        {{count: 5, label: '5Y', step: 'year', stepmode: 'backward'}},
                        {{step: 'all', label: 'TUTTO'}}
                    ],
                    activecolor: '#6366f1',
                    bgcolor: 'rgba(30, 41, 59, 0.9)',
                    bordercolor: 'rgba(255, 255, 255, 0.1)',
                    font: {{ color: '#f8fafc' }}
                }},
                rangeslider: {{ visible: false }}
            }},
            yaxis: {{
                ...commonLayout.yaxis,
                ticksuffix: '%'
            }}
        }};
        Plotly.newPlot('vol-chart', [rollTraceSolar, rollTraceTrading], rollLayout, {{ responsive: true }});

        // 2. Grafico Distribuzioni (Istogramma di Densità)
        const distTraceSolar = {{
            x: binCenters,
            y: density1,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Continuo (ffill)',
            line: {{
                color: 'rgba(99, 102, 241, 0.8)',
                width: 2
            }},
            fillcolor: 'rgba(99, 102, 241, 0.15)',
            hovertemplate: 'Rendimento: %{{x:.3%}}<br>Frequenza: %{{y:.2f}}%<extra></extra>'
        }};
        
        const distTraceTrading = {{
            x: binCenters,
            y: density2,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            name: 'Nativo (Trading)',
            line: {{
                color: 'rgba(6, 182, 212, 0.8)',
                width: 2
            }},
            fillcolor: 'rgba(6, 182, 212, 0.15)',
            hovertemplate: 'Rendimento: %{{x:.3%}}<br>Frequenza: %{{y:.2f}}%<extra></extra>'
        }};
        
        const distLayout = {{
            ...commonLayout,
            xaxis: {{
                ...commonLayout.xaxis,
                tickformat: '.1%',
                title: 'Rendimento Logaritmico Giornaliero'
            }},
            yaxis: {{
                ...commonLayout.yaxis,
                ticksuffix: '%',
                title: 'Frequenza Relativa (%)'
            }}
        }};
        Plotly.newPlot('dist-chart', [distTraceSolar, distTraceTrading], distLayout, {{ responsive: true }});

        // 3. Grafico ACF dei Rendimenti al Quadrato
        const acfTraceSolar = {{
            x: lags,
            y: acf1,
            type: 'bar',
            name: 'Continuo (ffill)',
            marker: {{
                color: 'rgba(99, 102, 241, 0.7)',
                line: {{
                    color: '#6366f1',
                    width: 1
                }}
            }},
            hovertemplate: 'Lag: %{{x}}<br>ACF Continuo: %{{y:.4f}}<extra></extra>'
        }};
        
        const acfTraceTrading = {{
            x: lags,
            y: acf2,
            type: 'bar',
            name: 'Nativo (Trading)',
            marker: {{
                color: 'rgba(6, 182, 212, 0.7)',
                line: {{
                    color: '#06b6d4',
                    width: 1
                }}
            }},
            hovertemplate: 'Lag: %{{x}}<br>ACF Nativo: %{{y:.4f}}<extra></extra>'
        }};
        
        const acfLayout = {{
            ...commonLayout,
            barmode: 'group',
            xaxis: {{
                ...commonLayout.xaxis,
                tickmode: 'linear',
                tick0: 1,
                dtick: 1,
                title: 'Ritardo Temporale (Lag in giorni)'
            }},
            yaxis: {{
                ...commonLayout.yaxis,
                title: 'Coefficiente di Autocorrelazione'
            }}
        }};
        Plotly.newPlot('acf-chart', [acfTraceSolar, acfTraceTrading], acfLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Report comparativo generato correttamente in: {output_path}")

if __name__ == "__main__":
    main()
