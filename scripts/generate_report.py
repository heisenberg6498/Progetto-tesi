import os
import pandas as pd
import numpy as np
import json

def generate_interactive_dashboard(data_path="data/ndx_processed_data.csv", output_path="results/ndx_data_report.html"):
    """
    Legge il file CSV dei dati elaborati del NASDAQ 100, calcola statistiche descrittive
    e genera un report interattivo HTML con grafici Plotly.js e un design premium.
    """
    print(f"Lettura dei dati da {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File di dati non trovato in: {data_path}. Eseguire prima fetch_data.py.")
        
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Calcolo delle statistiche descrittive chiave
    start_date = df['Date'].min().strftime('%d-%m-%Y')
    end_date = df['Date'].max().strftime('%d-%m-%Y')
    total_days = len(df)
    
    # Prezzo di chiusura
    last_price = df['Adj Close'].iloc[-1]
    max_price = df['Adj Close'].max()
    min_price = df['Adj Close'].min()
    mean_price = df['Adj Close'].mean()
    
    # Rendimenti logaritmici
    log_returns = df['Log_Return']
    mean_return = log_returns.mean()
    std_return = log_returns.std()
    
    # Volatilità annualizzata (su base di 252 giorni lavorativi e 365 giorni solari)
    vol_ann_252 = std_return * np.sqrt(252)
    vol_ann_365 = std_return * np.sqrt(365)
    
    # Skewness e Kurtosis
    skewness = log_returns.skew()
    kurtosis = log_returns.kurt() # eccesso di curtosi di pandas (kurtosis di Fisher, normale = 0)
    
    # Massimo Drawdown
    roll_max = df['Adj Close'].cummax()
    drawdowns = df['Adj Close'] / roll_max - 1.0
    max_drawdown = drawdowns.min()
    
    # Prepariamo i dati per l'iniezione nel file HTML in formato JSON
    dates_json = json.dumps(df['Date'].dt.strftime('%Y-%m-%d').tolist())
    close_json = json.dumps(df['Adj Close'].tolist())
    returns_json = json.dumps(df['Log_Return'].tolist())
    range_json = json.dumps(df['Log_Range'].tolist())
    volume_json = json.dumps(df['Volume'].tolist())
    
    # 2. Struttura del file HTML con design premium (dark mode, glassmorphism, Outfit font)
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analisi Esplorativa (EDA) - NASDAQ 100 (^NDX)</title>
    
    <!-- Google Fonts: Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Plotly.js via CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.0.min.js"></script>
    
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --primary-accent: #6366f1; /* Indigo */
            --secondary-accent: #06b6d4; /* Cyan */
            --success-color: #10b981; /* Emerald */
            --warning-color: #f43f5e; /* Rose */
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

        header {{
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }}

        .badge {{
            display: inline-block;
            background: linear-gradient(135deg, var(--primary-accent), var(--secondary-accent));
            color: #ffffff;
            padding: 0.25rem 0.75rem;
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
            background: linear-gradient(135deg, #ffffff 30%, var(--text-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 300;
        }}

        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .kpi-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.3);
        }}

        .kpi-title {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }}

        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }}

        .kpi-subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .kpi-highlight-up {{
            color: var(--success-color);
            font-weight: 600;
        }}

        .kpi-highlight-down {{
            color: var(--warning-color);
            font-weight: 600;
        }}

        /* Dashboard Layout */
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
            -webkit-backdrop-filter: blur(12px);
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
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
        }}

        .chart-subtitle {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .chart-container {{
            width: 100%;
            height: 400px;
        }}

        /* Table & Text Card */
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        @media (min-width: 992px) {{
            .info-grid {{
                grid-template-columns: 2fr 3fr;
            }}
        }}

        .info-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 35px -15px rgba(0, 0, 0, 0.4);
        }}

        .info-card h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.75rem;
        }}

        /* Table styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.95rem;
        }}

        th {{
            color: var(--text-secondary);
            font-weight: 500;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--card-border);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-primary);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}

        .metric-name {{
            font-weight: 500;
        }}

        .metric-value {{
            font-family: 'Courier New', Courier, monospace;
            font-weight: bold;
            text-align: right;
        }}

        /* Analysis text */
        .analysis-text p {{
            margin-bottom: 1.2rem;
            color: #cbd5e1;
            font-size: 1rem;
            text-align: justify;
        }}

        .analysis-text ul {{
            margin-left: 1.5rem;
            margin-bottom: 1.2rem;
            color: #cbd5e1;
        }}

        .analysis-text li {{
            margin-bottom: 0.5rem;
        }}

        .highlight-accent {{
            color: var(--secondary-accent);
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
        <div class="badge">Fase 1: Preparazione Dati & EDA</div>
        <h1>Analisi Esplorativa e Pre-elaborazione del NASDAQ 100</h1>
        <div class="subtitle">Analisi empirica per la tesi di laurea in Data Mining Finanziario (Periodo: 2015 - 2025)</div>
    </header>

    <!-- KPI Metrics Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Periodo di Osservazione</div>
            <div class="kpi-value" style="font-size: 1.4rem; padding: 0.25rem 0;">{start_date} al {end_date}</div>
            <div class="kpi-subtext">Totale giorni solari: <span class="highlight-accent">{total_days}</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Ultimo Prezzo (^NDX)</div>
            <div class="kpi-value">${last_price:,.2f}</div>
            <div class="kpi-subtext">Prezzo Massimo: <span class="kpi-highlight-up">${max_price:,.2f}</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Volatilità Annualizzata</div>
            <div class="kpi-value">{vol_ann_252*100:.2f}%</div>
            <div class="kpi-subtext">Base 252 gg: <span class="highlight-accent">{vol_ann_252*100:.2f}%</span> | Base 365 gg: <span class="highlight-accent">{vol_ann_365*100:.2f}%</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Massimo Drawdown</div>
            <div class="kpi-value kpi-highlight-down">{max_drawdown*100:.2f}%</div>
            <div class="kpi-subtext">Cattura la massima perdita dal picco al minimo storico</div>
        </div>
    </div>

    <!-- Charts Dashboard Grid -->
    <div class="dashboard-layout">
        <!-- Chart 1: Price -->
        <div class="chart-card span-full">
            <div class="chart-header">
                <div class="chart-title">Indice NASDAQ 100 (^NDX) - Prezzo di Chiusura Rettificato</div>
                <div class="chart-subtitle">Andamento storico del prezzo di chiusura giornaliero rettificato dal 2015 al 2025</div>
            </div>
            <div id="price-chart" class="chart-container"></div>
        </div>

        <!-- Chart 2: Log Returns -->
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Rendimenti Logaritmici (Log-Returns)</div>
                <div class="chart-subtitle">Rendimenti giornalieri logaritmici. Nota la stazionarietà della serie storica</div>
            </div>
            <div id="returns-chart" class="chart-container"></div>
        </div>

        <!-- Chart 3: Parkinson Volatility (Log Range) -->
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-title">Range Logaritmico Giornaliero (Volatilità Proxy)</div>
                <div class="chart-subtitle">Calcolato come ln(High / Low), proxy per la volatilità intraday</div>
            </div>
            <div id="range-chart" class="chart-container"></div>
        </div>

        <!-- Chart 4: Volume -->
        <div class="chart-card span-full">
            <div class="chart-header">
                <div class="chart-title">Volume di Scambio Giornaliero</div>
                <div class="chart-subtitle">Volume di scambi dell'indice, fondamentale come feature esogena per LSTM</div>
            </div>
            <div id="volume-chart" class="chart-container"></div>
        </div>
    </div>

    <!-- Info and Analysis Section -->
    <div class="info-grid">
        <!-- Table Card -->
        <div class="info-card">
            <h2>Statistiche Descrittive</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metrica</th>
                        <th>Valore</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="metric-name">Totale Osservazioni (Calendario Continuo)</td>
                        <td class="metric-value">{total_days}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Prezzo Massimo ($)</td>
                        <td class="metric-value">${max_price:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Prezzo Minimo ($)</td>
                        <td class="metric-value">${min_price:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Prezzo Medio ($)</td>
                        <td class="metric-value">${mean_price:,.2f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Rendimento Logaritmico Medio (Giornaliero)</td>
                        <td class="metric-value">{mean_return:.6f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Deviazione Standard (Volatilità Giornaliera)</td>
                        <td class="metric-value">{std_return:.6f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Volatilità Annualizzata (Trading - 252 gg)</td>
                        <td class="metric-value">{(vol_ann_252 * 100):.4f}%</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Volatilità Annualizzata (Solare - 365 gg)</td>
                        <td class="metric-value">{(vol_ann_365 * 100):.4f}%</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Asimmetria (Skewness) dei Rendimenti</td>
                        <td class="metric-value">{skewness:.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Eccesso di Curtosi (Kurtosis Excess)</td>
                        <td class="metric-value">{kurtosis:.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric-name">Massimo Drawdown Storico</td>
                        <td class="metric-value">{(max_drawdown * 100):.2f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Academic Commentary Card -->
        <div class="info-card analysis-text">
            <h2>Considerazioni Metodologiche ed Analisi dei Dati</h2>
            <p>
                La fase di preparazione dei dati per la tesi di laurea si è concentrata sulla creazione di un dataset strutturato e rigoroso, idoneo sia per i modelli econometrici classici (<span class="highlight-accent">GARCH</span>) sia per le architetture di Deep Learning (<span class="highlight-accent">LSTM</span>).
            </p>
            <p>
                Di seguito vengono dettagliate le scelte metodologiche effettuate, in linea con la teoria economica e la pratica empirica di Quantitative Finance:
            </p>
            <ul>
                <li>
                    <strong>Calendario Solare Continuo & Forward-Fill:</strong> Per garantire l'assenza di discontinuità temporali (fondamentale per le reti LSTM che richiedono sequenze temporali regolari), il dataset è stato reindicizzato su base giornaliera continua (compresi i weekend e le festività). I dati mancanti sono stati trattati mediante <span class="highlight-accent">Forward-Fill (ffill)</span>. Questa tecnica propaga l'ultimo prezzo disponibile, modellando l'assenza di scambi senza introdurre <em>look-ahead bias</em> (distorsioni previsionali che utilizzerebbero informazioni future).
                </li>
                <li>
                    <strong>Rendimenti Logaritmici (Log-Returns):</strong> I prezzi storici dell'indice sono stati trasformati in rendimenti logaritmici per raggiungere la stazionarietà della serie storica (media e varianza costanti nel tempo), un prerequisito essenziale per la convergenza dei modelli econometrici di volatilità. I rendimenti logaritmici offrono inoltre la proprietà di additività temporale.
                </li>
                <li>
                    <strong>Volatilità Implicita e Proxy:</strong> Oltre al rendimento, è stato calcolato il <span class="highlight-accent">Log-Range Giornaliero</span> (differenza logaritmica tra High e Low del giorno). Questa metrica funge da eccellente proxy per la volatilità intraday (Parkinson Volatility), catturando informazioni sulla dispersione dei prezzi che il solo prezzo di chiusura non registrerebbe.
                </li>
                <li>
                    <strong>Volume di Scambio come Variabile Esogena:</strong> I volumi storici sono stati integrati nel dataset per agire da feature esogena nei modelli previsionali neurali. Empiricamente, i volumi riflettono l'attività di mercato e sono correlati con i cambiamenti improvvisi nei regimi di volatilità (come previsto dall'ipotesi MDH - Mixture of Distributions Hypothesis).
                </li>
                <li>
                    <strong>Caratteristiche della Distribuzione:</strong> L'eccesso di curtosi positivo (<span class="highlight-accent">{kurtosis:.2f}</span>) evidenzia che la distribuzione dei rendimenti del NASDAQ 100 è <em>leptocurtica</em> (presenta code pesanti o "fat tails"). Ciò conferma la presenza di eventi estremi più frequenti rispetto a una distribuzione normale, giustificando l'impiego di modelli GARCH e LSTM per catturare l'eteroschedasticità condizionale e i cluster di volatilità.
                </li>
            </ul>
        </div>
    </div>

    <footer>
        <p>Progetto di Tesi di Laurea in Economia e Finanza | Sviluppato con Antigravity AI</p>
    </footer>

    <!-- Plotly Charts Scripts -->
    <script>
        // Caricamento dei dati JSON passati da Python
        const dates = {dates_json};
        const prices = {close_json};
        const returns = {returns_json};
        const logRanges = {range_json};
        const volumes = {volume_json};

        // Opzioni grafiche comuni per il tema scuro
        const commonLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{
                family: 'Outfit, sans-serif',
                color: '#94a3b8'
            }},
            margin: {{ t: 20, r: 20, b: 40, l: 60 }},
            xaxis: {{
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                rangeselector: {{
                    buttons: [
                        {{count: 1, label: '1M', step: 'month', stepmode: 'backward'}},
                        {{count: 6, label: '6M', step: 'month', stepmode: 'backward'}},
                        {{count: 1, label: '1Y', step: 'year', stepmode: 'backward'}},
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
                gridcolor: 'rgba(255, 255, 255, 0.05)',
                linecolor: 'rgba(255, 255, 255, 0.1)',
                zerolinecolor: 'rgba(255, 255, 255, 0.1)'
            }}
        }};

        // 1. Grafico Prezzo
        const priceTrace = {{
            x: dates,
            y: prices,
            type: 'scatter',
            mode: 'lines',
            name: 'Prezzo (^NDX)',
            line: {{
                color: '#6366f1',
                width: 2
            }},
            hovertemplate: 'Data: %{{x}}<br>Prezzo: $%{{y:,.2f}}<extra></extra>'
        }};
        const priceLayout = {{ ...commonLayout }};
        Plotly.newPlot('price-chart', [priceTrace], priceLayout, {{ responsive: true }});

        // 2. Grafico Rendimenti Logaritmici
        const returnsTrace = {{
            x: dates,
            y: returns,
            type: 'scatter',
            mode: 'lines',
            name: 'Rendimento Log',
            line: {{
                color: '#06b6d4',
                width: 1
            }},
            hovertemplate: 'Data: %{{x}}<br>Rendimento: %{{y:.6f}}<extra></extra>'
        }};
        const returnsLayout = {{ 
            ...commonLayout,
            xaxis: {{ ...commonLayout.xaxis, rangeselector: null }} // Rimuoviamo il selettore per i grafici piccoli per pulizia
        }};
        Plotly.newPlot('returns-chart', [returnsTrace], returnsLayout, {{ responsive: true }});

        // 3. Grafico Volatilità Proxy (Log Range)
        const rangeTrace = {{
            x: dates,
            y: logRanges,
            type: 'scatter',
            mode: 'lines',
            name: 'Log Range',
            line: {{
                color: '#f43f5e',
                width: 1
            }},
            hovertemplate: 'Data: %{{x}}<br>Range Log: %{{y:.6f}}<extra></extra>'
        }};
        const rangeLayout = {{ 
            ...commonLayout,
            xaxis: {{ ...commonLayout.xaxis, rangeselector: null }}
        }};
        Plotly.newPlot('range-chart', [rangeTrace], rangeLayout, {{ responsive: true }});

        // 4. Grafico Volumi
        const volumeTrace = {{
            x: dates,
            y: volumes,
            type: 'bar',
            name: 'Volume',
            marker: {{
                color: 'rgba(99, 102, 241, 0.4)'
            }},
            hovertemplate: 'Data: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>'
        }};
        const volumeLayout = {{ 
            ...commonLayout,
            xaxis: {{ ...commonLayout.xaxis, rangeselector: null }}
        }};
        Plotly.newPlot('volume-chart', [volumeTrace], volumeLayout, {{ responsive: true }});
    </script>
</body>
</html>
"""
    
    # Evita di sovrascrivere il file se esiste già, incrementando il nome del file
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        counter = 1
        new_output_path = f"{base}_{counter}{ext}"
        while os.path.exists(new_output_path):
            counter += 1
            new_output_path = f"{base}_{counter}{ext}"
        print(f"Il file '{output_path}' esiste già. Il report verrà salvato come '{new_output_path}' per evitare sovrascritture.")
        output_path = new_output_path

    # Crea cartella dei risultati se non esiste
    results_dir = os.path.dirname(output_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Cartella '{results_dir}' creata.")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Report interattivo HTML generato correttamente in: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera un report HTML interattivo per ^NDX.")
    parser.add_argument("--data_path", type=str, default="data/ndx_processed_data.csv", help="Percorso del file CSV dei dati.")
    parser.add_argument("--output_path", type=str, default="results/ndx_data_report.html", help="Percorso del file HTML di output.")
    args = parser.parse_args()
    
    generate_interactive_dashboard(data_path=args.data_path, output_path=args.output_path)

