import os
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_process_ndx(start_date="2015-01-01", end_date="2026-01-01", output_dir="data"):
    """
    Scarica i dati storici dell'indice NASDAQ 100 (^NDX) da Yahoo Finance,
    mantiene il calendario di trading nativo (senza inserire date sintetiche),
    calcola le metriche finanziarie chiave (rendimenti logaritmici e range di prezzo)
    e valida l'integrità del dataset rispetto alla fonte prima del salvataggio.
    
    Parameters:
    - start_date (str): Data di inizio del periodo di osservazione (inclusiva).
    - end_date (str): Data di fine del periodo (esclusiva per yfinance, quindi 2026-01-01 per includere il 2025-12-31).
    - output_dir (str): Cartella di destinazione per il salvataggio dei file.
    """
    print(f"Inizio download dei dati per ^NDX dal {start_date} al 2025-12-31...")
    
    # 1. Download dei dati storici tramite yfinance
    # auto_adjust=False mantiene i prezzi di chiusura standard e rettificati separati
    ticker = yf.Ticker("^NDX")
    df_raw = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    
    if df_raw.empty:
        raise ValueError("Nessun dato scaricato. Verificare la connessione internet o il simbolo del ticker.")
        
    print(f"Dati scaricati con successo. Righe totali di trading originali: {len(df_raw)}")
    
    # Creiamo una copia per non modificare il dataframe originale direttamente
    df = df_raw.copy()
    
    # Rimuoviamo eventuali informazioni sul fuso orario dall'indice
    df.index = df.index.tz_localize(None)
    
    # Assicuriamoci che l'indice sia di tipo DatetimeIndex
    df.index = pd.to_datetime(df.index)
    
    # Rimuoviamo colonne non necessarie ed esogene che possono propagare rumore
    df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
    
    # 2. Calcolo dei Rendimenti Logaritmici (Log-Returns)
    # I rendimenti logaritmici sono definiti come r_t = ln(P_t / P_{t-1}).
    # Utilizziamo 'Adj Close' per riflettere il reale valore economico dell'indice rettificato.
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Rimuoviamo la prima riga che contiene NaN a causa dello shift temporale,
    # evitando così l'immissione di dati arbitrari o di riempimento fittizi
    df = df.dropna(subset=['Log_Return'])
    
    # 3. Estrazione e calcolo di altre informazioni utili (Features per LSTM)
    # Volatilità intraday proxy: Parkinson Volatility (basata sul range High-Low)
    # Il range High-Low cattura la dispersione del prezzo all'interno del giorno di negoziazione.
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Log_Range'] = np.log(df['High'] / df['Low'])
    
    # 4. Validazione esplicita dei requisiti di qualità dei dati
    # - Non ci devono essere valori nulli
    assert df.isnull().sum().sum() == 0, "Errore di validazione: Presenti valori nulli (NaN) nel dataset."
    # - Non ci devono essere date duplicate
    assert not df.index.duplicated().any(), "Errore di validazione: Rilevate date duplicate nell'indice."
    # - Fedeltà 1:1 alla fonte: il numero di righe deve essere pari a quello originario meno la prima riga rimossa
    assert len(df) == len(df_raw) - 1, f"Errore di validazione: Il numero di righe ({len(df)}) non corrisponde a quello atteso ({len(df_raw) - 1})."
    
    print("Verifica dei requisiti superata con successo (nessun valore nullo, nessun duplicato, fedeltà 1:1 alla fonte).")
    
    # 5. Reset dell'indice per includere la data come colonna esplicita
    df.index.name = 'Date'
    df_processed = df.reset_index()
    
    # 6. Salvataggio dei risultati
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Cartella '{output_dir}' creata.")
        
    output_path = os.path.join(output_dir, "ndx_processed_data.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"Dataset processato salvato correttamente in: {output_path}")
    
    # Mostriamo alcune statistiche descrittive per validare il processo
    print("\n--- Statistiche Descrittive del Dataset ---")
    print(f"Periodo coperto: dal {df_processed['Date'].min().strftime('%Y-%m-%d')} al {df_processed['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Numero totale di giorni di trading (lavorativi): {len(df_processed)}")
    print(f"Valori nulli residui nel dataset:\n{df_processed.isnull().sum()}")
    print("\nPrime 5 righe del dataset processato:")
    print(df_processed[['Date', 'Adj Close', 'Log_Return', 'High_Low_Range', 'Volume']].head())
    
    return df_processed

if __name__ == "__main__":
    # Avvio del processo di estrazione e salvataggio
    fetch_and_process_ndx()
