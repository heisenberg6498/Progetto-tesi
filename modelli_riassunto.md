# Riassunto dei Modelli di Forecasting della Volatilità

Descrizione dei modelli statistici e di machine learning utilizzati per il forecasting della volatilità sull'indice NASDAQ 100, con i relativi parametri e variabili.
## 1. Modello GARCH(1,1)

*   **Variabili di Input:** Rendimenti logaritmici giornalieri ($r_t = \ln(P_t / P_{t-1})$). I dati vengono scalati moltiplicandoli per $100$ per favorire la stabilità e la convergenza numerica dell'algoritmo di massima verosimiglianza.
*   **Variabili di Output:** Varianza condizionale stimata ($\sigma_t^2$).
*   **Parametrizzazione e Specifiche:**
    *   **Equazione della Media:** Costante ($r_t = \mu + \epsilon_t$).
    *   **Distribuzione degli Errori:** Normale.
    *   **Architettura (p=1, q=1):** Il modello utilizza 1 ritardo per i residui al quadrato (termine ARCH) e 1 ritardo per la varianza stessa (termine GARCH). L'equazione per la varianza condizionale è:
        $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
    *   **Configurazione del Campione:** Suddivisione strettamente cronologica in $70\%$ Train, $15\%$ Validation, $15\%$ Test. La stima dei parametri ($\omega, \alpha, \beta$) avviene sul set combinato di Train e Validation, mentre le previsioni *out-of-sample* si ottengono ricorsivamente sul Test set (fissando i parametri ex-ante per evitare *look-ahead bias*).

## 2. Modello Neurale LSTM (Long Short-Term Memory)

*   **Variabili di Input (Feature):** Sequenze di dati storici del mercato, specificamente il rendimento logaritmico e il residuo quadratico proxy scalato. I dati subiscono una standardizzazione (`StandardScaler`) calcolata sui set di addestramento, in modo da evitare *data leakage*.
*   **Variabili di Output:** Previsione puntuale e continua della **varianza condizionale** al tempo $t+1$ (spesso riferita in modo generale come forecasting della volatilità).
*   **Parametrizzazione Architetturale Congelata:**
    *   **Tensore di Input (Look-back window):** Finestra temporale impostata a $60$ giorni (time-steps = $60$), fornendo al modello un contesto storico corrispondente a circa tre mesi di negoziazione. Per prevedere la varianza al tempo $t+1$, ogni sequenza di input contiene esclusivamente osservazioni disponibili fino al tempo $t$, evitando così qualsiasi *data leakage*.
    *   **Hidden Layer:** Esattamente $1$ layer LSTM contenente $32$ neuroni. L'utilizzo di una rete compatta funge da "collo di bottiglia", contribuendo a contenere il rischio di overfitting sul rumore finanziario.
    *   **Fase di Regolarizzazione:** $1$ layer di Dropout con un $rate = 0.2$ posto sequenzialmente dopo il layer LSTM.
    *   **Output Layer:** Esattamente $1$ layer Dense contenente un singolo neurone con funzione di attivazione **Softplus** ($\ln(1 + e^x)$), garantendo matematicamente una previsione della varianza strettamente positiva ($h_t > 0$).
*   **Metodologia di Training e Validazione ("Two-Phase Protocol"):**
    *   **Loss Function & Ottimizzatore:** L'addestramento (con ottimizzatore Adam) avviene minimizzando direttamente la loss asimmetrica **QLIKE** (Quasi-Likelihood), la cui espressione matematica implementata è:
        $$QLIKE = \frac{1}{T} \sum_{t=1}^T \left[ \ln(\hat{h}_t) + \frac{y_t}{\hat{h}_t} \right]$$
        dove $y_t$ rappresenta la proxy osservata della varianza al tempo $t$, mentre $\hat{h}_t$ rappresenta la varianza condizionale prevista dal modello. Valori inferiori della QLIKE indicano previsioni migliori. Quando $y_t = \epsilon_t^2$ (residuo quadratico), questa loss è equivalente, a meno di costanti e fattori moltiplicativi, alla *negative Gaussian quasi-log-likelihood* utilizzata nella stima del GARCH.
    *   **Protocollo in Due Fasi:** 
        1. **Selezione:** Addestramento esclusivo sul Train set e valutazione sul Validation set (tramite Early Stopping) per individuare la `best_epoch`. 
        2. **Refit Definitivo:** Costruzione di un nuovo modello e riaddestramento sull'unione di Train e Validation (con nuovo Scaler integrato) per il numero esatto di `best_epoch`. Questo garantisce una totale uguaglianza informativa con il GARCH.
    *   **Valutazione Finale su Metriche Decimale e Multi-Proxy:** Per un confronto perfettamente paritetico con l'econometria, le metriche (QLIKE, MSE, MAE, RMSE, $R^2_{OOS}$) vengono calcolate sulle stime convertite alla scala originale decimale. Prima del calcolo delle metriche finali, le previsioni di varianza vengono riportate alla scala decimale dividendo per $100^2=10000$. Anche le proxy osservate e i benchmark vengono espressi nella medesima scala. 
    *   **Definizione delle Proxy:** L'addestramento della LSTM avviene utilizzando il residuo quadratico, ma in fase di valutazione vengono impiegate due diverse proxy della varianza per misurare le performance in scenari differenti:
        *   **Squared Residual:** $y_t^{SR} = \epsilon_t^2$
        *   **Parkinson Variance:** $y_t^{P} = \frac{1}{4 \ln 2} [\ln(H_t / L_t)]^2$
    *   **Calcolo Metriche Finali:** Oltre alla QLIKE, le prestazioni sono valutate mediante:
        $$MSE = \frac{1}{T} \sum_{t=1}^T (y_t - \hat{h}_t)^2 \quad \quad MAE = \frac{1}{T} \sum_{t=1}^T |y_t - \hat{h}_t|$$
        $$RMSE = \sqrt{MSE} \quad \quad R^2_{OOS} = 1 - \frac{\sum_{t=1}^T (y_t - \hat{h}_t)^2}{\sum_{t=1}^T (y_t - \hat{h}_t^B)^2}$$
        Per QLIKE, MSE, MAE e RMSE, valori inferiori indicano prestazioni migliori. Per l'$R^2_{OOS}$, valori positivi indicano un miglioramento predittivo rispetto al benchmark, zero indica prestazioni equivalenti e valori negativi indicano prestazioni peggiori. In particolare, viene utilizzato come benchmark statico $\hat{h}_t^B$, per ciascuna proxy, il valore medio osservato sul campione combinato Train + Validation. Questo garantisce equità d'informazione con i modelli refittati prima del Test, escludendo ogni *look-ahead bias*.
    *   **Riproducibilità e Analisi di Robustezza:** La stabilità è assicurata fissando l'op-determinism di TensorFlow, i seed di stato Keras prima di ogni costruzione architetturale, e svolgendo un'analisi aggregata conclusiva iterata su 5 seed (es. `11, 22, 33, 42, 55`).

   