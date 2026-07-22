---
"title": "script python e analisi tecnica per progetto tesi di laurea"
"description": "Serie di script python per sviluppo progetto tesi"
"summary": Script per: ottenimento di dataset utili alla comparazione di modelli statistici nella valutazione della volatilità finanizaria, sviluppo modello GARCH per analisi delle serie storiche, modello neurale LSTM e ibridazione di entrambe"
"audience": ["humans", "agents"]
"date": "11/07/2026"
---

# Scope

The main purpose of this project is the comparation between different statistical models, used to predict volatily forecasting in finance.

Specifically, we'll compare a more "traditional" statistical model, GARCH (Autoregressive conditional heteroskedasticity), a really usefull tool designed to handle situations where volatility (the amount of variation in returns) changes from period to period, and a more modern approach derived from neural networks, called LSTM (long-short term memory) ,an improved version of the Recurrent Neural Network (RNN) designed to capture long-term dependencies in sequential data, that uses a memory cell to store information over time, solving the limitations of traditional RNNs.

There will also be a third type of model, which is an hybridation of GARCH and LSTM, to see if the overall performance will increase.

# Product Specifications: LSTM Architecture for Volatility Forecasting

## 1. Architectural Mandate
This document establishes the unalterable topological parameters for the Long Short-Term Memory (LSTM) neural network. The automated agent must construct the Python pipeline strictly adhering to these network specifications to model the conditional variance of the financial index.

## 2. Hyperparameter Configuration
Based on established empirical literature for financial time series, the neural network must be instantiated with the following exact constraints:
*   **Input Tensor**: Configured for sequence processing utilizing a strict 10-day look-back window to capture short-term memory effects without absorbing circular dependencies.
*   **Hidden Layer**: Exactly ONE (1) LSTM layer containing 64 neurons. Do not construct deep/stacked LSTM architectures to prevent extreme overfitting on financial noise.
*   **Regularization Phase**: A Dropout layer with a rate of 0.2 must sequentially follow the primary LSTM layer.
*   **Output Layer**: Exactly ONE (1) Dense layer emitting a single continuous linear value (the point-forecast of the volatility).

## 3. Training & Validation Protocol
*   **Loss Function & Optimization**: Employ Mean Squared Error (MSE) as the primary loss function, utilizing the Adam optimizer. Monitor the Mean Absolute Percentage Error (MAPE) during validation.
*   **Cross-Validation**: Implement a strict Walk-Forward Validation scheme. Random train-test splits are absolutely forbidden due to the temporal dependencies inherent in the index data. Data scaling (e.g., MinMaxScaler) must be computed solely on the training fold to prevent data leakage into the out-of-sample test sets.