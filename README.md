# Thesis Project: Volatility Forecasting on Nasdaq100

## Abstract
This repository contains the empirical codebase for an undergraduate thesis in Economics and Finance. The research focuses on a comparative analysis of financial time series volatility forecasting, explicitly evaluating traditional econometric architectures against deep learning paradigms.

## Objective
The core objective is to analyze the daily temporal frame of the Nasdaq100 index over a 10-year horizon, comparing the predictive accuracy of the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model with Recurrent Neural Networks, specifically Long Short-Term Memory (LSTM) models.

## Repository Structure
* `/data`: Contains raw and processed CSV datasets extracted via reliable financial APIs.
* `/scripts`: Python scripts for data preprocessing, exploratory data analysis (EDA), and model training.
* `/models`: Saved weights and configurations of the trained LSTM networks and GARCH parameters.
* `/results`: Output metrics including RMSE, MAE, and R-squared evaluations.

## Execution
This environment is managed utilizing autonomous AI coding agents. Please refer to `agents.md` and `product.md` for strict agent directives and systemic workflows.

## Data Preprocessing Policy
To guarantee the methodological integrity of the predictive analyses (GARCH and LSTM), the project adopts a **Native Trading Calendar** policy:
* **No Synthetic Data**: Weekends and holidays are not inserted into the dataset through temporal reindexing or forward-fill techniques.
* **Prevention of Distortions**: Avoiding synthetic data prevents the injection of artificial zero log-returns (which would spuriously deflate the estimated volatility) and the replication of non-existent trading volumes (which would pollute the exogenous features for the LSTM).
* **Log-Returns Processing**: Log-returns are calculated using adjusted closing prices (`Adj Close`), removing the first undefined observation (`NaN`) without resorting to arbitrary zero injections.