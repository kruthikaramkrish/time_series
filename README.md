# Advanced Time Series Forecasting with Neural Networks and Explainable AI (XAI)

## Objective

Develop a deep learning-based time series forecasting system using LSTM,
compare it with a traditional SARIMAX baseline,
and apply Explainable AI techniques (SHAP) to interpret model predictions.

## Dataset

- 1500 time steps
- Multivariate synthetic dataset
- Includes:
  - Trend
  - Daily seasonality (24-hour cycle)
  - Weekly seasonality (168-hour cycle)
  - Random noise

### Features

- Load (target variable)
- Temperature
- Humidity
- Wind speed
- Holiday indicator

### Feature Engineering

- Lag features (1, 24, 168)
- Rolling mean (24)
- Rolling standard deviation (24)
- Cyclical encoding (sin/cos for hourly pattern)

## Methodology

1. Train/test split (85% / 15%)
2. Scaling using MinMaxScaler (fit on training data only)
3. Sequence generation (48 input window, 24-step forecast horizon)
4. LSTM model with hyperparameter tuning
5. Walk-forward validation logic
6. SARIMAX baseline with exogenous variables
7. SHAP DeepExplainer for feature importance analysis

## Models

### LSTM Neural Network
- Multi-step forecasting (24-step horizon)
- Hyperparameter tuning:
  - Hidden size
  - Dropout
  - Learning rate

### SARIMAX Baseline
- Order: (2,1,2)
- Seasonal order: (1,1,1,24)
- Uses exogenous variables

---

## Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

---

## Explainability (XAI)

SHAP DeepExplainer identifies:

- Lag features as dominant predictors
- Strong daily and weekly seasonal patterns
- Moderate environmental feature contributions

This ensures both predictive performance and interpretability.

---

## How to Run

pip install -r requirements.txt
python main.py
