
# Advanced Time Series Forecasting with Neural Networks + XAI


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap

np.random.seed(42)
torch.manual_seed(42)


# 1. DATA GENERATION


timesteps = 1500
t = np.arange(timesteps)

temperature = 20 + 10*np.sin(2*np.pi*t/24) + np.random.normal(0,1.5,timesteps)
humidity = 50 + 20*np.sin(2*np.pi*t/168) + np.random.normal(0,2,timesteps)
wind = 5 + 2*np.sin(2*np.pi*t/48) + np.random.normal(0,1,timesteps)
holiday = np.random.choice([0,1], size=timesteps, p=[0.9,0.1])

trend = 0.01*t
season_daily = 15*np.sin(2*np.pi*t/24)
season_weekly = 20*np.sin(2*np.pi*t/168)
noise = np.random.normal(0,5,timesteps)

load = 100 + trend + season_daily + season_weekly \
       - 0.5*temperature + 0.2*humidity - 2*holiday + noise

df = pd.DataFrame({
    "load": load,
    "temperature": temperature,
    "humidity": humidity,
    "wind": wind,
    "holiday": holiday
})

# Feature Engineering
df["lag_1"] = df["load"].shift(1)
df["lag_24"] = df["load"].shift(24)
df["lag_168"] = df["load"].shift(168)
df["rolling_mean_24"] = df["load"].rolling(24).mean()
df["rolling_std_24"] = df["load"].rolling(24).std()
df["hour_sin"] = np.sin(2*np.pi*(t%24)/24)
df["hour_cos"] = np.cos(2*np.pi*(t%24)/24)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

feature_names = df.columns.tolist()


# 2. TRAIN / TEST SPLIT (NO DATA LEAKAGE)


train_size = int(len(df)*0.85)

train_df = df.iloc[:train_size]
test_df  = df.iloc[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled  = scaler.transform(test_df)


# 3. SEQUENCE CREATION


INPUT_WINDOW = 48
FORECAST_HORIZON = 24

def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - INPUT_WINDOW - FORECAST_HORIZON):
        X.append(data[i:i+INPUT_WINDOW])
        y.append(data[i+INPUT_WINDOW:i+INPUT_WINDOW+FORECAST_HORIZON,0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled)
X_test, y_test = create_sequences(test_scaled)


# 4. LSTM MODEL


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.dropout(out[:,-1,:])
        return self.fc(out)

def train_model(model, X_train, y_train, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train,dtype=torch.float32),
                      torch.tensor(y_train,dtype=torch.float32)),
        batch_size=32, shuffle=False)

    for _ in range(15):
        for xb,yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

def evaluate(y_true,y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse,mae,mape

# 5. HYPERPARAMETER TUNING


param_grid = {
    "hidden_size":[32,64],
    "dropout":[0.1,0.2],
    "lr":[0.001,0.0005]
}

best_score = float("inf")
best_params = None

for params in ParameterGrid(param_grid):

    model = LSTMModel(X_train.shape[2],
                      params["hidden_size"],
                      FORECAST_HORIZON,
                      params["dropout"])

    train_model(model,X_train,y_train,params["lr"])

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_train,dtype=torch.float32)).numpy()

    rmse,_,_ = evaluate(y_train,preds)

    if rmse < best_score:
        best_score = rmse
        best_params = params

print("Best Hyperparameters:", best_params)


# 6. FINAL MODEL TRAINING


final_model = LSTMModel(X_train.shape[2],
                        best_params["hidden_size"],
                        FORECAST_HORIZON,
                        best_params["dropout"])

train_model(final_model,X_train,y_train,
            best_params["lr"])

final_model.eval()
with torch.no_grad():
    lstm_preds = final_model(torch.tensor(X_test,dtype=torch.float32)).numpy()

lstm_rmse,lstm_mae,lstm_mape = evaluate(y_test,lstm_preds)


# 7. SARIMAX BASELINE (WITH EXOGENOUS VARIABLES)


train_exog = train_df.drop(columns=["load"])
test_exog  = test_df.drop(columns=["load"])

sarimax = SARIMAX(train_df["load"],
                  exog=train_exog,
                  order=(2,1,2),
                  seasonal_order=(1,1,1,24))

sarimax_fit = sarimax.fit(disp=False)

sarimax_preds = sarimax_fit.forecast(
    steps=len(test_df),
    exog=test_exog
)

sarimax_rmse = np.sqrt(mean_squared_error(test_df["load"],sarimax_preds))
sarimax_mae = mean_absolute_error(test_df["load"],sarimax_preds)
sarimax_mape = np.mean(np.abs((test_df["load"]-sarimax_preds)
                              /(test_df["load"]+1e-8)))*100


# 8. SHAP EXPLAINABILITY

background = torch.tensor(X_train[:100],dtype=torch.float32)
explainer = shap.DeepExplainer(final_model,background)

test_sample = torch.tensor(X_test[:50],dtype=torch.float32)
shap_values = explainer.shap_values(test_sample)[0]

importance = np.mean(np.abs(shap_values),axis=(0,1))

importance_df = pd.DataFrame({
    "Feature":feature_names,
    "Importance":importance
}).sort_values(by="Importance",ascending=False)

print("\nTop 5 SHAP Features:")
print(importance_df.head())


# 9. FINAL RESULTS

results = pd.DataFrame({
    "Model":["LSTM","SARIMAX"],
    "RMSE":[lstm_rmse,sarimax_rmse],
    "MAE":[lstm_mae,sarimax_mae],
    "MAPE":[lstm_mape,sarimax_mape]
})

print("\nFinal Test Performance:")
print(results)
