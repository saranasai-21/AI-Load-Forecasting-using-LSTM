# TRAIN + SAVE LOAD FORECASTING MODEL

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. LOAD DATASET

df = pd.read_csv("Downloads/ekpc_usage.csv")

# 2. DATETIME PROCESSING

df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df.sort_values('Datetime').reset_index(drop=True)

# 3. FEATURE ENGINEERING

df['Hour'] = df['Datetime'].dt.hour

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)

df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

df['Rolling_Mean_24'] = (df['EKPC_MW'].rolling(window=24).mean())
df = df.dropna().reset_index(drop=True)

# 4. FEATURES

features = [
    'EKPC_MW',
    'Rolling_Mean_24',
    'Hour_sin',
    'Hour_cos'
]

data = df[features]

# 5. SCALE DATA

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 6. CREATE SEQUENCES

SEQUENCE_LENGTH = 24

X = []
y = []
for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# 7. TRAIN TEST SPLIT

train_size = int(0.8 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# 8. BUILD MODEL

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])), Dropout(0.3),
    Bidirectional(LSTM(64)), Dropout(0.3),
    Dense(32, activation='relu'), Dense(1)
])

# 9. COMPILE MODEL

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 10. TRAIN MODEL

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stop], verbose=1)

# 11. PREDICTIONS

y_pred_scaled = model.predict(X_test)

# 12. INVERSE TRANSFORM

dummy_pred = np.zeros((len(y_pred_scaled), len(features)))
dummy_test = np.zeros((len(y_test), len(features)))
dummy_pred[:, 0] = (y_pred_scaled.flatten())
dummy_test[:, 0] = (y_test.flatten())
y_pred = scaler.inverse_transform(dummy_pred)[:, 0]
y_actual = scaler.inverse_transform(dummy_test)[:, 0]

# 13. METRICS

mae = mean_absolute_error(y_actual,y_pred)
mse = mean_squared_error(y_actual,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual,y_pred)
percentage_error = np.abs((y_actual - y_pred) / y_actual) * 100
accuracy_10 = np.mean(percentage_error <= 10) * 100

# 14. PRINT RESULTS

print("MODEL PERFORMANCE")
print("========================================")

print(f"MAE                  : {mae:.2f}")
print(f"MSE                  : {mse:.2f}")
print(f"RMSE                 : {rmse:.2f}")
print(f"R2 Score             : {r2:.4f}")
print(f"Accuracy Within 10%  : {accuracy_10:.2f}%")

# 15. SAVE MODEL + SCALER + METRICS

model.save("load_forecasting_model.keras")
joblib.dump(scaler,"scaler.save")

metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Accuracy10": accuracy_10
}

joblib.dump(metrics,"metrics.save")

print("\n===================================")
print("MODEL FILES SAVED SUCCESSFULLY")
print("1. load_forecasting_model.keras")
print("2. scaler.save")
print("3. metrics.save")
