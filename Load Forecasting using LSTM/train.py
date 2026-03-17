import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.metrics import MeanSquaredError

# 1. importing of dataset
df=pd.read_csv("Downloads/Load Forecasting using LSTM/ekpc_usage.csv")
df.info()

# 2. Data Cleaning
df=df.sort_index()
print(df.isnull().sum())
df_model = df.dropna().reset_index(drop=True)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Hour'] = df['Datetime'].dt.hour
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Day_of_Week'] = df['Datetime'].dt.dayofweek

# 3. Data Preprocessing
df['Lag1'] = df['EKPC_MW'].shift(1)
df['Lag24'] = df['EKPC_MW'].shift(24)       # same hour yesterday
df['Lag168'] = df['EKPC_MW'].shift(168)     # same hour lastweek
df=df.dropna()

scaler_X=MinMaxScaler()
scaler_y=MinMaxScaler()
X = scaler_X.fit_transform(df[['Lag168', 'Lag24', 'Lag1','Hour','Day','Month']])
y = scaler_y.fit_transform(df[['EKPC_MW']])
X = X.reshape((X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]
df=df.dropna()

# 4. Train/Test Split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[MeanSquaredError()]
)
# 6. Train Model and Record History
history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1)

# 7. Evaluate Model
y_pred = model.predict(X_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred)**2)

r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 8. Plot Training Loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("LSTM Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 9. Plot Actual vs Predicted Load
plt.figure(figsize=(12,5))
plt.plot(y_test[:300], label='Actual Load', color='blue')
plt.plot(y_pred[:300], label='Predicted Load', color='red')
plt.title("Actual vs Predicted Load (Test Data)")
plt.xlabel("Time Steps (hours)")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 10. Save Trained Model
model.save("ekpc_lstm_model.h5")
print("\nModel saved as ekpc_lstm_model.h5")