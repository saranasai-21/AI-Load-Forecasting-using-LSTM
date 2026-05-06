# =========================================================
# ADVANCED LOAD FORECASTING USING BIDIRECTIONAL LSTM
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "Downloads/ekpc_usage.csv"
)
print("\nDataset Loaded Successfully")

# =========================================================
# 2. DATETIME PROCESSING
# =========================================================

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

# Time Features
df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Weekend'] = (
    df['Day_of_Week'] >= 5
).astype(int)
df['Hour_sin'] = np.sin(
    2 * np.pi * df['Hour'] / 24
)
df['Hour_cos'] = np.cos(
    2 * np.pi * df['Hour'] / 24
)
df['Rolling_Mean_24'] = (
    df['EKPC_MW']
    .rolling(window=24)
    .mean()
)
df['Rolling_Std_24'] = (
    df['EKPC_MW']
    .rolling(window=24)
    .std()
)
# Drop missing rows
df = df.dropna().reset_index(drop=True)

# =========================================================
# 4. FEATURE SELECTION
# =========================================================

features = [
    'EKPC_MW',
    'Hour_sin',
    'Hour_cos',
    'Day_of_Week',
    'Weekend',
    'Rolling_Mean_24',
    'Rolling_Std_24'
]
data = df[features]

# =========================================================
# 5. SCALING
# =========================================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================================================
# 6. CREATE TRUE LSTM SEQUENCES
# =========================================================
# Using Previous 24 Hours
# Predict Next Hour
# =========================================================

SEQUENCE_LENGTH = 24

X = []
y = []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(
        scaled_data[i-SEQUENCE_LENGTH:i]
    )
    y.append(
        scaled_data[i, 0]
    )
X = np.array(X)
y = np.array(y)
print("\nSequence Data Shape")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# =========================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[
    train_size:train_size + val_size
]
y_val = y[
    train_size:train_size + val_size
]
X_test = X[
    train_size + val_size:
]
y_test = y[
    train_size + val_size:
]
print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Testing Shape:", X_test.shape)

# =========================================================
# 8. BUILD ADVANCED MODEL
# =========================================================

model = Sequential([
    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        ),
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),
    Dropout(0.3),
    Bidirectional(
        LSTM(64)
    ),
    Dropout(0.3),
    Dense(
        32,
        activation='relu'
    ),
    Dense(
        16,
        activation='relu'
    ),
    Dense(1)
])

# =========================================================
# 9. COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss='mse',
    metrics=['mae']
)
print("\n================ MODEL SUMMARY ================\n")
model.summary()

# =========================================================
# 10. CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# 11. TRAIN MODEL
# =========================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=30,
    batch_size=32,
    callbacks=[
        early_stop,
        reduce_lr
    ],
    verbose=1
)

# =========================================================
# 12. PREDICTIONS
# =========================================================

y_pred_scaled = model.predict(X_test)

# =========================================================
# 13. INVERSE TRANSFORM
# =========================================================

# Create dummy arrays
# because scaler expects full feature count

dummy_pred = np.zeros(
    (len(y_pred_scaled), len(features))
)

dummy_test = np.zeros(
    (len(y_test), len(features))
)

# Insert predictions into load column
dummy_pred[:, 0] = y_pred_scaled.flatten()

dummy_test[:, 0] = y_test.flatten()

# Inverse transform
y_pred = scaler.inverse_transform(
    dummy_pred
)[:, 0]

y_actual = scaler.inverse_transform(
    dummy_test
)[:, 0]

# =========================================================
# 14. EVALUATION METRICS
# =========================================================

mae = mean_absolute_error(
    y_actual,
    y_pred
)
mse = mean_squared_error(
    y_actual,
    y_pred
)
rmse = np.sqrt(mse)
r2 = r2_score(
    y_actual,
    y_pred
)
mape = np.mean(
    np.abs(
        (y_actual - y_pred) / y_actual
    )
) * 100

print("\n================ MODEL PERFORMANCE ================\n")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# =========================================================
# 15. TRAINING LOSS CURVE
# =========================================================

plt.figure(figsize=(10, 5))
plt.plot(
    history.history['loss'],
    label='Training Loss'
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)
plt.title(
    "Training vs Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)
plt.show()

# =========================================================
# 16. ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(14, 6))
plt.plot(
    y_actual[:300],
    label='Actual Load'
)
plt.plot(
    y_pred[:300],
    label='Predicted Load'
)
plt.title(
    "Actual vs Predicted Load Forecast"
)
plt.xlabel("Time Steps")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)
plt.show()

# =========================================================
# 17. RESIDUAL ANALYSIS
# =========================================================

residuals = y_actual - y_pred
plt.figure(figsize=(12, 5))
plt.plot(residuals)
plt.title(
    "Residual Error Analysis"
)
plt.xlabel("Time Steps")
plt.ylabel("Residual Error")
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)
plt.show()

# =========================================================
# 18. HISTOGRAM OF ERRORS
# =========================================================

plt.figure(figsize=(8, 5))
plt.hist(
    residuals,
    bins=50
)
plt.title(
    "Distribution of Prediction Errors"
)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)
plt.show()

# =========================================================
# 19. SAVE MODEL
# =========================================================

model.save(
    "advanced_ekpc_load_forecasting_model.keras"
)
print("\nModel Saved Successfully")
print(
    "\nSaved as: "
    "advanced_ekpc_load_forecasting_model.keras"
)

import joblib

joblib.dump(
    scaler,
    "scaler.save"
)
