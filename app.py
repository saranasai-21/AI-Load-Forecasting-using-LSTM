# =========================================================
# STREAMLIT LOAD FORECASTING APP
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# =========================================================
# LOAD MODEL + SCALER + METRICS
# =========================================================

model = load_model(
    "load_forecasting_model.keras"
)

scaler = joblib.load(
    "scaler.save"
)

metrics = joblib.load(
    "metrics.save"
)

# =========================================================
# PAGE TITLE
# =========================================================

st.title(
    "⚡ Next Hour Load Forecasting"
)

st.write(
    "Enter the last 24 hour load values."
)

# =========================================================
# SHOW IMPORTANT METRICS
# =========================================================

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{metrics['Accuracy10']:.2f}%"
    )

with col2:

    st.metric(
        "MAE",
        f"{metrics['MAE']:.2f}"
    )

with col3:

    st.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

# =========================================================
# USER INPUTS
# =========================================================

runtime_loads = []

for i in range(24):

    value = st.number_input(
        f"Hour {i+1} Load (MW)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    runtime_loads.append(value)

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Next Hour Load"):

    runtime_loads = np.array(runtime_loads)

    # =====================================================
    # CREATE FEATURES
    # =====================================================

    current_hour = pd.Timestamp.now().hour

    hours = []

    for i in range(24):

        hours.append(
            (current_hour - 23 + i) % 24
        )

    hours = np.array(hours)

    hour_sin = np.sin(
        2 * np.pi * hours / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hours / 24
    )

    rolling_mean = pd.Series(
        runtime_loads
    ).rolling(window=24).mean()

    rolling_mean = rolling_mean.fillna(
        rolling_mean.mean()
    )

    # =====================================================
    # BUILD INPUT FEATURES
    # =====================================================

    runtime_features = np.column_stack([

        runtime_loads,
        rolling_mean,
        hour_sin,
        hour_cos
    ])

    # =====================================================
    # SCALE INPUT
    # =====================================================

    runtime_scaled = scaler.transform(
        runtime_features
    )

    runtime_scaled = np.expand_dims(
        runtime_scaled,
        axis=0
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    pred_scaled = model.predict(
        runtime_scaled
    )

    dummy = np.zeros((1, 4))

    dummy[:, 0] = pred_scaled[0][0]

    prediction = scaler.inverse_transform(
        dummy
    )[0, 0]

    # =====================================================
    # OUTPUT
    # =====================================================

    st.success(
        f"Predicted Next Hour Load: "
        f"{prediction:.2f} MW"
    )
