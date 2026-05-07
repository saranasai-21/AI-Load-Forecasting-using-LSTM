# =========================================================
# STREAMLIT LOAD FORECASTING APP
# LAST 4 HOURS INPUT
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Load Forecasting",
    page_icon="⚡",
    layout="centered"
)

# =========================================================
# LOAD MODEL + SCALER + METRICS
# =========================================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "load_forecasting_model.keras",
        compile=False
    )

    scaler = joblib.load(
        "scaler.save"
    )

    metrics = joblib.load(
        "metrics.save"
    )

    return model, scaler, metrics

model, scaler, metrics = load_artifacts()

# =========================================================
# TITLE
# =========================================================

st.title(
    "⚡ AI Load Forecasting"
)

st.write(
    "Enter the last 4 hour load values "
    "to predict the next hour load."
)

# =========================================================
# MODEL METRICS
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

st.subheader("📝 Enter Last 4 Hour Loads")

col1, col2 = st.columns(2)

with col1:

    load1 = st.number_input(
        "Hour 1 Load",
        min_value=0.0,
        value=100.0
    )

    load2 = st.number_input(
        "Hour 2 Load",
        min_value=0.0,
        value=100.0
    )

with col2:

    load3 = st.number_input(
        "Hour 3 Load",
        min_value=0.0,
        value=100.0
    )

    load4 = st.number_input(
        "Hour 4 Load",
        min_value=0.0,
        value=100.0
    )

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("🚀 Predict Next Hour Load"):

    # =====================================================
    # CREATE 24-HOUR SEQUENCE
    # =====================================================

    runtime_loads = np.array([

        load1,
        load2,
        load3,
        load4
    ])

    # Repeat values to make 24 inputs
    repeated_loads = np.tile(
        runtime_loads,
        6
    )

    # =====================================================
    # CREATE FEATURES
    # =====================================================

    current_hour = pd.Timestamp.now().hour

    hours = np.array([

        (current_hour - 23 + i) % 24

        for i in range(24)
    ])

    hour_sin = np.sin(
        2 * np.pi * hours / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hours / 24
    )

    rolling_mean = pd.Series(
        repeated_loads
    ).rolling(window=24).mean()

    rolling_mean = rolling_mean.fillna(
        repeated_loads.mean()
    )

    # =====================================================
    # BUILD INPUT FEATURES
    # =====================================================

    runtime_features = np.column_stack([

        repeated_loads,
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
    # PREDICT
    # =====================================================

    pred_scaled = model.predict(
        runtime_scaled,
        verbose=0
    )

    # =====================================================
    # INVERSE TRANSFORM
    # =====================================================

    dummy = np.zeros((1, 4))

    dummy[:, 0] = pred_scaled[0][0]

    prediction = scaler.inverse_transform(
        dummy
    )[0, 0]

    # =====================================================
    # OUTPUT
    # =====================================================

    st.success(
        f"⚡ Predicted Next Hour Load: "
        f"{prediction:.2f} MW"
    )
