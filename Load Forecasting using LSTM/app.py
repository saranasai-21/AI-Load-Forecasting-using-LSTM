import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("ekpc_lstm_model.h5")

st.title("⚡ EKPC Load Forecasting using LSTM")

st.write("Enter the required inputs to predict next-hour electricity load.")

# ---- User Inputs ----
lag168 = st.number_input("Load same hour last week (Lag168 MW)", value=3000.0)
lag24 = st.number_input("Load same hour yesterday (Lag24 MW)", value=3000.0)
lag1 = st.number_input("Load previous hour (Lag1 MW)", value=3000.0)
hour = st.number_input("Hour", min_value=0, max_value=23)

# ---- Prediction ----
if st.button("Predict Next Hour Load"):

    x_input = np.array([[lag168, lag24, lag1, hour]])

    # scale input
    x_input = scaler.transform(x_input)

    # reshape for LSTM
    x_input = x_input.reshape((1,1,4))

    prediction = model.predict(x_input)

    # inverse scale output
    predicted_load = scaler.inverse_transform(prediction)[0][0]

    st.success(f"Predicted EKPC Load: {predicted_load:.2f} MW")