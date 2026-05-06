import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EKPC Load Forecasting Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

@st.cache_resource
def load_artifacts():
    model = load_model(
        "advanced_ekpc_load_forecasting_model.keras"
    )
    scaler = joblib.load(
        "scaler.save"
    )
    return model, scaler
model, scaler = load_artifacts()

# =========================================================
# TITLE
# =========================================================

st.title("⚡ EKPC Smart Load Forecasting Dashboard")
st.markdown("""
Advanced electricity load forecasting using
Bidirectional LSTM Deep Learning.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

st.sidebar.subheader("Model Performance")
st.sidebar.metric("R² Score", "0.9905")
st.sidebar.metric("MAPE", "1.89%")
st.sidebar.metric("RMSE", "38.80")

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    # =====================================================
    # VALIDATION
    # =====================================================

    if 'EKPC_MW' not in df.columns:
        st.error(
            "Dataset must contain 'EKPC_MW' column."
        )
    else:
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(
                df['Datetime']
            )

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        df['Hour'] = np.arange(len(df)) % 24
        df['Day_of_Week'] = (
            np.arange(len(df)) // 24
        ) % 7
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
            .rolling(24)
            .mean()
        )
        df['Rolling_Std_24'] = (
            df['EKPC_MW']
            .rolling(24)
            .std()
        )
        df = df.dropna()

        # =================================================
        # LAST 24 HOURS
        # =================================================

        latest = df.tail(24)
        features = [
            'EKPC_MW',
            'Hour_sin',
            'Hour_cos',
            'Day_of_Week',
            'Weekend',
            'Rolling_Mean_24',
            'Rolling_Std_24'
        ]
        sequence = latest[features].values

        # =================================================
        # SCALE
        # =================================================

        sequence_scaled = scaler.transform(sequence)
        X_input = np.expand_dims(
            sequence_scaled,
            axis=0
        )

        # =================================================
        # PREDICTION
        # =================================================

        prediction_scaled = model.predict(
            X_input
        )
        dummy = np.zeros((1, 7))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction = scaler.inverse_transform(
            dummy
        )[0, 0]

        # =================================================
        # DISPLAY METRICS
        # =================================================

        st.subheader("Forecast Result")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Predicted Next Hour Load",
            f"{prediction:.2f} MW"
        )
        col2.metric(
            "Last Observed Load",
            f"{latest['EKPC_MW'].iloc[-1]:.2f} MW"
        )
        change = (
            prediction -
            latest['EKPC_MW'].iloc[-1]
        )
        col3.metric(
            "Forecast Change",
            f"{change:.2f} MW"
        )

        # =================================================
        # LOAD STATUS
        # =================================================

        if prediction > 4000:
            st.error(
                "⚠ High Electricity Demand Expected"
            )
        elif prediction > 3000:
            st.warning(
                "⚡ Moderate Electricity Demand"
            )
        else:
            st.success(
                "✅ Normal Electricity Demand"
            )

        # =================================================
        # PLOT LAST 24 HOURS
        # =================================================

        st.subheader("Last 24-Hour Load Trend")
        fig1, ax1 = plt.subplots(
            figsize=(12, 4)
        )
        ax1.plot(
            latest['EKPC_MW'].values
        )
        ax1.set_title(
            "Historical Load"
        )
        ax1.set_xlabel("Hours")
        ax1.set_ylabel("Load (MW)")
        ax1.grid(True)
        st.pyplot(fig1)

        # =================================================
        # FORECAST VISUALIZATION
        # =================================================

        st.subheader(
            "Next-Hour Forecast"
        )
        future_values = list(
            latest['EKPC_MW'].values
        )
        future_values.append(prediction)
        fig2, ax2 = plt.subplots(
            figsize=(12, 4)
        )
        ax2.plot(
            future_values[:-1],
            label='Historical Load'
        )
        ax2.plot(
            [23, 24],
            [
                future_values[-2],
                future_values[-1]
            ],
            linewidth=3,
            label='Forecast'
        )
        ax2.set_title(
            "Forecast Extension"
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Load (MW)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # =================================================
        # STATISTICS
        # =================================================

        st.subheader("Dataset Statistics")
        st.write(
            df['EKPC_MW'].describe()
        )
else:
    st.info(
        "Upload an EKPC CSV dataset to begin forecasting."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption(
    "Deep Learning Electricity Forecasting "
    "using Bidirectional LSTM"
)
