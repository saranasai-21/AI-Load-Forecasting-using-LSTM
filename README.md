# ⚡ EKPC Load Forecasting using Bidirectional LSTM

An advanced deep learning-based electricity load forecasting system developed using **TensorFlow, Bidirectional LSTM, and Streamlit**.

This project predicts next-hour electricity demand using historical load patterns and time-series forecasting techniques.

# Project Overview

Electricity demand forecasting is critical for:

- Power system planning
- Load balancing
- Energy optimization
- Grid stability
- Smart energy management

This project uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network to accurately forecast electricity load based on historical consumption data.

The system also includes a professional **Streamlit dashboard** for real-time forecasting and visualization.

# Deep Learning Architecture

The forecasting model uses:

- Bidirectional LSTM Layers
- Dense Neural Layers
- Dropout Regularization
- Adam Optimizer
- EarlyStopping
- ReduceLROnPlateau

# Model Performance

| Metric & Value |
| MAE - 29.31 |
| MSE - 1505.41 |
| RMSE - 38.80 |
| R² Score - 0.9905 |
| MAPE - 1.89% |

### Interpretation

- The model achieves extremely high forecasting accuracy.
- R² score of 0.9905 indicates excellent trend learning.
- MAPE below 2% demonstrates highly reliable predictions.

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

# Project Structure

Load Forecasting using LSTM
- Requirements.txt
- advanced_ekpc_load_forecasting_model.keras
- app.py
- ekpc_usage.csv
- scaler.save
- train.py

