import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# ====== Load Pretrained Model and Scaler (only once) ======
MODEL_PATH = os.path.join("models", "spy_lstm_model.h5")
SCALER_PATH = os.path.join("models", "scaler.save")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ====== Predict Function ======
def predict_with_spy_model(ticker: str, window_size: int = 60) -> float:
    """
    Predict the next closing price for the given ticker using a pretrained SPY-based model.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "TSLA")
        window_size (int): Number of days to use for prediction (default: 60)

    Returns:
        float: Predicted next closing price in USD
    """
    # Download the last 90 days of closing data (in case of weekends/holidays)
    df = yf.download(ticker, period="90d")['Close'].dropna()

    if len(df) < window_size:
        raise ValueError(f"Not enough data to predict for {ticker}. Need at least {window_size} days.")

    # Get the last window_size days
    last_60 = df[-window_size:].values.reshape(-1, 1)

    # Scale the data
    scaled = scaler.transform(last_60)

    # Reshape to (1, 60, 1) for LSTM
    input_seq = scaled.reshape((1, window_size, 1))

    # Predict
    prediction_scaled = model.predict(input_seq)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    return float(prediction)
