import numpy as np
import yfinance as yf

def predict_next_n_days(ticker, model, scaler, window_size=60, days_ahead=5):
    try:
        df = yf.download(ticker, period="90d")['Close'].dropna()
    except Exception:
        raise ValueError(f"⚠️ Could not retrieve data for ticker '{ticker.upper()}'. Please check the ticker symbol.")

    if df.empty:
        raise ValueError(f"⚠️ No data found for ticker '{ticker.upper()}'. Please verify the symbol or try another one.")

    if len(df) < window_size:
        raise ValueError(f"⚠️ Ticker '{ticker.upper()}' doesn't have enough recent data (need at least {window_size} days).")

    last_60 = df[-window_size:].values.reshape(-1, 1)
    scaled_seq = scaler.transform(last_60)

    future_predictions = []

    for _ in range(days_ahead):
        input_seq = scaled_seq.reshape((1, window_size, 1))
        pred_scaled = model.predict(input_seq)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        future_predictions.append(pred_price)
        scaled_seq = np.append(scaled_seq[1:], pred_scaled).reshape(-1, 1)

    return future_predictions
