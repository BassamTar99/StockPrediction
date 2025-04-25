import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib  # Add this import to handle loading the scaler

# Load the saved model and scaler
model = tf.keras.models.load_model('my_model.keras')
scaler = joblib.load('scaler.save')  # Corrected code to load the scaler

# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # User input for stock ticker
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):", "AAPL")

    # User input for prediction date range
    start_date = st.date_input("Start Date:", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date:", value=pd.to_datetime("2025-04-24"))

    if st.button("Predict"):
        # Fetch stock data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the given ticker and date range.")
            return

        # Feature engineering
        df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        for w in (10, 20, 50):
            df_feat[f"SMA_{w}"] = df_feat['Close'].rolling(w).mean()
        rsi_w = 14
        delta = df_feat['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_w).mean()
        avg_loss = loss.rolling(rsi_w).mean()
        rs = avg_gain / avg_loss
        df_feat['RSI'] = 100 - (100 / (1 + rs))
        fast, slow, sig = 12, 26, 9
        ema_fast = df_feat['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df_feat['Close'].ewm(span=slow, adjust=False).mean()
        df_feat['MACD'] = ema_fast - ema_slow
        df_feat['MACD_SIGNAL'] = df_feat['MACD'].ewm(span=sig, adjust=False).mean()
        df_feat.dropna(inplace=True)

        # Scale and prepare data
        data = df_feat.values
        scaled = scaler.transform(data)
        seq_len = 30
        X = []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i, :])
        X = np.array(X)

        # Predict
        predictions = model.predict(X).flatten()
        predictions = predictions * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]  # Inverse scaling

        # Display results
        st.subheader("Predicted Prices")
        df_results = pd.DataFrame({"Date": df.index[-len(predictions):], "Predicted Close": predictions})
        st.write(df_results)

        st.line_chart(df_results.set_index("Date"))

if __name__ == "__main__":
    main()