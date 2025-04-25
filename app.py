import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import joblib  # Add this import to handle loading the scaler

# Load the saved model and scaler
model = tf.keras.models.load_model('lstm_model.keras')
scaler = joblib.load('close_scaler.save')  # Corrected code to load the scaler

# Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Input for stock ticker
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL, TSLA):").upper().strip()

    # Counter for number of days to predict
    days_to_predict = st.number_input("Enter the number of days to predict:", min_value=1, max_value=365, value=30)

    if st.button("Predict"):
        # Fetch stock data
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.Timedelta(days=365 * 5)  # Fetch 5 years of data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the entered stock ticker.")
            return

        # Extract and scale 'Close' prices
        close_series = df['Close'].values.reshape(-1, 1)
        scaled = scaler.transform(close_series).flatten()
        data_min = scaler.data_min_[0]
        data_max = scaler.data_max_[0]

        # Prepare the last sequence for recursive forecasting
        forecast_seq = scaled[-30:].copy()
        forecasts = []

        for _ in range(days_to_predict):
            nxt = model.predict(forecast_seq.reshape(1, 30, 1)).flatten()[0]
            forecasts.append(nxt)
            forecast_seq = np.append(forecast_seq[1:], nxt)

        # Inverse scale the forecasts
        forecasts_uv = [f * (data_max - data_min) + data_min for f in forecasts]

        # Display results
        st.subheader("Forecasted Prices")
        future_dates = pd.date_range(start=df.index[-1], periods=days_to_predict + 1, freq='B')[1:]
        df_results = pd.DataFrame({"Date": future_dates, "Forecasted Close": forecasts_uv})
        st.write(df_results)

        # Plot the results
        st.line_chart(df_results.set_index("Date"))

if __name__ == "__main__":
    main()