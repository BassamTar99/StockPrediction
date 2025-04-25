import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load models and scalers from the root models directory
lstm_model = load_model('models/lstm_model.keras')
scaler = joblib.load('models/close_scaler.save')

# Define a predict function for LSTM

def lstm_predict(ticker, date):
    """
    Predict using the LSTM model.

    Parameters:
        ticker (str): Stock ticker symbol.
        date (str): Date for prediction.

    Returns:
        numpy array: Predicted values.
    """
    # Fetch stock data for the given ticker
    df = yf.download(ticker, start="2023-01-01", end=date)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    # Extract and scale 'Close' prices
    close_series = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_series)

    # Prepare the last sequence for prediction
    input_seq = scaled_data[-30:].reshape(1, 30, 1)

    # Predict using the LSTM model
    predictions = lstm_model.predict(input_seq)
    return scaler.inverse_transform(predictions)

# Add the Samir-Streamlit-Final directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Samir-Streamlit-Final'))

# Import prediction functions from different models
from Samir_Streamlit_Final.logic.predict_spy import predict_with_spy_model, predict_next_n_days
from Samir_Streamlit_Final.logic.retrain_model import retrain_and_predict, retrain_and_predict_multi

st.set_page_config(page_title="Stock Market Prediction System", layout="wide")

def main():
    st.title("📈 Stock Market Prediction System")
    st.markdown("### Combining LSTM, News Analysis, and SPY-based predictions")

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters")
        ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL").upper()
        n_days = st.slider("Number of days to predict:", min_value=1, max_value=5, value=1)
        date = st.date_input("Select date:", pd.to_datetime("today"))

        # Advanced options
        with st.expander("Advanced Options"):
            use_news = st.checkbox("Include News Analysis", value=True)
            use_spy = st.checkbox("Include SPY-based Model", value=True)
            use_lstm = st.checkbox("Include LSTM Model", value=True)

    if st.button("Generate Predictions"):
        try:
            # Create three columns for different models
            col1, col2, col3 = st.columns(3)

            # 1. LSTM Model Predictions
            if use_lstm:
                with col1:
                    st.markdown("### 🤖 LSTM Model")
                    with st.spinner("LSTM model is processing..."):
                        # Updated to format NumPy array elements before displaying
                        lstm_predictions = lstm_predict(ticker, date)
                        if lstm_predictions.size > 0:
                            st.metric("LSTM Prediction (Next Day)", f"${lstm_predictions[0][0]:.2f}")
                            if n_days > 1:
                                st.write("Multi-day Predictions:")
                                for i, pred in enumerate(lstm_predictions[1:], 2):
                                    st.write(f"Day {i}: **${pred[0]:.2f}**")

            # 2. SPY-based Model Predictions
            if use_spy:
                with col2:
                    st.markdown("### 📊 SPY-Based Model")
                    with st.spinner("SPY-based model is predicting..."):
                        if n_days == 1:
                            spy_preds = [predict_with_spy_model(ticker)]
                            st.metric("SPY Model (Next Day)", f"${spy_preds[0]:.2f}")
                        else:
                            spy_preds = predict_next_n_days(ticker, n_days=n_days)
                            st.write("Multi-day Predictions:")
                            for i, pred in enumerate(spy_preds, 1):
                                st.write(f"Day {i}: **${pred:.2f}**")

            # 3. News Analysis
            if use_news:
                with col3:
                    st.markdown("### 📰 News Impact Analysis")
                    with st.spinner("Analyzing recent news..."):
                        # Add news analysis integration here
                        st.write("News analysis feature coming soon!")

            # Display historical data
            st.markdown("### 📈 Historical Data")
            df = yf.download(ticker, start="2023-01-01")
            st.line_chart(df['Close'])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()