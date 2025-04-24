import streamlit as st
from tensorflow.keras.models import load_model
import joblib
from logic.predict_spy import predict_next_n_days

# Load model and scaler
model = load_model("models/spy_lstm_model.h5")
scaler = joblib.load("models/scaler.save")

# Streamlit UI
st.title("📈 Stock Price Predictor (SPY-Based)")

ticker = st.text_input("Enter stock ticker:", "SPY")
days = st.number_input("Days to predict", min_value=1, max_value=10, value=1, step=1)
st.caption("ℹ️ You can predict up to 10 future days using this model.")


if st.button("Predict"):
    try:
        predictions = predict_next_n_days(ticker, model, scaler, days_ahead=days)
        st.subheader(f"Predicted next {days} closing prices for {ticker.upper()}:")
        for i, price in enumerate(predictions, 1):
            st.write(f"Day {i}: ${price:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
