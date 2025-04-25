import streamlit as st
from logic.predict_spy import predict_with_spy_model
from logic.retrain_model import retrain_and_predict

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Prediction (Pretrained vs Retrainable LSTM)")

ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL").upper()

if st.button("Predict"):
    col1, col2 = st.columns(2)

    try:
        # --- Predict using SPY-based model (fast) ---
        with col1:
            with st.spinner("SPY-based model is predicting..."):
                spy_prediction = predict_with_spy_model(ticker)
                st.metric("SPY-Based Model", f"${spy_prediction:.2f}")

        # --- Predict using retrainable model (slower) ---
        with col2:
            with st.spinner("Retraining & predicting..."):
                retrain_prediction = retrain_and_predict(ticker)
                st.metric("Retrainable Model", f"${retrain_prediction:.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
