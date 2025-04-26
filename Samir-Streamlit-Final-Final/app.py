import streamlit as st
from logic.predict_spy import predict_with_spy_model, predict_next_n_days
from logic.retrain_model import retrain_and_predict, retrain_and_predict_multi

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Prediction (Pretrained vs Retrainable LSTM)")

# --- Initialize prediction history ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- User inputs ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL").upper()
n_days = st.slider("How many future days to predict?", min_value=1, max_value=5, value=1)

if st.button("Predict"):
    col1, col2 = st.columns(2)

    try:
        # --- SPY-based prediction ---
        with col1:
            with st.spinner("SPY-based model is predicting..."):
                if n_days == 1:
                    spy_preds = [predict_with_spy_model(ticker)]
                    st.metric("SPY-Based Model", f"${spy_preds[0]:.2f}")
                else:
                    spy_preds = predict_next_n_days(ticker, n_days=n_days)
                    st.markdown("### 📈 SPY-Based Multi-Day Prediction")
                    for i, p in enumerate(spy_preds, 1):
                        st.write(f"Day {i}: **${p:.2f}**")

        # --- Retrainable prediction ---
        with col2:
            with st.spinner("Retraining & predicting..."):
                if n_days == 1:
                    retrain_preds = [retrain_and_predict(ticker)]
                    st.metric("Retrainable Model", f"${retrain_preds[0]:.2f}")
                else:
                    retrain_preds = retrain_and_predict_multi(ticker, n_days=n_days)
                    st.markdown("### 📈 Retrainable Multi-Day Prediction")
                    for i, p in enumerate(retrain_preds, 1):
                        st.write(f"Day {i}: **${p:.2f}**")

        # --- Save all predictions to history ---
        st.session_state.history.append({
            "Ticker": ticker,
            "SPY-Based": [f"${p:.2f}" for p in spy_preds],
            "Retrainable": [f"${p:.2f}" for p in retrain_preds]
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Show prediction history ---
if st.session_state.history:
    st.subheader("🕓 Prediction History")
    st.dataframe(st.session_state.history[::-1], use_container_width=True)

    if st.button("🧹 Clear History"):
        st.session_state.history.clear()
        st.info("History cleared — click 'Predict' to refresh the table.")
