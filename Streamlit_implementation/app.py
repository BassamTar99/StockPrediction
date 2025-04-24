import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time

# Initialize session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "training_progress" not in st.session_state:
    st.session_state.training_progress = 0

# Load pre-trained S&P 500 model
sp500_model = load_model("model.h5")

# Page setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Advanced Stock Price Predictor")
st.markdown("""
    This application offers two prediction models:
    1. **S&P 500 Model**: Pre-trained LSTM model for S&P 500 index predictions
    2. **Custom Stock Model**: Automated LSTM model that trains on your chosen stock
""")

# Model Selection
model_type = st.radio(
    "Select Prediction Model",
    ["S&P 500 Model", "Custom Stock Model"],
    horizontal=True
)

if model_type == "S&P 500 Model":
    st.info("Using pre-trained LSTM model optimized for S&P 500 predictions")
    ticker = "^GSPC"  # S&P 500 ticker
    num_days = st.number_input("Days to Predict", min_value=1, max_value=30, value=1)
    run_prediction = st.button("🔮 Predict S&P 500")
    
    if run_prediction:
        with st.spinner("Generating S&P 500 predictions..."):
            df = yf.download(ticker, period="6mo")
            close_prices = df[['Close']].values
            
            # Normalize prices
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            # Last 30 days input sequence
            input_sequence = scaled_data[-30:].copy()
            predicted_prices = []
            
            for _ in range(num_days):
                input_seq = input_sequence.reshape(1, 30, 1)
                pred_scaled = sp500_model.predict(input_seq, verbose=0)
                pred_price = scaler.inverse_transform(pred_scaled)
                predicted_prices.append(pred_price[0][0])
                input_sequence = np.append(input_sequence, [[pred_scaled[0][0]]], axis=0)[-30:]
            
            # Display results
            st.subheader("📊 S&P 500 Predictions")
            st.metric(label=f"Prediction for Day 1", value=f"${predicted_prices[0]:.2f}")
            
            # Visualization
            fig, ax = plt.subplots()
            past_days = df['Close'][-60:]
            future_days = pd.date_range(start=past_days.index[-1] + pd.Timedelta(days=1), periods=num_days)
            
            ax.plot(past_days.index, past_days.values, label="Past 60 Days")
            ax.plot(future_days, predicted_prices, linestyle="--", color="orange", label="Predicted")
            
            ax.set_title("S&P 500 – Past 60 Days + Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            st.pyplot(fig)
            
            # Save to history
            st.session_state.prediction_history.append({
                "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Model": "S&P 500",
                "Days Predicted": num_days,
                "Day 1 Prediction ($)": f"{predicted_prices[0]:.2f}",
                "Last Price ($)": f"{float(df['Close'].iloc[-1]):.2f}"
            })

else:  # Custom Stock Model
    st.info("This model will automatically train on your chosen stock")
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
        num_days = st.number_input("Days to Predict", min_value=1, max_value=30, value=1)
    
    with col2:
        training_period = st.selectbox(
            "Training Data Period",
            ["1 Year", "2 Years", "5 Years"],
            index=0
        )
        epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50)
    
    train_and_predict = st.button("🚀 Train & Predict")
    
    if train_and_predict:
        with st.spinner("Training model and generating predictions..."):
            # Download data
            period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
            df = yf.download(ticker, period=period_map[training_period])
            
            if df.empty:
                st.error("⚠️ Invalid or unknown ticker symbol. Try again with something like AAPL or TSLA.")
            else:
                # Prepare data
                close_prices = df[['Close']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_prices)
                
                # Create sequences
                X, y = [], []
                for i in range(30, len(scaled_data)):
                    X.append(scaled_data[i-30:i, 0])
                    y.append(scaled_data[i, 0])
                
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Create and train model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(30, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Training progress bar
                progress_bar = st.progress(0)
                for epoch in range(epochs):
                    model.fit(X, y, epochs=1, batch_size=32, verbose=0)
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    time.sleep(0.1)  # Simulate training time
                
                # Generate predictions
                input_sequence = scaled_data[-30:].copy()
                predicted_prices = []
                
                for _ in range(num_days):
                    input_seq = input_sequence.reshape(1, 30, 1)
                    pred_scaled = model.predict(input_seq, verbose=0)
                    pred_price = scaler.inverse_transform(pred_scaled)
                    predicted_prices.append(pred_price[0][0])
                    input_sequence = np.append(input_sequence, [[pred_scaled[0][0]]], axis=0)[-30:]
                
                # Display results
                st.subheader(f"📊 {ticker} Predictions")
                st.metric(label=f"Prediction for Day 1", value=f"${predicted_prices[0]:.2f}")
                
                # Visualization
                fig, ax = plt.subplots()
                past_days = df['Close'][-60:]
                future_days = pd.date_range(start=past_days.index[-1] + pd.Timedelta(days=1), periods=num_days)
                
                ax.plot(past_days.index, past_days.values, label="Past 60 Days")
                ax.plot(future_days, predicted_prices, linestyle="--", color="orange", label="Predicted")
                
                ax.set_title(f"{ticker} – Past 60 Days + Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                st.pyplot(fig)
                
                # Save to history
                st.session_state.prediction_history.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Model": "Custom",
                    "Ticker": ticker,
                    "Days Predicted": num_days,
                    "Day 1 Prediction ($)": f"{predicted_prices[0]:.2f}",
                    "Last Price ($)": f"{float(df['Close'].iloc[-1]):.2f}"
                })

# Display history at the bottom
st.markdown("---")
st.subheader("📁 Prediction History (This Session)")

if st.session_state.prediction_history:
    hist_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(hist_df, use_container_width=True)
    
    csv = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions made yet.")
