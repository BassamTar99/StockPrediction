# 📈 Stock Price Prediction App

This project provides a simple web app that predicts future stock prices using two models:
- A Pretrained LSTM Model based on SPY stock data
- A Retrainable LSTM Model that custom-trains for each new stock ticker

The app is built using Streamlit, TensorFlow, Keras, Docker, and yfinance.

---

## 🛠️ How to Run the App (Using Docker)

1. Clone the repository:
   git clone https://github.com/BassamTar99/StockPrediction.git
   cd StockPrediction

2. Make sure Docker is installed:
   Download Docker Desktop: https://www.docker.com/products/docker-desktop

3. Build the Docker image:
   docker build -t streamlit-stock-app .

4. Run the Docker container:
   docker run -p 8501:8501 streamlit-stock-app

5. Open your browser and access the app:
   http://localhost:8501

---

## 📋 Features

- Input any stock ticker (example: AAPL, TSLA, NVDA)
- Predict 1 to 5 future days
- Choose to show/hide Pretrained or Retrainable models
- View historical stock price chart (last 90 days)
- Download prediction history as CSV
- Clear history with one click

---

## 📢 Notes

- The retrainable model takes longer because it retrains a new model for each stock ticker.
- A stable internet connection is required to fetch real-time stock data.

---

## 📬 Contact

GitHub Profile: @samir-44

---
