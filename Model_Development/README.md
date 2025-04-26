# 🧠 Model Development

This folder contains the original Jupyter notebooks used to build, train, and test the machine learning models before integrating them into the Streamlit web app.

## 📚 Contents

- **Submission_LSTM-SPY.ipynb**  
  This notebook builds and trains a stock price prediction model based on SPY (S&P 500 ETF) data.  
  The final pretrained model is later used directly inside the app for fast predictions.

- **Submission_Automated_LSTM.ipynb**  
  This notebook builds an **automated retrainable LSTM model** that performs hyperparameter tuning and retraining for any selected stock ticker.  
  ⚡ **Note:** The exact version of the retrainable model implemented inside the Streamlit app is an earlier and lighter version of this notebook.  
  The full automated retrainable model is **more powerful**, but **takes longer to execute** and was not suitable for real-time web deployment.  
  The lighter version was used in the app to ensure faster prediction for demo purposes.

---

## 📢 Notes

- These notebooks are for **reference and documentation** purposes only.
- You do **not** need to run them to use the Streamlit app.
- The models are already integrated and automated inside the app.

---
