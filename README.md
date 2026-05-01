# Reliance Stock Price Prediction System 📊

A high-end AI research application built for the Machine Learning Foundation project, specializing in forecasting **Reliance Industries** stock prices using Long Short-Term Memory (LSTM) neural networks.

## 🚀 The "Confidence-First" Architecture

This system prioritizes data integrity and predictive reliability over raw forecasting. Unlike traditional models, this app implements a **Live Back-test** logic gate:

1.  **Real-Time Ingestion:** Fetches the latest market data via `yfinance`.
2.  **Live Validation:** Performs a back-test on the current day's closing price.
3.  **Accuracy Threshold:**
    -   If the Live Accuracy Score is between **75% and 85%**, the system generates the forecast for the next trading day.
    -   If accuracy falls outside this "High Reliability" window, the system inhibits the forecast to prevent acting on low-confidence data (potential over-fitting or extreme volatility).

## 🛠 Technical Stack

-   **Deep Learning:** Keras Sequential Model with LSTM Layers.
-   **Data Processing:** Pandas, NumPy, Scikit-learn (MinMaxScaler).
-   **Frontend/UX:** Streamlit with a professional slate/deep-blue aesthetic.
-   **Visualization:** Plotly Interactive Charts.

## 🔬 Mathematical Foundation

The core of this system is the **LSTM (Long Short-Term Memory)** network. LSTMs are engineered to solve the *Vanishing Gradient* problem in standard Recurrent Neural Networks (RNNs) by using a gating mechanism (Forget, Input, and Output gates). This allows the model to maintain a long-term memory of stock price dependencies over a 60-day lookback window.

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📁 Project Structure

-   `app.py`: The main Streamlit application script.
-   `models/`: Contains the pre-trained `stock_lstm_model.keras` and `scaler.pkl`.
-   `requirements.txt`: List of required Python packages.
-   `README.md`: Professional project documentation.

---
**Disclaimer:** *This tool is for educational purposes as part of a Machine Learning Foundation project. Stock market investments carry inherent risks.*
