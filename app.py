import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config for a professional look
st.set_page_config(
    page_title="Reliance Stock Predictor | AI Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Deep Blue/Slate Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        color: #00d1b2;
    }
    .stMetric {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .forecast-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the Keras LSTM model and MinMaxScaler."""
    try:
        model = tf.keras.models.load_model('models/stock_lstm_model.keras')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

def fetch_stock_data(ticker):
    """Fetch recent historical data for back-testing and prediction."""
    # Fetching 90 days to be safe and handle non-trading days.
    data = yf.download(ticker, period='90d')
    # Flatten multi-index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def main():
    st.title("📊 RELIANCE INDUSTRIES | Stock Forecasting System")
    st.markdown("### *Senior AI Research Interface*")
    st.divider()

    model, scaler = load_model_and_scaler()
    if not model or not scaler:
        st.error("Model assets not found in /models directory.")
        return

    ticker = "RELIANCE.NS"

    with st.spinner("Executing real-time data ingestion..."):
        df = fetch_stock_data(ticker)

    if len(df) < 61:
        st.error(f"Insufficient market data ({len(df)} days). Need at least 61 trading days.")
        return

    # 1. LIVE BACK-TEST (Confidence-First Logic)
    # ----------------------------------------
    # Use the first 60 days of the last 61 days to predict today's close
    backtest_segment = df.iloc[-61:]
    actual_today = backtest_segment['Close'].iloc[-1]

    # Input for today's prediction (the 60 days leading up to today)
    input_today = backtest_segment.iloc[:-1]['Close'].values.reshape(-1, 1)
    input_today_scaled = scaler.transform(input_today)
    input_today_reshaped = np.reshape(input_today_scaled, (1, 60, 1))

    # Predict Today
    pred_today_scaled = model.predict(input_today_reshaped, verbose=0)
    pred_today = scaler.inverse_transform(pred_today_scaled)[0][0]

    # Live Accuracy Score Calculation
    error_percent = abs(actual_today - pred_today) / actual_today
    accuracy_score = (1 - error_percent) * 100

    # Display Metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Market Close (Today)", f"₹{actual_today:.2f}")

    with m2:
        st.metric("Live Accuracy Score", f"{accuracy_score:.2f}%")

    # Logic Gate for Forecast
    # -----------------------
    if 75 <= accuracy_score <= 85:
        with m3:
            st.success("CONFIDENCE: OPTIMAL")

        # 2. GENERATE FORECAST FOR TOMORROW
        # ---------------------------------
        # Use the absolute last 60 days to predict tomorrow
        input_tomorrow = df.iloc[-60:]['Close'].values.reshape(-1, 1)
        input_tomorrow_scaled = scaler.transform(input_tomorrow)
        input_tomorrow_reshaped = np.reshape(input_tomorrow_scaled, (1, 60, 1))

        pred_tomorrow_scaled = model.predict(input_tomorrow_reshaped, verbose=0)
        pred_tomorrow = scaler.inverse_transform(pred_tomorrow_scaled)[0][0]

        # Display Forecast Result
        st.markdown(f"""
            <div class="forecast-box">
                <h2 style='margin-top:0; color:#3b82f6;'>🚀 Next-Day Forecast</h2>
                <p style='font-size:1.5rem;'>Predicted Closing Price: <b>₹{pred_tomorrow:.2f}</b></p>
                <p style='color:#94a3b8;'><i>Validation Logic: Confidence score within operational threshold (75-85%).</i></p>
            </div>
        """, unsafe_allow_html=True)

        # Visuals: Last 7 Days vs Prediction
        st.subheader("📈 Interaction Analysis")
        last_7_days = df.iloc[-7:].copy()

        # Calculate tomorrow's date (skip weekends if necessary)
        last_date = last_7_days.index[-1]
        tomorrow_date = last_date + timedelta(days=1)
        if tomorrow_date.weekday() >= 5: # 5=Sat, 6=Sun
            tomorrow_date += timedelta(days=(7-tomorrow_date.weekday()))

        fig = go.Figure()

        # Historical Trace
        fig.add_trace(go.Scatter(
            x=last_7_days.index,
            y=last_7_days['Close'],
            name='Actual Market Price',
            line=dict(color='#00d1b2', width=4),
            mode='lines+markers'
        ))

        # Forecast Trace
        fig.add_trace(go.Scatter(
            x=[last_7_days.index[-1], tomorrow_date],
            y=[actual_today, pred_tomorrow],
            name='Forecast Projection',
            line=dict(color='#3b82f6', width=4, dash='dash'),
            mode='lines+markers'
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#30363d'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        with m3:
            st.warning("CONFIDENCE: LOW")

        st.error(f"**Forecasting Inhibited.** Current Live Accuracy ({accuracy_score:.2f}%) is outside the high-reliability window (75%-85%).")
        st.info("The system requires a specific calibration range to ensure data integrity. When accuracy deviates from the 75-85% band, the model is either under-fitting current volatility or showing signs of over-fitting historical noise.")

    # Mathematical Deep-Dive
    # ----------------------
    st.divider()
    with st.expander("🔬 Mathematical Deep-Dive: LSTM Network Architecture"):
        st.markdown(r"""
        ### Long Short-Term Memory (LSTM) Logic
        LSTMs are a specialized type of Recurrent Neural Network designed to learn long-term dependencies in sequential data. They are particularly effective for stock market analysis where the time-lag between significant events can be large.

        #### 1. The Core Innovation: The Cell State ($C_t$)
        Standard RNNs suffer from the **Vanishing Gradient Problem**—as sequences get longer, gradients become infinitely small, making the network "forget" earlier information. LSTMs solve this by introducing a **Cell State**, which acts as a memory belt that carries information across time steps with only minor linear interactions.

        #### 2. The Gating Mechanism
        Information flow is regulated by three distinct 'Gates':

        *   **Forget Gate ($f_t$):** Decides what information from the previous state is irrelevant and should be discarded.
            $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
        *   **Input Gate ($i_t$):** Determines which new information will be stored in the cell state. A `tanh` layer creates a vector of new candidate values ($\tilde{C}_t$), while a `sigmoid` layer decides which of those values to update.
            $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
        *   **Output Gate ($o_t$):** Decides what the next hidden state ($h_t$) should be. It passes the current cell state through a `tanh` and multiplies it by the output of a `sigmoid` gate.
            $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
            $$h_t = o_t * \tanh(C_t)$$

        #### 3. Preventing Vanishing Gradients
        Because the cell state update is primarily additive ($C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$), gradients can flow through the network over long time steps without being repeatedly multiplied by small weights, effectively preserving long-term dependencies in stock price trends.
        """)

    st.markdown("<p style='text-align: center; color: #4b5563; margin-top: 50px;'>Reliance Stock Prediction System | Project ML Foundation | Senior AI Research Engineer Edition</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
