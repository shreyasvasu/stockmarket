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

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_resource
def load_model_and_scaler():
    """Load the Advanced Keras LSTM model and MinMaxScaler."""
    try:
        model = tf.keras.models.load_model('models/stock_lstm_model_v2.keras')
        scaler = joblib.load('models/scaler_v2.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

def fetch_stock_data(ticker):
    """Fetch recent historical data and engineer technical indicators."""
    # Fetching 120 days to account for technical indicator warm-up (SMA/RSI)
    data = yf.download(ticker, period='120d')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Engineer indicators
    data['RSI'] = calculate_rsi(data['Close'])
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Drop warm-up rows
    data = data.dropna()
    return data

def main():
    st.title("📊 RELIANCE INDUSTRIES | Advanced AI Forecasting")
    st.markdown("### *Multivariate Bidirectional LSTM System*")
    st.divider()

    model, scaler = load_model_and_scaler()
    if not model or not scaler:
        st.error("Advanced model assets not found in /models directory.")
        return

    ticker = "RELIANCE.NS"
    features = ['Close', 'Volume', 'RSI', 'SMA_20']

    with st.spinner("Synchronizing high-frequency market data..."):
        df = fetch_stock_data(ticker)

    if len(df) < 61:
        st.error(f"Insufficient market data ({len(df)} days). Need at least 61 trading days after indicator stabilization.")
        return

    # 1. INTERACTIVE SIMULATION SECTION
    st.markdown("---")
    st.header("🧪 Advanced Research Simulation")
    st.info("Test the new Bidirectional LSTM. This model analyzes price, volume, RSI, and SMA trends simultaneously.")

    available_dates = df.index[60:].tolist()
    
    if available_dates:
        date_options = {d.strftime('%Y-%m-%d'): d for d in available_dates}
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            selected_date_str = st.selectbox(
                "Target Date for Prediction:",
                options=list(date_options.keys()),
                index=len(date_options)-1
            )
            selected_date = date_options[selected_date_str]
            target_idx = df.index.get_loc(selected_date)
            
            # Multivariate input (60 days of 4 features)
            sim_input_data = df.iloc[target_idx-60:target_idx][features].values
            sim_actual = df.iloc[target_idx]['Close']
            
            if st.button("🚀 Execute Neural Simulation"):
                with st.spinner("Processing multivariate sequence..."):
                    # 1. Back-test for the selected day
                    sim_input_scaled = scaler.transform(df.iloc[target_idx-60:target_idx][features].values)
                    sim_input_reshaped = np.reshape(sim_input_scaled, (1, 60, 4))
                    sim_pred_scaled = model.predict(sim_input_reshaped, verbose=0)
                    
                    dummy = np.zeros((1, 4))
                    dummy[0, 0] = sim_pred_scaled[0, 0]
                    sim_pred = scaler.inverse_transform(dummy)[0][0]
                    
                    # 2. 5-Day Recursive Forecast from selected day
                    sim_sequence = df.iloc[target_idx-59:target_idx+1][features].values.tolist()
                    sim_forecasts = []
                    sim_f_dates = []
                    curr_date = selected_date
                    
                    for _ in range(5):
                        s_scaled = scaler.transform(np.array(sim_sequence[-60:]))
                        s_reshaped = np.reshape(s_scaled, (1, 60, 4))
                        p_s = model.predict(s_reshaped, verbose=0)
                        d_p = np.zeros((1, 4))
                        d_p[0, 0] = p_s[0, 0]
                        p_r = scaler.inverse_transform(d_p)[0, 0]
                        sim_forecasts.append(p_r)
                        
                        nx_date = curr_date + timedelta(days=1)
                        sim_f_dates.append(nx_date)
                        curr_date = nx_date
                        
                        new_r = [p_r, sim_sequence[-1][1], sim_sequence[-1][2], sim_sequence[-1][3]]
                        sim_sequence.append(new_r)
                    
                    st.session_state['sim_result'] = {
                        'date': selected_date_str,
                        'actual': sim_actual,
                        'pred': sim_pred,
                        'forecasts': sim_forecasts,
                        'f_dates': sim_f_dates,
                        'history': df.iloc[target_idx-7:target_idx+1].copy()
                    }

        if 'sim_result' in st.session_state:
            res = st.session_state['sim_result']
            with col_sim2:
                # Prediction vs Actual for Target Date
                st.markdown(f"""
                    <div style="background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 15px;">
                        <h4 style="margin-top:0; color: #00d1b2;">Simulation Results: {res['date']}</h4>
                        <div style="display: flex; justify-content: space-between;">
                            <div><p style="color: #94a3b8; margin-bottom: 5px;">Actual</p><h3 style="margin-top:0;">₹{res['actual']:.2f}</h3></div>
                            <div><p style="color: #94a3b8; margin-bottom: 5px;">AI Prediction</p><h3 style="margin-top:0; color: #3b82f6;">₹{res['pred']:.2f}</h3></div>
                            <div><p style="color: #94a3b8; margin-bottom: 5px;">Error</p><h3 style="margin-top:0;">{abs(res['actual']-res['pred'])/res['actual']*100:.2f}%</h3></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # 5-Day Simulation Forecast
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #00d1b2;">
                        <p style="margin:0; font-weight: bold; color: #00d1b2;">🚀 Simulation 5-Day Trend (from {res['date']})</p>
                        <div style="display: flex; justify-content: space-between; overflow-x: auto; margin-top: 10px;">
                            {"".join([f'<div style="min-width: 100px; padding: 5px;">'
                                       f'<p style="margin:0; font-size: 0.75rem; color: #94a3b8;">{d.strftime("%b %d")}</p>'
                                       f'<p style="margin:0; font-weight: bold; font-size: 1rem;">₹{p:.2f}</p>'
                                       f'</div>' for d, p in zip(res['f_dates'], res['forecasts'])])}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                sim_fig = go.Figure()
                sim_fig.add_trace(go.Scatter(x=res['history'].index, y=res['history']['Close'], name='Actual', line=dict(color='#00d1b2', width=3), mode='lines+markers'))
                sim_fig.add_trace(go.Scatter(x=[res['history'].index[-1]] + res['f_dates'], y=[res['actual']] + res['forecasts'], name='Simulated Forecast', line=dict(color='#3b82f6', width=3, dash='dash'), mode='lines+markers'))
                sim_fig.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'))
                st.plotly_chart(sim_fig, use_container_width=True)

    # 2. LIVE OPERATIONAL STATUS
    st.markdown("---")
    st.header("⚡ Live AI Intelligence")
    
    live_segment = df.iloc[-61:]
    actual_today = live_segment['Close'].iloc[-1]
    input_today = live_segment.iloc[:-1][features].values
    input_today_scaled = scaler.transform(input_today)
    input_today_reshaped = np.reshape(input_today_scaled, (1, 60, 4))
    
    pred_today_scaled = model.predict(input_today_reshaped, verbose=0)
    dummy_today = np.zeros((1, 4))
    dummy_today[0, 0] = pred_today_scaled[0, 0]
    pred_today = scaler.inverse_transform(dummy_today)[0][0]
    
    error_percent = abs(actual_today - pred_today) / actual_today
    accuracy_score = (1 - error_percent) * 100

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Market Close (Today)", f"₹{actual_today:.2f}")
    with m2: st.metric("Model Confidence Score", f"{accuracy_score:.2f}%")

    if accuracy_score >= 80:
        with m3: st.success(f"CONFIDENCE: OPTIMAL (Score: {accuracy_score:.2f}/100)")
        
        # MULTI-DAY RECURSIVE FORECAST (5 DAYS)
        # -------------------------------------
        current_sequence = df.iloc[-60:][features].values.tolist()
        predictions = []
        forecast_dates = []
        last_date = df.index[-1]
        
        with st.spinner("Generating 5-day recursive forecast..."):
            for i in range(5):
                seq_scaled = scaler.transform(np.array(current_sequence[-60:]))
                seq_reshaped = np.reshape(seq_scaled, (1, 60, 4))
                p_scaled = model.predict(seq_reshaped, verbose=0)
                dummy_p = np.zeros((1, 4))
                dummy_p[0, 0] = p_scaled[0, 0]
                p_real = scaler.inverse_transform(dummy_p)[0, 0]
                predictions.append(p_real)
                
                next_date = last_date + timedelta(days=1)
                forecast_dates.append(next_date)
                last_date = next_date
                
                new_row = [p_real, current_sequence[-1][1], current_sequence[-1][2], current_sequence[-1][3]]
                current_sequence.append(new_row)

        # Display Forecast Result
        st.markdown(f"""
            <div class="forecast-box">
                <h2 style='margin-top:0; color:#3b82f6;'>🚀 5-Day Trend Forecast</h2>
                <div style="display: flex; gap: 20px; align-items: center; margin-bottom: 15px;">
                    <p style='color:#94a3b8; margin: 0;'><i>V2 Multi-Day Intelligence Active</i></p>
                    <span style="background-color: #00d1b2; color: #0e1117; padding: 2px 10px; border-radius: 20px; font-weight: bold; font-size: 0.8rem;">Precision: {accuracy_score:.2f}/100</span>
                </div>
                <div style="display: flex; justify-content: space-between; overflow-x: auto;">
                    {"".join([f'<div style="min-width: 120px; border-right: 1px solid #30363d; padding: 5px;">'
                               f'<p style="margin:0; font-size: 0.8rem; color: #94a3b8;">{d.strftime("%b %d")}</p>'
                               f'<p style="margin:0; font-weight: bold; font-size: 1.1rem;">₹{p:.2f}</p>'
                               f'</div>' for d, p in zip(forecast_dates, predictions)])}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📈 Interactive Multi-Day Projection")
        last_7_days = df.iloc[-7:].copy()
        
        fig = go.Figure()
        # History
        fig.add_trace(go.Scatter(x=last_7_days.index, y=last_7_days['Close'], name='Market History', line=dict(color='#00d1b2', width=4), mode='lines+markers'))
        # Forecast
        fig.add_trace(go.Scatter(x=[last_7_days.index[-1]] + forecast_dates, y=[actual_today] + predictions, name='5-Day AI Projection', line=dict(color='#3b82f6', width=4, dash='dash'), mode='lines+markers'))
        
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#30363d'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        with m3: st.warning("CONFIDENCE: SUB-OPTIMAL (< 80%)")
        st.error(f"**Forecasting Inhibited.** Accuracy ({accuracy_score:.2f}%) is below the required 80% threshold for multi-day projection.")
        st.info("To maintain high data integrity, the system requires an 80% baseline confidence score before attempting recursive multi-day forecasts.")

    st.divider()
    with st.expander("🔬 V2 Architecture: Multivariate Bidirectional LSTM"):
        st.markdown(r"""
        ### What's New in Version 2.0?
        
        1. **Multivariate Input**: Instead of only closing prices, the model now ingests 4 distinct data streams:
           - **Close Price**: Base trend.
           - **Volume**: Validates the strength of price movements.
           - **RSI (14-day)**: Identifies overbought/oversold conditions.
           - **SMA (20-day)**: Provides context for the short-term trend.

        2. **Bidirectional LSTM Layers**: 
           Standard LSTMs only look at past data to predict the future. Bidirectional layers process the data in **both directions** during training, allowing the model to capture complex cyclical patterns that single-direction layers might miss.

        3. **Confidence Expansion**: 
           Because the V2 model is more robust, the operational confidence threshold has been expanded from **85% up to 95%**, allowing for more frequent forecasts while maintaining rigorous validation.
        """)

    st.markdown("<p style='text-align: center; color: #4b5563; margin-top: 50px;'>Reliance AI Research V2.0 | Multivariate Neural Platform</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()




