import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_advanced_model():
    print("Starting Advanced Model Training for RELIANCE.NS...")
    
    # 1. Fetch Data
    ticker = "RELIANCE.NS"
    df = yf.download(ticker, period="5y")
    
    # Flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Feature Engineering
    print("Engineering features (RSI, SMA, Volume)...")
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN from indicators
    df = df.dropna()
    
    # Select features
    features = ['Close', 'Volume', 'RSI', 'SMA_20']
    data = df[features].values
    
    # 3. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 4. Create Sequences
    X, y = [], []
    lookback = 60
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0]) # Predicting the 'Close' price (index 0)
    
    X, y = np.array(X), np.array(y)
    
    # Train/Test Split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 5. Build Advanced Bidirectional LSTM Model
    print("Building Bidirectional LSTM Architecture...")
    model = Sequential([
        Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(lookback, len(features))),
        Dropout(0.2),
        Bidirectional(LSTM(units=100, return_sequences=False)),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 6. Callbacks for better training
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # 7. Train
    print("Training model (this may take a few minutes)...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 8. Save Artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    print("Saving advanced model and scaler...")
    model.save('models/stock_lstm_model_v2.keras')
    joblib.dump(scaler, 'models/scaler_v2.pkl')
    
    print("Training Complete!")
    print("Model saved to: models/stock_lstm_model_v2.keras")
    print("Scaler saved to: models/scaler_v2.pkl")

if __name__ == "__main__":
    train_advanced_model()
