# Artificial Traders v4/Multi_Ai/src/models/train_volatility.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from ..features.features import calculate_features
import logging
from pathlib import Path

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'volatility'
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_volatility.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_volatility_model(data_path, seq_length=20, epochs=50, batch_size=32, patience=5):
    """
    Train an LSTM model to forecast volatility with chunked loading and early stopping.

    Args:
        data_path (Path): Path to OHLCV CSV file.
        seq_length (int): Lookback period for sequences.
        epochs (int): Max training epochs.
        batch_size (int): Batch size for training.
        patience (int): Early stopping patience.

    Returns:
        keras.Model: Trained LSTM model.
    """
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Load data in chunks to handle large files
        logging.info("Loading data from %s in chunks", data_path)
        chunks = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp', chunksize=100000)
        df_full = pd.concat([calculate_features(chunk) for chunk in chunks])

        logging.info("Calculating volatility")
        volatility = df_full['volatility'].dropna().values.reshape(-1, 1)

        # Normalize
        scaler = MinMaxScaler()
        volatility_scaled = scaler.fit_transform(volatility)

        # Prepare sequences
        logging.info("Preparing sequences with length %d", seq_length)
        X, y = prepare_sequences(volatility_scaled, seq_length)

        # Train-validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Build LSTM model
        logging.info("Building LSTM model")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train
        logging.info("Training model for up to %d epochs", epochs)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        logging.info("Final validation loss: %.6f", history.history['val_loss'][-1])

        # Save model and scaler
        model_path = MODEL_DIR / 'lstm_volatility.h5'
        scaler_path = MODEL_DIR / 'scaler.pkl'
        model.save(model_path)
        pd.to_pickle(scaler, scaler_path)
        logging.info("Model saved to %s", model_path)
        logging.info("Scaler saved to %s", scaler_path)

        return model

    except Exception as e:
        logging.error("Error in train_volatility_model: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_volatility_model(data_path)
        print("Volatility model trained and saved successfully!")
        print(f"Check logs at: {LOG_DIR / 'train_volatility.log'}")
    except Exception as e:
        print(f"Failed to train model: {e}")