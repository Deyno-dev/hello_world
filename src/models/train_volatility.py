# Artificial Traders v4/Multi_Ai/src/models/train_volatility.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ..features.features import calculate_features
import logging
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import socketio
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'volatility'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'
PLOT_DIR = PROJECT_ROOT / 'results' / 'plots'
for d in [LOG_DIR, MODEL_DIR, PRED_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG_DIR / 'train_volatility.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_PATH = PROJECT_ROOT / 'config' / 'ai_config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f).get('volatility', {})
except FileNotFoundError:
    config = {}
    logging.warning("ai_config.json not found, using defaults")

sio = socketio.Client()


def prepare_sequences(data, seq_length, features):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([inputs, attn_output]))
    ff_output = Dense(ff_dim, activation="relu")(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout)(ff_output)
    return LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))


def build_transformer(seq_length, num_features, units, dropout, num_heads):
    inputs = Input(shape=(seq_length, num_features))
    x = transformer_block(inputs, head_size=units, num_heads=num_heads, ff_dim=units * 4, dropout=dropout)
    x = transformer_block(x, head_size=units, num_heads=num_heads, ff_dim=units * 4, dropout=dropout)
    x = Dense(1)(x[:, -1, :])  # Predict last timestep
    model = Model(inputs, x)
    return model


def objective(trial, data_path, seq_length, X_train, X_val, y_train, y_val):
    units = trial.suggest_int('units', 32, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    model = build_transformer(seq_length, X_train.shape[2], units, dropout, num_heads)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )
    return min(history.history['val_loss'])


def train_volatility_model(data_path, seq_length=config.get('seq_length', 20), n_trials=10):
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        logging.info("Loading data from %s in chunks", data_path)
        chunks = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp', chunksize=100000)
        df_full = pd.concat([calculate_features(chunk) for chunk in tqdm(chunks, desc="Processing chunks")])

        features = ['volatility', 'rsi', 'macd', 'adx']
        data = df_full[features].dropna().values

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        logging.info("Preparing sequences with length %d", seq_length)
        X, y = prepare_sequences(data_scaled, seq_length, features)

        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        logging.info("Optimizing hyperparameters with Optuna")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, data_path, seq_length, X_train, X_val, y_train, y_val),
                       n_trials=n_trials)

        best_params = study.best_params
        logging.info("Best parameters: %s", best_params)

        model = build_transformer(seq_length, len(features), best_params['units'], best_params['dropout'],
                                  best_params['num_heads'])
        model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        for epoch in tqdm(range(len(history.history['loss'])), desc="Training epochs"):
            sio.emit('training_update', {
                'model': 'volatility',
                'epoch': epoch + 1,
                'loss': float(history.history['loss'][epoch]),
                'val_loss': float(history.history['val_loss'][epoch])
            })

        y_pred_scaled = model.predict(X_val, verbose=0)
        y_val_unscaled = scaler.inverse_transform(
            np.hstack([y_val.reshape(-1, 1), np.zeros((len(y_val), len(features) - 1))))[:, 0]
        y_pred_unscaled = scaler.inverse_transform(
            np.hstack([y_pred_scaled, np.zeros((len(y_pred_scaled), len(features) - 1))))[:, 0]
        rmse = np.sqrt(mean_squared_error(y_val_unscaled, y_pred_unscaled))
        logging.info("Validation RMSE: %.6f", rmse)
        sio.emit('training_complete', {'model': 'volatility', 'metrics': {'rmse': float(rmse)}})

        pred_df = pd.DataFrame({'actual': y_val_unscaled[-100:], 'predicted': y_pred_unscaled[-100:]})
        pred_path = PRED_DIR / 'volatility_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        plt.figure(figsize=(10, 6))
        plt.plot(pred_df['actual'], label='Actual Volatility')
        plt.plot(pred_df['predicted'], label='Predicted Volatility')
        plt.title('Volatility Forecast Sample')
        plt.legend()
        plot_path = PLOT_DIR / 'volatility_plot.png'
        plt.savefig(plot_path)

        model_path = MODEL_DIR / 'transformer_volatility.h5'
        scaler_path = MODEL_DIR / 'scaler.pkl'
        model.save(model_path)
        pd.to_pickle(scaler, scaler_path)

        sio.disconnect()
        return model

    except Exception as e:
        logging.error("Error in train_volatility_model: %s", str(e))
        sio.emit('error', {'model': 'volatility', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_volatility_model(data_path)
        print("Volatility Transformer model trained and saved successfully!")
    except Exception as e:
        print(f"Failed to train model: {e}")