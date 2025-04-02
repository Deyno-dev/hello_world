# Artificial Traders v4/Multi_Ai/src/models/train_regime.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..features.features import calculate_features
import logging
from pathlib import Path
from tqdm import tqdm
import json
import socketio
import matplotlib.pyplot as plt

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'regime'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'
PLOT_DIR = PROJECT_ROOT / 'results' / 'plots'
for d in [LOG_DIR, MODEL_DIR, PRED_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_regime.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
CONFIG_PATH = PROJECT_ROOT / 'config' / 'ai_config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f).get('regime', {})
except FileNotFoundError:
    config = {}
    logging.warning("ai_config.json not found, using defaults")

# SocketIO for dashboard
sio = socketio.Client()


def train_regime_model(
        data_path,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 42)
):
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        logging.info("Calculating features")
        df = calculate_features(df)

        # Define regime: 1 for trending (high ADX), 0 for ranging (low ADX)
        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        X = df[features].dropna()
        y = (df['adx'] > 25).astype(int).reindex(X.index)  # ADX > 25 = trending

        logging.info("Splitting data")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        logging.info("Training Random Forest")
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)

        # Simulate epochs for dashboard
        for epoch in tqdm(range(1, 4), desc="Simulated epochs"):
            train_pred = model.predict(X_train[:len(X_train) // 3 * epoch])
            val_pred = model.predict(X_val[:len(X_val) // 3 * epoch])
            train_loss = 1 - accuracy_score(y_train[:len(train_pred)], train_pred)
            val_loss = 1 - accuracy_score(y_val[:len(val_pred)], val_pred)
            sio.emit('training_update', {
                'model': 'regime',
                'epoch': epoch,
                'loss': float(train_loss),
                'val_loss': float(val_loss)
            })

        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info("Validation accuracy: %.4f", accuracy)
        sio.emit('training_complete', {
            'model': 'regime',
            'metrics': {'accuracy': float(accuracy)}
        })

        # Save predictions
        pred_df = pd.DataFrame({'actual': y_val[-100:], 'predicted': y_pred[-100:]})
        pred_path = PRED_DIR / 'regime_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        logging.info("Predictions saved to %s", pred_path)

        # Plot feature importance
        feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imp.plot(kind='bar', title='Feature Importance - Regime Model')
        plt.tight_layout()
        plot_path = PLOT_DIR / 'regime_feature_importance.png'
        plt.savefig(plot_path)
        logging.info("Plot saved to %s", plot_path)

        # Save model
        import joblib
        model_path = MODEL_DIR / 'rf_regime.pkl'
        joblib.dump(model, model_path)
        logging.info("Model saved to %s", model_path)

        sio.disconnect()
        return model

    except Exception as e:
        logging.error("Error in train_regime_model: %s", str(e))
        sio.emit('error', {'model': 'regime', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_regime_model(data_path)
        print("Regime model trained and saved successfully!")
    except Exception as e:
        print(f"Failed to train model: {e}")