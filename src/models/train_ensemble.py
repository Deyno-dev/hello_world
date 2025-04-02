# Artificial Traders v4/Multi_Ai/src/models/train_ensemble.py
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..features.features import calculate_features
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib
import logging
from pathlib import Path
from tqdm import tqdm
import json
import socketio

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'ensemble'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'
for d in [LOG_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_ensemble.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
CONFIG_PATH = PROJECT_ROOT / 'config' / 'ai_config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f).get('ensemble', {})
except FileNotFoundError:
    config = {}
    logging.warning("ai_config.json not found, using defaults")

# SocketIO for dashboard
sio = socketio.Client()


def load_trained_models():
    trend_model = xgb.Booster()
    trend_model.load_model(PROJECT_ROOT / 'models' / 'trend' / 'xgboost_trend.json')
    volatility_model = load_model(PROJECT_ROOT / 'models' / 'volatility' / 'lstm_volatility.h5')
    regime_model = joblib.load(PROJECT_ROOT / 'models' / 'regime' / 'rf_regime.pkl')
    execution_model = DQN.load(PROJECT_ROOT / 'models' / 'execution' / 'dqn_execution.zip')
    return trend_model, volatility_model, regime_model, execution_model


def train_ensemble_model(
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

        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        X = df[features].dropna()
        y = (df['close'].shift(-1) > df['close']).astype(int).reindex(X.index)  # Trend target

        logging.info("Splitting data")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        # Load pre-trained models
        trend_model, volatility_model, regime_model, execution_model = load_trained_models()

        # Generate predictions from each model
        logging.info("Generating base model predictions")
        trend_pred = trend_model.predict(xgb.DMatrix(X_train))
        vol_pred = volatility_model.predict(X_train.values.reshape(-1, 1, len(features))[:, -20:, :])[:, 0]
        regime_pred = regime_model.predict(X_train)
        # Execution DQN predictions (simplified)
        env = DummyVecEnv([lambda: TradingEnv(df[features].dropna())])
        obs = env.reset()
        exec_pred = []
        for i in range(len(X_train)):
            action, _ = execution_model.predict(obs)
            obs, _, _, _ = env.step(action)
            exec_pred.append(action[0] % 2)  # Map to 0/1

        # Stack predictions as features
        X_stack = np.column_stack((trend_pred, vol_pred, regime_pred, exec_pred[:len(trend_pred)]))
        X_val_stack = np.column_stack((
            trend_model.predict(xgb.DMatrix(X_val)),
            volatility_model.predict(X_val.values.reshape(-1, 1, len(features))[:, -20:, :])[:, 0],
            regime_model.predict(X_val),
            exec_pred[len(trend_pred):len(trend_pred) + len(X_val)]
        ))

        # Train ensemble
        logging.info("Training ensemble model")
        ensemble = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state))
        ], voting='soft')
        ensemble.fit(X_stack, y_train)

        # Simulate epochs
        for epoch in tqdm(range(1, 4), desc="Simulated epochs"):
            train_pred = ensemble.predict_proba(X_stack)[:, 1][:len(X_stack) // 3 * epoch]
            val_pred = ensemble.predict_proba(X_val_stack)[:, 1][:len(X_val_stack) // 3 * epoch]
            train_loss = 1 - accuracy_score(y_train[:len(train_pred)], (train_pred > 0.5).astype(int))
            val_loss = 1 - accuracy_score(y_val[:len(val_pred)], (val_pred > 0.5).astype(int))
            sio.emit('training_update', {
                'model': 'ensemble',
                'epoch': epoch,
                'loss': float(train_loss),
                'val_loss': float(val_loss)
            })

        # Evaluate
        y_pred = ensemble.predict(X_val_stack)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info("Validation accuracy: %.4f", accuracy)
        sio.emit('training_complete', {
            'model': 'ensemble',
            'metrics': {'accuracy': float(accuracy)}
        })

        # Save predictions
        pred_df = pd.DataFrame({'actual': y_val[-100:], 'predicted': y_pred[-100:]})
        pred_path = PRED_DIR / 'ensemble_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        logging.info("Predictions saved to %s", pred_path)

        # Save model
        model_path = MODEL_DIR / 'ensemble.pkl'
        joblib.dump(ensemble, model_path)
        logging.info("Model saved to %s", model_path)

        sio.disconnect()
        return ensemble

    except Exception as e:
        logging.error("Error in train_ensemble_model: %s", str(e))
        sio.emit('error', {'model': 'ensemble', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_ensemble_model(data_path)
        print("Ensemble model trained and saved successfully!")
    except Exception as e:
        print(f"Failed to train model: {e}")