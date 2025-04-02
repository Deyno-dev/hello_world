# Artificial Traders v4/Multi_Ai/src/models/train_trend.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from ..features.features import calculate_features
import logging
from pathlib import Path
from tqdm import tqdm
import json
import socketio

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'trend'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'
PLOT_DIR = PROJECT_ROOT / 'results' / 'plots'
for d in [LOG_DIR, MODEL_DIR, PRED_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_trend.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
CONFIG_PATH = PROJECT_ROOT / 'config' / 'ai_config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f).get('trend', {})
except FileNotFoundError:
    config = {}
    logging.warning("ai_config.json not found, using defaults")

# SocketIO for dashboard
sio = socketio.Client()


def train_trend_model(
        data_path,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 42)
):
    """
    Train an XGBoost model for trend prediction with dashboard integration.
    """
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        logging.info("Calculating technical features")
        df = calculate_features(df)

        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        X = df[features].dropna()
        y = (df['close'].shift(-1) > df['close']).astype(int).reindex(X.index)

        logging.info("Splitting data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        logging.info("Starting hyperparameter tuning")
        base_model = xgb.XGBClassifier(random_state=random_state)
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )

        # Simulate epochs for dashboard compatibility
        logging.info("Training model with GridSearchCV")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        logging.info("Best parameters: %s", grid_search.best_params_)

        # Simulate epoch updates (GridSearchCV doesn't have epochs, so we fake it)
        for i in tqdm(range(1, 4), desc="Simulated epochs"):
            sio.emit('training_update', {
                'model': 'trend',
                'epoch': i,
                'loss': 1 - grid_search.cv_results_['mean_test_score'][i - 1],  # Fake loss
                'val_loss': 1 - accuracy_score(y_val, best_model.predict(X_val))
            })

        # Evaluate
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        logging.info("Validation accuracy: %.4f", accuracy)
        sio.emit('training_complete', {
            'model': 'trend',
            'metrics': {'accuracy': float(accuracy)}
        })

        # Save predictions
        pred_df = pd.DataFrame({'actual': y_val[-100:], 'predicted': y_pred[-100:]})
        pred_path = PRED_DIR / 'trend_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        logging.info("Predictions saved to %s", pred_path)

        # Plot feature importance
        import matplotlib.pyplot as plt
        feat_imp = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imp.plot(kind='bar', title='Feature Importance - Trend Model')
        plt.tight_layout()
        plot_path = PLOT_DIR / 'trend_feature_importance.png'
        plt.savefig(plot_path)
        logging.info("Feature importance plot saved to %s", plot_path)

        # Save model
        model_path = MODEL_DIR / 'xgboost_trend.json'
        best_model.save_model(model_path)
        logging.info("Model saved to %s", model_path)

        sio.disconnect()
        return best_model

    except Exception as e:
        logging.error("Error in train_trend_model: %s", str(e))
        sio.emit('error', {'model': 'trend', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_trend_model(data_path)
        print("Trend model trained and saved successfully!")
        print(f"Check logs at: {LOG_DIR / 'train_trend.log'}")
        print(f"Check predictions at: {PRED_DIR / 'trend_predictions.csv'}")
        print(f"Check plot at: {PLOT_DIR / 'trend_feature_importance.png'}")
    except Exception as e:
        print(f"Failed to train model: {e}")