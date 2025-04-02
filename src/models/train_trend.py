# Artificial Traders v4/Multi_Ai/src/models/train_trend.py
import pandas as pd
import xgboost as xgb
from ..features.features import calculate_features
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Up 2 levels from src/models/
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'trend'
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_trend.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_trend_model(data_path, test_size=0.2, random_state=42):
    """
    Train an XGBoost model to predict price direction with validation and tuning.

    Args:
        data_path (Path): Path to the CSV file with OHLCV data.
        test_size (float): Fraction of data for validation.
        random_state (int): Seed for reproducibility.

    Returns:
        xgb.XGBClassifier: Trained and tuned model.
    """
    try:
        # Check data file
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Load and preprocess data
        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        logging.info("Calculating technical features")
        df = calculate_features(df)

        # Define features and target
        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        X = df[features].dropna()
        y = (df['close'].shift(-1) > df['close']).astype(int).reindex(X.index)

        # Train-validation split
        logging.info("Splitting data into train and validation sets")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        # Define base model
        base_model = xgb.XGBClassifier(random_state=random_state)

        # Hyperparameter tuning
        logging.info("Starting hyperparameter tuning")
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100]
        }
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best model
        model = grid_search.best_estimator_
        logging.info("Best parameters: %s", grid_search.best_params_)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        logging.info("Validation accuracy: %.4f", val_accuracy)

        # Retrain on full dataset
        logging.info("Retraining on full dataset with best parameters")
        model.fit(X, y)

        # Save model
        model_path = MODEL_DIR / 'xgboost_trend.json'
        model.save_model(model_path)
        logging.info("Model saved to %s", model_path)

        return model

    except Exception as e:
        logging.error("Error in train_trend_model: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_trend_model(data_path)
        print("Trend model trained and saved successfully!")
        print(f"Check logs at: {LOG_DIR / 'train_trend.log'}")
    except Exception as e:
        print(f"Failed to train model: {e}")