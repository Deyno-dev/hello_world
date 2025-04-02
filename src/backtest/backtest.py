# Artificial Traders v4/Multi_Ai/src/backtest/backtrade.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from ..features.features import calculate_features
from ..models.train_execution import TradingEnv  # Reuse env from train_execution
import logging
from pathlib import Path
import socketio
from tqdm import tqdm

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
BACKTEST_DIR = PROJECT_ROOT / 'results' / 'backtest'
for d in [LOG_DIR, BACKTEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'backtrade.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# SocketIO for dashboard
sio = socketio.Client()


def load_models():
    """Load all trained models."""
    trend_model = xgb.Booster()
    trend_model.load_model(PROJECT_ROOT / 'models' / 'trend' / 'xgboost_trend.json')
    volatility_model = load_model(PROJECT_ROOT / 'models' / 'volatility' / 'lstm_volatility.h5')
    regime_model = joblib.load(PROJECT_ROOT / 'models' / 'regime' / 'rf_regime.pkl')
    execution_model = DQN.load(PROJECT_ROOT / 'models' / 'execution' / 'dqn_execution.zip')
    ensemble_model = joblib.load(PROJECT_ROOT / 'models' / 'ensemble' / 'ensemble.pkl')
    scaler = pd.read_pickle(PROJECT_ROOT / 'models' / 'volatility' / 'scaler.pkl')
    return trend_model, volatility_model, regime_model, execution_model, ensemble_model, scaler


def calculate_metrics(returns):
    """Calculate annualized return, Sharpe ratio, and max drawdown."""
    annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100  # Assuming daily data, adjust for 1-min
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() * 100
    return annualized_return, sharpe_ratio, max_drawdown


def backtrade(data_path, seq_length=20):
    """Backtest all models and ensemble."""
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        # Load data
        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df = calculate_features(df)

        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        df = df[features + ['close']].dropna()

        # Load models
        trend_model, volatility_model, regime_model, execution_model, ensemble_model, scaler = load_models()

        # Prepare data for volatility model
        vol_data = scaler.transform(df[features])
        X_vol = np.array([vol_data[i - seq_length:i] for i in range(seq_length, len(vol_data))])
        df = df.iloc[seq_length:]  # Align with volatility sequences

        # Backtest each model
        models = {
            'trend': trend_model,
            'volatility': volatility_model,
            'regime': regime_model,
            'execution': execution_model,
            'ensemble': ensemble_model
        }
        results = {}

        for model_name, model in tqdm(models.items(), desc="Backtesting models"):
            logging.info("Backtesting %s", model_name)
            balance = 10000
            position = 0
            trades = []
            returns = []

            for i in tqdm(range(len(df)), desc=f"Simulating {model_name} trades"):
                row = df.iloc[i]
                if model_name == 'trend':
                    pred = model.predict(xgb.DMatrix([row[features]]))[0]
                elif model_name == 'volatility':
                    pred = model.predict(X_vol[i].reshape(1, seq_length, len(features)), verbose=0)[0, 0]
                    pred = 1 if pred > scaler.inverse_transform([[0.5] + [0] * (len(features) - 1))])[0, 0] else 0
                    elif model_name == 'regime':
                    pred = model.predict([row[features]])[0]
                    elif model_name == 'execution':
                    env = DummyVecEnv([lambda: TradingEnv(df[features + ['close']])])
                    obs = env.reset()
                    env.envs[0].current_step = i
                    action, _ = model.predict(obs)
                    pred = action[0] % 2  # 0 or 1
                    else:  # ensemble
                    trend_pred = trend_model.predict(xgb.DMatrix([row[features]]))[0]
                    vol_pred = volatility_model.predict(X_vol[i].reshape(1, seq_length, len(features)), verbose=0)[0, 0]
                    regime_pred = regime_model.predict([row[features]])[0]
                    exec_pred = execution_model.predict(obs)[0] % 2
                    stack = np.array([[trend_pred, vol_pred, regime_pred, exec_pred]])
                    pred = model.predict(stack)[0]

                    price = row['close']
                    if pred == 1 and balance >= price:  # Buy
                        position += 1
                    balance -= price
                    trades.append({'time': df.index[i], 'action': 'buy', 'price': price, 'balance': balance})
                    elif pred == 0 and position > 0:  # Sell
                    position -= 1
                    balance += price
                    trades.append({'time': df.index[i], 'action': 'sell', 'price': price, 'balance': balance})

                    if i > 0:
                        daily_return = (balance + position * price - 10000) / 10000
                    returns.append(daily_return)

                    # Send update every 1000 steps
                    if i % 1000 == 0:
                        sio.emit('training_update', {
                            'model': model_name,
                            'epoch': i // 1000 + 1,
                            'loss': float(np.mean(returns[-1000:]) if returns else 0),  # Placeholder
                            'val_loss': float(np.std(returns[-1000:]) if returns else 0)
                        })

                    # Calculate metrics
                    returns = pd.Series(returns)
                    ann_return, sharpe, max_dd = calculate_metrics(returns)
                    results[model_name] = {
                'final_balance': balance + position * df['close'].iloc[-1],
                'annualized_return': ann_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            }
            logging.info("%s - Final Balance: %.2f, Ann Return: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
            model_name, results[model_name]['final_balance'], ann_return, sharpe, max_dd)
            sio.emit('training_complete', {
                'model': model_name,
                'metrics': {
                    'final_balance': float(results[model_name]['final_balance']),
                    'annualized_return': float(ann_return),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(max_dd)
                }
            })

            # Save trades
            pd.DataFrame(trades).to_csv(BACKTEST_DIR / f'{model_name}_trades.csv', index=False)
            logging.info("Trades saved to %s", BACKTEST_DIR / f'{model_name}_trades.csv')

            sio.disconnect()
    return results

except Exception as e:
logging.error("Error in backtrade: %s", str(e))
sio.emit('error', {'model': 'backtrade', 'message': str(e)})
sio.disconnect()
raise

if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        results = backtrade(data_path)
        print("Backtesting completed successfully!")
        for model, metrics in results.items():
            print(f"{model}: {metrics}")
    except Exception as e:
        print(f"Failed to backtest: {e}")