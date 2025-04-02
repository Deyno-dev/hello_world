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
from ..models.train_execution import TradingEnv
import logging
from pathlib import Path
import socketio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
BACKTEST_DIR = PROJECT_ROOT / 'results' / 'backtest'
PLOT_DIR = PROJECT_ROOT / 'results' / 'plots'
for d in [LOG_DIR, BACKTEST_DIR, PLOT_DIR]:
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
    trend_model = xgb.Booster()
    trend_model.load_model(PROJECT_ROOT / 'models' / 'trend' / 'xgboost_trend.json')
    volatility_model = load_model(PROJECT_ROOT / 'models' / 'volatility' / 'lstm_volatility.h5')
    regime_model = joblib.load(PROJECT_ROOT / 'models' / 'regime' / 'rf_regime.pkl')
    execution_model = DQN.load(PROJECT_ROOT / 'models' / 'execution' / 'dqn_execution.zip')
    ensemble_model = joblib.load(PROJECT_ROOT / 'models' / 'ensemble' / 'ensemble.pkl')
    scaler = pd.read_pickle(PROJECT_ROOT / 'models' / 'volatility' / 'scaler.pkl')
    return trend_model, volatility_model, regime_model, execution_model, ensemble_model, scaler


def calculate_metrics(returns):
    annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100  # Adjust for 1-min data if needed
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min() * 100
    return annualized_return, sharpe_ratio, max_drawdown


def backtrade(data_path, seq_length=20, initial_balance=10000, fee=0.001, sl=0.02, tp=0.05):
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df = calculate_features(df)

        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend',
                    'volatility', 'adx', 'bb_upper', 'bb_lower', 'cci']
        df = df[features + ['close']].dropna()

        trend_model, volatility_model, regime_model, execution_model, ensemble_model, scaler = load_models()

        vol_data = scaler.transform(df[features])
        X_vol = np.array([vol_data[i - seq_length:i] for i in range(seq_length, len(vol_data))])
        df = df.iloc[seq_length:]

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
            balance = initial_balance
            position = 0
            entry_price = 0
            trades = []
            equity = [balance]
            returns = []

            for i in tqdm(range(len(df)), desc=f"Simulating {model_name} trades"):
                row = df.iloc[i]
                price = row['close']

                # Generate prediction
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
                    pred = action[0] % 2
                    else:  # ensemble
                    trend_pred = trend_model.predict(xgb.DMatrix([row[features]]))[0]
                    vol_pred = volatility_model.predict(X_vol[i].reshape(1, seq_length, len(features)), verbose=0)[0, 0]
                    regime_pred = regime_model.predict([row[features]])[0]
                    exec_pred = execution_model.predict(obs)[0] % 2
                    stack = np.array([[trend_pred, vol_pred, regime_pred, exec_pred]])
                    pred = model.predict(stack)[0]

                    # Trading logic with SL/TP
                    if position > 0:
                        if
                    price <= entry_price * (1 - sl) or price >= entry_price * (1 + tp):
                    balance += position * price * (1 - fee)
                    trades.append({'time': df.index[i], 'action': 'sell', 'price': price, 'balance': balance})
                    returns.append((price - entry_price) / entry_price - fee)
                    position = 0

                    if pred == 1 and balance >= price and position == 0:
                        position = balance * 0.1 / price  # 10% position sizing
                    balance -= position * price * (1 + fee)
                    entry_price = price
                    trades.append({'time': df.index[i], 'action': 'buy', 'price': price, 'balance': balance})

                    equity.append(balance + position * price)

                    # Dashboard update every 1000 steps
                    if i % 1000 == 0 and i > 0:
                        sio.emit('training_update', {
                            'model': model_name,
                            'epoch': i // 1000 + 1,
                            'loss': float(np.mean(equity[-1000:])),  # Equity as "loss"
                            'val_loss': float(np.std(returns[-1000:]) if returns else 0)  # Volatility as "val_loss"
                        })

                    # Finalize
                    if position > 0:
                        balance += position * df['close'].iloc[-1] * (1 - fee)
                    trades.append(
                        {'time': df.index[-1], 'action': 'sell', 'price': df['close'].iloc[-1], 'balance': balance})
                    position = 0

                    # Metrics
                    returns = pd.Series(returns)
                    ann_return, sharpe, max_dd = calculate_metrics(returns)
                    final_balance = balance
                    results[model_name] = {
                'final_balance': final_balance,
                'annualized_return': ann_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            }
            logging.info("%s - Final Balance: %.2f, Ann Return: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
            model_name, final_balance, ann_return, sharpe, max_dd)
            sio.emit('training_complete', {
                'model': model_name,
                'metrics': {
                    'final_balance': float(final_balance),
                    'annualized_return': float(ann_return),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(max_dd)
                }
            })

            # Save trades and equity curve
            pd.DataFrame(trades).to_csv(BACKTEST_DIR / f'{model_name}_trades.csv', index=False)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[1:], equity[1:], label=f'{model_name} Equity')
            plt.title(f'{model_name} Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.savefig(PLOT_DIR / f'{model_name}_equity_curve.png')
            logging.info("Trades saved to %s, Equity curve to %s",
            BACKTEST_DIR / f'{model_name}_trades.csv',
            PLOT_DIR / f'{model_name}_equity_curve.png')

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