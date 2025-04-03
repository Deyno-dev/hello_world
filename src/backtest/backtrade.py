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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
BACKTEST_DIR = PROJECT_ROOT / 'results' / 'backtest'
PLOT_DIR = PROJECT_ROOT / 'results' / 'plots'
for d in [LOG_DIR, BACKTEST_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG_DIR / 'backtrade.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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


def calculate_metrics(trades_df, equity):
    returns = pd.Series([t['balance'] / equity[i - 1] - 1 for i, t in trades_df.iterrows() if i > 0])
    ann_return = ((1 + returns.mean()) ** 252 - 1) * 100 if returns.mean() else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min() * 100
    wins = trades_df[trades_df['profit'] > 0]['profit'].sum()
    losses = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
    win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0
    profit_factor = wins / losses if losses > 0 else float('inf')
    return ann_return, sharpe, max_dd, win_rate, profit_factor


def backtrade(data_path, seq_length=20, initial_balance=10000, fee=0.001, sl=0.02, tp=0.05):
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df = calculate_features(df)
        features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend', 'volatility', 'adx', 'bb_upper', 'bb_lower',
                    'cci']
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
            balance = initial_balance
            position = 0
            entry_price = 0
            trades = []
            equity = [balance]

            for i in tqdm(range(len(df)), desc=f"Simulating {model_name} trades"):
                row = df.iloc[i]
                price = row['close']

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
                    else:
                    trend_pred = trend_model.predict(xgb.DMatrix([row[features]]))[0]
                    vol_pred = volatility_model.predict(X_vol[i].reshape(1, seq_length, len(features)), verbose=0)[0, 0]
                    regime_pred = regime_model.predict([row[features]])[0]
                    exec_pred = execution_model.predict(obs)[0] % 2
                    stack = np.array([[trend_pred, vol_pred, regime_pred, exec_pred]])
                    pred = model.predict(stack)[0]

                    if position > 0:
                        if
                    price <= entry_price * (1 - sl) or price >= entry_price * (1 + tp):
                    profit = position * (price - entry_price) - fee * position * price
                    balance += position * price * (1 - fee)
                    trades.append(
                        {'time': df.index[i], 'action': 'sell', 'price': price, 'balance': balance, 'profit': profit})
                    position = 0

                    if pred == 1 and balance >= price and position == 0:
                        position = balance * 0.1 / price
                    balance -= position * price * (1 + fee)
                    entry_price = price
                    trades.append(
                        {'time': df.index[i], 'action': 'buy', 'price': price, 'balance': balance, 'profit': 0})

                    equity.append(balance + position * price)

                    if i % 1000 == 0 and i > 0:
                        sio.emit('training_update', {
                            'model': model_name,
                            'epoch': i // 1000 + 1,
                            'loss': float(np.mean(equity[-1000:])),
                            'val_loss': float(np.std(equity[-1000:]) if len(equity) > 1000 else 0)
                        })

                    if position > 0:
                        profit = position * (df['close'].iloc[-1] - entry_price) - fee * position * df['close'].iloc[-1]
                    balance += position * df['close'].iloc[-1] * (1 - fee)
                    trades.append(
                        {'time': df.index[-1], 'action': 'sell', 'price': df['close'].iloc[-1], 'balance': balance,
                         'profit': profit})

                    trades_df = pd.DataFrame(trades)
                    ann_return, sharpe, max_dd, win_rate, profit_factor = calculate_metrics(trades_df, equity)
                    final_balance = balance
                    results[model_name] = {
                'final_balance': final_balance,
                'annualized_return': ann_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'profit_factor': profit_factor
            }
            logging.info(
                "%s - Final Balance: %.2f, Ann Return: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%, Win Rate: %.2f, Profit Factor: %.2f",
            model_name, final_balance, ann_return, sharpe, max_dd, win_rate, profit_factor)
            sio.emit('training_complete', {
                'model': model_name,
                'metrics': {
                    'final_balance': float(final_balance),
                    'annualized_return': float(ann_return),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(max_dd),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor)
                }
            })

            trades_df.to_csv(BACKTEST_DIR / f'{model_name}_trades.csv', index=False)
            plt.figure(figsize=(12, 6))
            plt.plot(df.index[1:], equity[1:], label=f'{model_name} Equity')
            plt.title(f'{model_name} Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.savefig(PLOT_DIR / f'{model_name}_equity_curve.png')

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