import ccxt
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
import time
import os

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
LIVE_DIR = PROJECT_ROOT / 'results' / 'livetrade'
for d in [LOG_DIR, LIVE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'livetrade.log',
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


def initialize_exchange(api_key, secret):
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv(exchange, symbol='BTC/USDT', timeframe='1m', limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return calculate_features(df)


def livetrade(api_key, secret, symbol='BTC/USDT', timeframe='1m', seq_length=20, initial_balance=1000, fee=0.001,
              sl=0.02, tp=0.05):
    try:
        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        exchange = initialize_exchange(api_key, secret)
        trend_model, volatility_model, regime_model, execution_model, ensemble_model, scaler = load_models()

        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []

        logging.info("Starting live trading loop for %s", symbol)
        while True:
            # Fetch live data
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=seq_length + 1)
            features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'supertrend', 'volatility', 'adx', 'bb_upper',
                        'bb_lower', 'cci']
            latest_data = df[features + ['close']].dropna()

            if len(latest_data) < seq_length + 1:
                time.sleep(60)
                continue

            # Prepare volatility input
            vol_data = scaler.transform(latest_data[features])
            X_vol = vol_data[-seq_length:].reshape(1, seq_length, len(features))
            row = latest_data.iloc[-1]
            price = row['close']

            # Predictions
            trend_pred = trend_model.predict(xgb.DMatrix([row[features]]))[0]
            vol_pred = volatility_model.predict(X_vol, verbose=0)[0, 0]
            vol_pred = 1 if vol_pred > scaler.inverse_transform([[0.5] + [0] * (len(features) - 1))])[0, 0] else 0
            regime_pred = regime_model.predict([row[features]])[0]
            env = DummyVecEnv([lambda: TradingEnv(latest_data[features + ['close']])])
            obs = env.reset()
            env.envs[0].current_step = len(latest_data) - 1
            action, _ = execution_model.predict(obs)
            exec_pred = action[0] % 2
            stack = np.array([[trend_pred, vol_pred, regime_pred, exec_pred]])
            ensemble_pred = ensemble_model.predict(stack)[0]

            # Trading logic
            if position > 0:
                if
            price <= entry_price * (1 - sl) or price >= entry_price * (1 + tp):
            profit = position * (price - entry_price) - fee * position * price
            balance += position * price * (1 - fee)
            exchange.create_market_sell_order(symbol, position)
            trades.append(
                {'time': pd.Timestamp.now(), 'action': 'sell', 'price': price, 'balance': balance, 'profit': profit})
            logging.info("Sell executed: %s at %.2f, Balance: %.2f", symbol, price, balance)
            position = 0

            if ensemble_pred == 1 and balance >= price and position == 0:
                position = balance * 0.1 / price  # 10% position sizing
            balance -= position * price * (1 + fee)
            entry_price = price
            exchange.create_market_buy_order(symbol, position)
            trades.append(
                {'time': pd.Timestamp.now(), 'action': 'buy', 'price': price, 'balance': balance, 'profit': 0})
            logging.info("Buy executed: %s at %.2f, Balance: %.2f", symbol, price, balance)

            # Dashboard update
            equity = balance + position * price
            sio.emit('training_update', {
                'model': 'livetrade',
                'epoch': len(trades),
                'loss': float(equity),  # Equity as "loss"
                'val_loss': float(np.std([t['profit'] for t in trades[-10:]]) if len(trades) > 10 else 0)
            })

            # Save trades periodically
            if len(trades) % 10 == 0:
                pd.DataFrame(trades).to_csv(LIVE_DIR / 'livetrade_trades.csv', index=False)

            time.sleep(60)  # Wait for next minute

    except Exception as e:
        logging.error("Error in livetrade: %s", str(e))
        sio.emit('error', {'model': 'livetrade', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET')
        if not api_key or not secret:
            raise ValueError("Set BINANCE_API_KEY and BINANCE_SECRET environment variables")
        livetrade(api_key, secret)
    except Exception as e:
        print(f"Failed to start live trading: {e}")