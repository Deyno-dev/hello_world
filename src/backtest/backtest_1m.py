# trading_scripts/backtest_1m.py
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

DATA_PATH = "data/BTC_USD_1min.csv"
MODEL_PATH = "models/xgboost_1m_model.json"
FEE_RATE = 0.0005  # 0.05% per trade
SLIPPAGE = 0.0001   # 0.01% slippage

def calculate_features(df):
    """Generate 1-minute specific features"""
    df = df.copy()
    
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))

    df['volatility_5m'] = df['returns'].rolling(5).std()
    df['volatility_15m'] = df['returns'].rolling(15).std()
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['macd'], df['signal'] = calculate_macd(df['close'])
    df['obv'] = calculate_obv(df['close'], df['volume'])
    
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['is_ny_open'] = ((df.index.hour >= 13) & (df.index.hour < 20)).astype(int)

    for lag in [1, 5, 15]:
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
    
    return df.dropna()

def execute_strategy(df):
    """Run trading strategy with realistic execution"""
    df['position'] = np.where(df['predicted_return'] > FEE_RATE + SLIPPAGE, 1,
                             np.where(df['predicted_return'] < -FEE_RATE - SLIPPAGE, -1, 0))
    
    df['trade_size'] = df['position'].diff().abs()
    df['strategy_return'] = (
        df['position'].shift(1) * df['returns'] -
        df['trade_size'] * (FEE_RATE + SLIPPAGE)
    
    return df

def main(args):
    print(f"🚀 Starting 1m backtest for chat {args.chat_id}")
    
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    df = calculate_features(df)
    
    dmatrix = xgb.DMatrix(df[model.feature_names])
    df['predicted_return'] = model.predict(dmatrix)
    
    results = execute_strategy(df)
    
    performance = {
        'total_return': results['strategy_return'].sum(),
        'sharpe_ratio': (results['strategy_return'].mean() / 
                         results['strategy_return'].std()) * np.sqrt(525600),
        'max_drawdown': (results['strategy_return'].cumsum()
                        .expanding().max() - 
                        results['strategy_return'].cumsum()).max(),
        'win_rate': (results['strategy_return'] > 0).mean()
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    results['strategy_return'].cumsum().plot(ax=ax, label='Strategy')
    results['returns'].cumsum().plot(ax=ax, label='Buy & Hold')
    ax.set_title("1-Minute Strategy Performance")
    plt.savefig(f"results/1m_backtest_{args.chat_id}.png")
    
    with open(f"results/1m_backtest_{args.chat_id}.json", 'w') as f:
        json.dump(performance, f)
    
    print("✅ Backtest completed!")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-id', required=True)
    args = parser.parse_args()
    
    main(args)