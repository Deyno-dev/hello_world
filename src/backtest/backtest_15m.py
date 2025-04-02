# trading_scripts/backtest_15m.py
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "data/BTC_USD_15min.csv"
MODEL_PATH = "models/xgboost_15m_model.json"
FEE_RATE = 0.0003  # 0.03% per trade (lower than 1m)
SLIPPAGE = 0.00005  # Reduced slippage assumption

def calculate_features(df):
    """Generate 15-minute specific features"""
    df = df.copy()
    
    # Core price transformations
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    
    # Enhanced volatility features
    df['volatility_4h'] = df['returns'].rolling(16).std()  # 4-hour window
    df['volatility_1d'] = df['returns'].rolling(96).std()   # Daily volatility
    df['atr_14'] = calculate_atr(df, 14)  # Average True Range
    
    # Trend indicators
    df['adx_14'] = calculate_adx(df, 14)
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Volume analysis
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(72).mean()) / \
                         df['volume'].rolling(72).std()
    df['obv'] = calculate_obv(df['close'], df['volume'])
    
    # Time-based features
    df['session'] = pd.cut(df.index.hour,
                          bins=[0, 8, 16, 24],
                          labels=['Asian', 'European', 'US'])
    df = pd.get_dummies(df, columns=['session'], prefix='session')
    
    # Lagged features for momentum
    for lag in [1, 4, 12]:  # 15m, 1h, 3h
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    return df.dropna()

def execute_strategy(df):
    """15m strategy execution with improved risk management"""
    # Entry/exit logic
    df['position'] = np.select(
        [
            (df['predicted_return'] > FEE_RATE*2) & (df['adx_14'] > 25),
            (df['predicted_return'] < -FEE_RATE*2) & (df['adx_14'] > 25)
        ],
        [1, -1],
        default=0
    )
    
    # Position smoothing
    df['position'] = df['position'].rolling(4, min_periods=1).mean()
    
    # Calculate returns with realistic execution
    df['trade_size'] = df['position'].diff().abs()
    df['strategy_return'] = (
        df['position'].shift(1) * df['returns'] -
        df['trade_size'] * (FEE_RATE + SLIPPAGE)
    )
    
    return df

def main(args):
    print(f"🚀 Starting 15m backtest for chat {args.chat_id}")
    
    # Load data and model
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    # Feature engineering
    df = calculate_features(df)
    
    # Make predictions
    dmatrix = xgb.DMatrix(df[model.feature_names])
    df['predicted_return'] = model.predict(dmatrix)
    
    # Execute strategy
    results = execute_strategy(df)
    
    # Calculate performance metrics
    strategy_returns = results['strategy_return']
    performance = {
        'total_return': strategy_returns.sum(),
        'sharpe_ratio': (strategy_returns.mean() / 
                        strategy_returns.std()) * np.sqrt(35040),  # Annualized
        'sortino_ratio': (strategy_returns.mean() /
                         strategy_returns[strategy_returns < 0].std()) * np.sqrt(35040),
        'max_drawdown': (strategy_returns.cumsum()
                        .expanding().max() - 
                        strategy_returns.cumsum()).max(),
        'profit_factor': (strategy_returns[strategy_returns > 0].sum() /
                         abs(strategy_returns[strategy_returns < 0].sum())),
        'momentum_correlation': results['strategy_return'].rolling(96).mean().corr(
                                results['returns'].rolling(96).mean())
    }
    
    # Generate enhanced visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Equity curve
    results['strategy_cum'] = strategy_returns.cumsum()
    results['buy_hold_cum'] = results['returns'].cumsum()
    results['strategy_cum'].plot(ax=ax1, label='15m Strategy')
    results['buy_hold_cum'].plot(ax=ax1, label='Buy & Hold')
    ax1.set_title("15-Minute Strategy Performance")
    ax1.legend()
    
    # Position sizing
    results['position'].plot(ax=ax2, kind='area', alpha=0.3)
    ax2.set_title("Position Sizing")
    
    plt.tight_layout()
    plt.savefig(f"results/15m_backtest_{args.chat_id}.png")
    
    # Save results
    with open(f"results/15m_backtest_{args.chat_id}.json", 'w') as f:
        json.dump(performance, f)
    
    print("✅ 15m Backtest completed!")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-id', required=True)
    args = parser.parse_args()
    
    main(args)