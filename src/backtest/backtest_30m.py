# trading_scripts/backtest_30m.py
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "data/BTC_USD_30min.csv"
MODEL_PATH = "models/xgboost_30m_model.json"
FEE_RATE = 0.0002  # 0.02% per trade
SLIPPAGE = 0.00003  # Reduced slippage assumption

def calculate_features(df):
    """30-minute specific feature engineering"""
    df = df.copy()
    
    # Price transformations
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    
    # Advanced volatility metrics
    df['volatility_6h'] = df['returns'].rolling(12).std()  # 6-hour window
    df['volatility_1d'] = df['returns'].rolling(48).std()   # Daily volatility
    df['atr_14'] = calculate_atr(df, 14)
    
    # Trend and momentum indicators
    df['ichimoku_cloud'] = calculate_ichimoku(df)
    df['ema_34'] = df['close'].ewm(span=34).mean()
    df['ema_89'] = df['close'].ewm(span=89).mean()
    df['dmi'] = calculate_dmi(df, 14)
    
    # Volume analysis
    df['vwap'] = calculate_vwap(df)
    df['volume_profile'] = df['volume'].rolling(48).apply(
        lambda x: x.quantile(0.8)/x.quantile(0.2))
    
    # Market regime features
    df['vol_regime'] = pd.cut(df['volatility_1d'],
                             bins=[0, 0.01, 0.03, 1],
                             labels=['low', 'medium', 'high'])
    df = pd.get_dummies(df, columns=['vol_regime'], prefix='vol')
    
    # Lagged features for momentum
    for lag in [2, 8, 24]:  # 1h, 4h, 12h
        df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    return df.dropna()

def execute_strategy(df):
    """30m strategy with regime-aware execution"""
    # Entry conditions
    long_cond = (
        (df['predicted_return'] > FEE_RATE*1.5) &
        (df['ichimoku_cloud'] == 1) &
        (df['ema_34'] > df['ema_89'])
    )
    
    short_cond = (
        (df['predicted_return'] < -FEE_RATE*1.5) &
        (df['ichimoku_cloud'] == -1) &
        (df['ema_34'] < df['ema_89'])
    )
    
    df['position'] = np.select([long_cond, short_cond], [1, -1], default=0)
    
    # Position sizing based on volatility
    df['position_size'] = np.clip(0.5 / df['volatility_1d'], 0.1, 2.0)
    df['position'] = df['position'] * df['position_size']
    
    # Calculate returns with realistic execution
    df['trade_size'] = df['position'].diff().abs()
    df['strategy_return'] = (
        df['position'].shift(1) * df['returns'] -
        df['trade_size'] * (FEE_RATE + SLIPPAGE)
    )
    
    return df

def main(args):
    print(f"🚀 Starting 30m backtest for chat {args.chat_id}")
    
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
                        strategy_returns.std()) * np.sqrt(17520),  # Annualized
        'calmar_ratio': strategy_returns.sum() / 
                       performance['max_drawdown'],
        'max_drawdown': (strategy_returns.cumsum()
                        .expanding().max() - 
                        strategy_returns.cumsum()).max(),
        'volatility_ratio': strategy_returns.std() / 
                           df['returns'].std(),
        'regime_performance': {
            'low': strategy_returns[df['vol_regime'] == 'low'].sum(),
            'medium': strategy_returns[df['vol_regime'] == 'medium'].sum(),
            'high': strategy_returns[df['vol_regime'] == 'high'].sum()
        }
    }
    
    # Generate visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Equity curve
    results['strategy_cum'] = strategy_returns.cumsum()
    results['buy_hold_cum'] = results['returns'].cumsum()
    results['strategy_cum'].plot(ax=ax1, label='30m Strategy')
    results['buy_hold_cum'].plot(ax=ax1, label='Buy & Hold')
    ax1.set_title("30-Minute Strategy Performance")
    ax1.legend()
    
    # Position sizing
    results['position'].plot(ax=ax2, kind='area', alpha=0.3)
    ax2.set_title("Position Sizing")
    
    # Volatility regime
    pd.cut(df['volatility_1d'], bins=[0, 0.01, 0.03, 1]).value_counts().plot(
        kind='bar', ax=ax3, rot=0)
    ax3.set_title("Volatility Regime Distribution")
    
    plt.tight_layout()
    plt.savefig(f"results/30m_backtest_{args.chat_id}.png")
    
    # Save results
    with open(f"results/30m_backtest_{args.chat_id}.json", 'w') as f:
        json.dump(performance, f)
    
    print("✅ 30m Backtest completed!")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-id', required=True)
    args = parser.parse_args()
    
    main(args)