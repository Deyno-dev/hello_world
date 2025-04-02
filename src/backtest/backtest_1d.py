# trading_scripts/backtest_1d.py
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = "data/BTC_USD_1d.csv"
MODEL_PATH = "models/xgboost_1d_model.json"
FEE_RATE = 0.00015  # 0.015% per trade
SLIPPAGE = 0.00002   # Minimal slippage assumption

def calculate_features(df):
    """Daily feature engineering for strategic positioning"""
    df = df.copy()
    
    # Core transformations
    df['log_returns'] = np.log(df['close']/df['close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(21).std() * np.sqrt(365)
    
    # Macro-aligned technical indicators
    df['monthly_trend'] = df['close'].rolling(21).mean() / df['close'].rolling(63).mean()
    df['quarterly_momentum'] = df['close'].pct_change(63)
    df['annual_vol'] = df['log_returns'].rolling(252).std() * np.sqrt(252)
    
    # Smart money indicators
    df['supply_zone'] = df['high'].rolling(21).max()
    df['demand_zone'] = df['low'].rolling(21).min()
    df['volume_profile'] = df['volume'].rolling(252).apply(
        lambda x: x.quantile(0.95)/x.quantile(0.05))
    
    # Cyclical features
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df = pd.get_dummies(df, columns=['month', 'quarter'], prefix=['m', 'q'])
    
    # Market regime classification
    conditions = [
        df['volatility'] < 0.4,
        (df['volatility'] >= 0.4) & (df['volatility'] < 0.8),
        df['volatility'] >= 0.8
    ]
    df['regime'] = np.select(conditions, [0, 1, 2], default=0)
    
    return df.dropna()

def execute_strategy(df):
    """Daily strategy with portfolio optimization"""
    # Signal generation
    df['raw_signal'] = np.where(
        df['predicted_return'] > FEE_RATE*3, 1,
        np.where(df['predicted_return'] < -FEE_RATE*3, -1, 0)
    
    # Trend confirmation filter
    df['trend_filter'] = (df['close'].rolling(21).mean() > 
                         df['close'].rolling(63).mean()).astype(int)
    df['position'] = df['raw_signal'] * df['trend_filter']
    
    # Volatility-adjusted position sizing
    df['position_size'] = 0.1 / df['volatility']
    df['position'] = df['position'] * np.clip(df['position_size'], 0.05, 0.5)
    
    # Portfolio simulation
    df['strategy_return'] = (
        df['position'].shift(1) * df['log_returns'] -
        (abs(df['position'].diff()) * (FEE_RATE + SLIPPAGE)
    )
    
    return df

def main(args):
    print(f"🚀 Starting 1D backtest for chat {args.chat_id}")
    
    # Load data and model
    df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    # Feature engineering
    df = calculate_features(df)
    
    # Model prediction
    dmatrix = xgb.DMatrix(df[model.feature_names])
    df['predicted_return'] = model.predict(dmatrix)
    
    # Execute strategy
    results = execute_strategy(df)
    
    # Calculate advanced metrics
    strategy_returns = results['strategy_return']
    performance = {
        'cagr': (np.exp(strategy_returns.sum()) ** (365/len(df))) - 1,
        'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
        'sortino': strategy_returns.mean() / 
                  strategy_returns[strategy_returns < 0].std() * np.sqrt(252),
        'max_drawdown': (strategy_returns.cumsum().expanding().max() - 
                        strategy_returns.cumsum()).max(),
        'profit_factor': (np.exp(strategy_returns[strategy_returns > 0].sum()) /
                        np.exp(-strategy_returns[strategy_returns < 0].sum()),
        'regime_performance': {
            str(r): strategy_returns[df['regime'] == r].sum() 
            for r in [0, 1, 2]
        }
    }
    
    # Generate institutional-grade visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1)
    
    # Equity curve
    ax1 = fig.add_subplot(gs[:2, :])
    (np.exp(strategy_returns.cumsum()) - 1).plot(ax=ax1, label='Strategy')
    (np.exp(df['log_returns'].cumsum()) - 1).plot(ax=ax1, label='Buy & Hold')
    ax1.set_title("Daily Strategy Performance", fontsize=14)
    ax1.set_ylabel("Return (%)", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    
    # Drawdowns
    ax2 = fig.add_subplot(gs[2, :])
    drawdown = (strategy_returns.cumsum().expanding().max() - 
               strategy_returns.cumsum())
    drawdown.plot(ax=ax2, color='red', alpha=0.3)
    ax2.fill_between(drawdown.index, drawdown, color='red', alpha=0.1)
    ax2.set_title("Strategy Drawdown", fontsize=14)
    
    # Position sizing
    ax3 = fig.add_subplot(gs[3, :])
    results['position'].plot(ax=ax3, kind='area', alpha=0.3)
    ax3.set_title("Position Sizing", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"results/1d_backtest_{args.chat_id}.png", dpi=150)
    
    # Save results
    with open(f"results/1d_backtest_{args.chat_id}.json", 'w') as f:
        json.dump(performance, f)
    
    print("✅ 1D Backtest completed!")
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat-id', required=True)
    args = parser.parse_args()
    
    main(args)