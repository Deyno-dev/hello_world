# Artificial Traders v4/Multi_Ai/src/cli.py
import argparse
from pathlib import Path
from .models import train_trend, train_volatility, train_regime, train_execution, train_ensemble
from .backtest import backtrade
from .livetrade import livetrade

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'


def main():
    parser = argparse.ArgumentParser(description="Artificial Traders v4 CLI")
    subparsers = parser.add_subparsers(dest='command', help="Available commands")

    # Train commands
    train_parser = subparsers.add_parser('train', help="Train a model")
    train_parser.add_argument('model', choices=['trend', 'volatility', 'regime', 'execution', 'ensemble'],
                              help="Model to train")
    train_parser.add_argument('--data', default=DATA_PATH, help="Path to training data")

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help="Run backtesting")
    backtest_parser.add_argument('--seq-length', type=int, default=20, help="Sequence length for volatility model")

    # Livetrade command
    live_parser = subparsers.add_parser('livetrade', help="Run live trading")
    live_parser.add_argument('--symbol', default='BTC/USDT', help="Trading pair")
    live_parser.add_argument('--timeframe', default='1m', help="Timeframe for OHLCV data")

    args = parser.parse_args()

    if args.command == 'train':
        if args.model == 'trend':
            train_trend.train_trend_model(DATA_PATH)
        elif args.model == 'volatility':
            train_volatility.train_volatility_model(DATA_PATH)
        elif args.model == 'regime':
            train_regime.train_regime_model(DATA_PATH)
        elif args.model == 'execution':
            train_execution.train_execution_model(DATA_PATH)
        elif args.model == 'ensemble':
            train_ensemble.train_ensemble_model(DATA_PATH)
    elif args.command == 'backtest':
        backtrade.backtrade(DATA_PATH, seq_length=args.seq_length)
    elif args.command == 'livetrade':
        import os
        api_key = os.getenv('BINANCE_API_KEY')
        secret = os.getenv('BINANCE_SECRET')
        if not api_key or not secret:
            raise ValueError("Set BINANCE_API_KEY and BINANCE_SECRET environment variables")
        livetrade.livetrade(api_key, secret, symbol=args.symbol, timeframe=args.timeframe)


if __name__ == "__main__":
    main()