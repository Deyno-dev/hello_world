# Artificial Traders v4 - Multi_Ai

A multi-AI crypto trading project with models for trend prediction, volatility forecasting, regime detection, execution strategy, and ensemble learning.

## Project Structure
- `src/models/`: Training scripts for each model.
- `src/backtest/`: Backtesting logic.
- `src/livetrade/`: Live trading with CCXT.
- `src/dashboard/`: Flask-based dashboard for real-time monitoring.
- `config/`: Configuration files (e.g., `ai_config.json`).
- `data/raw/`: Place `BTC_USD_1min_full.csv` here (2.5GB, not tracked).
- `results/`: Logs, predictions, plots, and backtest/livetrade outputs.
- `Dockerfile`: Container setup with GPU support.

## Setup
1. **Clone**: `git clone https://github.com/Deyno-dev/Artificial_Traders_v4.git`
2. **Docker**:
   - Build: `docker build -t artificial-traders:latest .`
   - Run: `docker run -it -v "$(pwd):/app" -p 5000:5000 artificial-traders:latest`
3. **Data**: Download `BTC_USD_1min_full.csv` from [Insert Cloud Link] and place in `data/raw/`.
4. **API Keys**: Set `BINANCE_API_KEY` and `BINANCE_SECRET` as environment variables.

## Usage
- **CLI**:
  - Train: `python -m src.cli train volatility`
  - Backtest: `python -m src.cli backtest --seq-length 20`
  - Live Trade: `python -m src.cli livetrade --symbol BTC/USDT`
- **Dashboard**: 
  - Start: `python src/dashboard/app.py`
  - Visit: `http://localhost:5000`
  - Click "Start Backtest" or "Start Live Trading" in respective tabs.

## Models
- **Trend**: XGBoost for price direction prediction.
- **Volatility**: LSTM with Optuna-tuned hyperparameters.
- **Regime**: Random Forest for market regime classification.
- **Execution**: DQN for trade execution timing.
- **Ensemble**: Voting classifier combining all models.

## Backtesting
- Simulates trading with fees (0.1%), stop-loss (2%), and take-profit (5%).
- Metrics: Final balance, annualized return, Sharpe ratio, max drawdown, win rate, profit factor.

## Live Trading
- Uses CCXT to trade on Binance (configurable for other exchanges).
- Fetches 1-minute OHLCV data, predicts with ensemble model, and executes market orders.
- Monitors equity and trade volatility on the dashboard.

## Requirements
See `requirements.txt` for dependencies, installed via Docker or `pip install -r requirements.txt`.

## Contributing
Fork, branch, and PR! See issues for open tasks.