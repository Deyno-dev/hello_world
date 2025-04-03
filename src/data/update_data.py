# Artificial Traders v4/Multi_Ai/src/data/update_data.py
import ccxt
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'

def update_data():
    exchange = ccxt.binance()
    since = pd.read_csv(DATA_PATH)['timestamp'].max() if DATA_PATH.exists() else None
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        df = pd.concat([existing, df]).drop_duplicates(subset='timestamp')
    df.to_csv(DATA_PATH, index=False)
    print("Data updated!")

if __name__ == "__main__":
    update_data()