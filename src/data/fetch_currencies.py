# Artificial Traders v4/Multi_Ai/src/data/fetch_currencies.py
import ccxt
import pandas as pd
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_currency_data(exchange, symbol, timeframe='1m', since='2020-01-01'):
    exchange = ccxt.binance()
    since_ts = exchange.parse8601(since)
    all_data = []

    print(f"Fetching {symbol}")
    while since_ts < exchange.milliseconds():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since_ts = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    output_file = DATA_RAW_DIR / f'{symbol.replace("/", "_")}_1min_full.csv'
    df.to_csv(output_file)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    symbols = ['ETH/USDT', 'XRP/USDT', 'LTC/USDT']
    for symbol in symbols:
        fetch_currency_data(None, symbol)