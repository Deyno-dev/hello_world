# backtest_multi_ai.py
import pandas as pd

def backtest(df):
    df['strategy_return'] = df['close'].pct_change()  # Placeholder
    df.to_csv('../../results/backtest_results.csv')
    print("Backtest complete!")

if __name__ == "__main__":
    df = pd.read_csv('../../data/raw/BTC_USD_1min_full.csv')
    backtest(df)
