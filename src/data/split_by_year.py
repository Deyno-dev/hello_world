# Artificial Traders v4/Multi_Ai/src/data/split_by_year.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_SPLIT_DIR = PROJECT_ROOT / 'data' / 'split'
DATA_SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def split_by_year(input_file):
    print(f"Loading {input_file}")
    df = pd.read_csv(input_file, parse_dates=['timestamp'], index_col='timestamp')

    print("Splitting by year")
    for year, group in df.groupby(df.index.year):
        output_file = DATA_SPLIT_DIR / f'BTC_USD_1min_{year}.csv'
        group.to_csv(output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    input_file = DATA_RAW_DIR / 'BTC_USD_1min_full.csv'
    split_by_year(input_file)