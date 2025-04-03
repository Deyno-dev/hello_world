# Artificial Traders v4/Multi_Ai/src/auto_train.py
import subprocess
from pathlib import Path
import logging
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_SPLIT_DIR = PROJECT_ROOT / 'data' / 'split'
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / 'auto_train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODELS = ['trend', 'volatility', 'regime', 'execution', 'ensemble']


def auto_train():
    datasets = list(DATA_SPLIT_DIR.glob('*.csv'))
    if not datasets:
        logging.error("No datasets found in data/split/. Run split_by_year.py first.")
        raise FileNotFoundError("No datasets available")

    for model in tqdm(MODELS, desc="Training models"):
        for data_file in tqdm(datasets, desc=f"Datasets for {model}", leave=False):
            logging.info(f"Training {model} on {data_file}")
            cmd = f"python -m src.cli train {model} --data {data_file}"
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                logging.info(f"Completed {model} on {data_file}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed {model} on {data_file}: {e.stderr}")

    logging.info("All training completed")


if __name__ == "__main__":
    try:
        auto_train()
    except Exception as e:
        logging.error(f"Auto training failed: {e}")
        print(f"Error: {e}")