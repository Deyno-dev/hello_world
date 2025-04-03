import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'

def generate_synthetic_data(num_samples=1000, seq_length=20, num_features=5):
    gan = load_model(DATA_DIR / 'pretrained_gan.h5', compile=False)
    noise = np.random.normal(0, 1, (num_samples, seq_length, num_features))
    synthetic = gan.predict(noise, verbose=0)
    synthetic = synthetic * 1000 + 50000  # Rough BTC range
    synthetic_df = pd.DataFrame(
        synthetic.reshape(-1, num_features),
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    SYNTHETIC_DIR.mkdir(exist_ok=True)
    output_path = SYNTHETIC_DIR / 'synthetic_BTC_USD_1min.csv'
    synthetic_df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    synthetic_path = generate_synthetic_data()
    print(f"Generated synthetic data at {synthetic_path}")