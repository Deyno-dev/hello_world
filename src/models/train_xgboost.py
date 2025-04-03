import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import keyboard
import telegram
import asyncio

# ✅ Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="model_training.log"
)

# ✅ Load configuration
CONFIG_PATH = os.path.abspath("training_config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DATASET_PATH = os.path.abspath(os.path.join(config["data_folder"], "BTC_USD_1min_expanded.csv"))
MODEL_PATH = os.path.abspath(config["model_path"])
EXPECTED_FEATURES = config["expected_features"]
TARGET_COLUMNS = config["target_columns"]

# ✅ Ensure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

# ✅ Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "8069506330:AAFCOeDXulQG9yeRB9yHsWFLrlluWNHqfzw"  # Replace with your bot token
TELEGRAM_CHAT_ID = "-1002220357189"  # Replace with your chat ID

# Initialize Telegram Bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)


async def send_telegram_message(message):
    """Send a message to the Telegram bot."""
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        logging.error(f"❌ Failed to send Telegram message: {str(e)}")


def add_time_features(df):
    """Add time-based features to the DataFrame."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    return df


def preprocess_data(df, chunk_idx):
    """Preprocess data: Add time-based features, normalize, and handle missing values."""
    # Add time-based features
    df = add_time_features(df)

    # Handle missing values - forward fill
    df = df.ffill()  # Forward fill missing values

    # Normalize numerical features
    scaler = StandardScaler()
    df[EXPECTED_FEATURES] = scaler.fit_transform(df[EXPECTED_FEATURES])

    # Cache the chunk for future use
    cache_filename = os.path.join(CACHE_DIR, f"chunk_{chunk_idx + 1}.pkl")
    df.to_pickle(cache_filename)  # Save the chunk to cache
    logging.info(f"Chunk {chunk_idx + 1} preprocessed and cached at {cache_filename}")

    return df


def train_xgboost():
    """Train or incrementally update the XGBoost model using GPU acceleration."""
    model_file = os.path.join(MODEL_PATH, "xgboost_model.json")

    # ✅ Load existing model or initialize a new one
    if os.path.exists(model_file):
        model = xgb.Booster()
        model.load_model(model_file)
        logging.info("🔄 Loaded existing XGBoost model.")
    else:
        model = None
        logging.info("🚀 Initializing new XGBoost model.")

    # ✅ Read dataset in chunks
    try:
        logging.info(f"Starting to load dataset from: {DATASET_PATH}")
        asyncio.run(send_telegram_message(f"Starting to load dataset from: {DATASET_PATH}"))

        for i, chunk in enumerate(pd.read_csv(DATASET_PATH, chunksize=CHUNKSIZE)):
            cache_filename = os.path.join(CACHE_DIR, f"chunk_{i + 1}.pkl")

            # If chunk is cached, load from cache
            if os.path.exists(cache_filename):
                chunk = pd.read_pickle(cache_filename)
                logging.info(f"Loaded cached chunk {i + 1} from {cache_filename}")
            else:
                chunk = preprocess_data(chunk, i)

            # ✅ Ensure dataset contains expected features
            missing_features = [col for col in EXPECTED_FEATURES if col not in chunk.columns]
            if missing_features:
                logging.warning(f"⚠️ Missing columns in {DATASET_PATH}: {missing_features}")
                continue  # Skip this chunk

            # ✅ Ensure target column exists
            target_col = next((col for col in TARGET_COLUMNS if col in chunk.columns), None)
            if not target_col:
                logging.warning(f"❌ No valid target column found in {DATASET_PATH}")
                continue

            # ✅ Extract features and target
            X = chunk[EXPECTED_FEATURES]
            y = chunk[target_col]

            # ✅ Convert to DMatrix for GPU optimization
            dtrain = xgb.DMatrix(X, label=y)

            # ✅ Train model (incrementally update if it exists)
            if model is None:
                model = xgb.train(
                    params=HYPERPARAMETERS,
                    dtrain=dtrain,
                    num_boost_round=NUM_BOOST_ROUNDS  # Correct replacement for n_estimators
                )
            else:
                model = xgb.train(
                    params=HYPERPARAMETERS,
                    dtrain=dtrain,
                    num_boost_round=NUM_BOOST_ROUNDS,  # Fix applied
                    xgb_model=model  # Incremental training
                )

            logging.info(f"📊 Processed chunk {i + 1}: {chunk.shape[0]} rows")

        # ✅ Save the final trained model
        model.save_model(model_file)
        logging.info(f"💾 Model updated and saved: {model_file}")
        asyncio.run(send_telegram_message(f"Training completed successfully. Model saved at {model_file}"))

    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {str(e)}")
        asyncio.run(send_telegram_message(f"❌ Data loading failed. Exiting training."))
        logging.error("❌ Data loading failed. Exiting training.")


if __name__ == "__main__":
    round_count = 0

    while True:
        try:
            round_count += 1
            logging.info(f"🚀 Starting Training Round {round_count}...")
            asyncio.run(send_telegram_message(f"🚀 Starting Training Round {round_count}..."))
            train_xgboost()
            logging.info(f"✅ Training Round {round_count} Completed Successfully!")
            asyncio.run(send_telegram_message(f"✅ Training Round {round_count} Completed Successfully!"))

            print("\n⏳ Waiting for next round... Press ESC to exit.")
            for _ in range(10):  # Wait 10 seconds before next round
                if keyboard.is_pressed("esc"):
                    print("\n❌ Exiting Training Loop Safely!")
                    asyncio.run(send_telegram_message("❌ Exiting Training Loop Safely!"))
                    logging.info("❌ User exited training loop safely.")
                    exit()
        except Exception as e:
            logging.error(f"🔥 Critical error: {str(e)}")
            asyncio.run(send_telegram_message(f"🔥 Critical error: {str(e)}"))
            break
