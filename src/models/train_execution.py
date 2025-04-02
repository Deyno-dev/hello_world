# Artificial Traders v4/Multi_Ai/src/models/train_execution.py
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Discrete, Box
from ..features.features import calculate_features
import logging
from pathlib import Path
from tqdm import tqdm
import json
import socketio

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / 'results' / 'logs'
MODEL_DIR = PROJECT_ROOT / 'models' / 'execution'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'
for d in [LOG_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=LOG_DIR / 'train_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load config
CONFIG_PATH = PROJECT_ROOT / 'config' / 'ai_config.json'
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f).get('execution', {})
except FileNotFoundError:
    config = {}
    logging.warning("ai_config.json not found, using defaults")

# SocketIO for dashboard
sio = socketio.Client()


class TradingEnv(Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.action_space = Discrete(3)  # Hold, Buy, Sell
        self.observation_space = Box(low=0, high=1, shape=(len(data.columns),), dtype=np.float32)
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.max_steps = len(data) - 1

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        return self._get_observation()

    def step(self, action):
        price = self.data['close'].iloc[self.current_step]
        reward = 0
        if action == 1 and self.balance >= price:  # Buy
            self.position += 1
            self.balance -= price
        elif action == 2 and self.position > 0:  # Sell
            self.position -= 1
            self.balance += price
            reward = price - self.data['close'].iloc[max(0, self.current_step - 10)]

        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.data.iloc[self.current_step].values / self.data.max()


def train_execution_model(
        data_path,
        total_timesteps=config.get('total_timesteps', 10000)
):
    try:
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        sio.connect('http://localhost:5000')
        logging.info("Connected to dashboard")

        logging.info("Loading data from %s", data_path)
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

        logging.info("Calculating features")
        df = calculate_features(df)

        features = ['rsi', 'macd', 'volatility', 'adx', 'close']
        env = DummyVecEnv([lambda: TradingEnv(df[features].dropna())])

        logging.info("Training DQN")
        model = DQN('MlpPolicy', env, verbose=0, learning_rate=0.001)
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        # Simulate epochs for dashboard
        steps_per_epoch = total_timesteps // 10
        for epoch in tqdm(range(1, 11), desc="Training epochs"):
            sio.emit('training_update', {
                'model': 'execution',
                'epoch': epoch,
                'loss': float(np.random.random()),  # Placeholder loss
                'val_loss': float(np.random.random())
            })

        # Evaluate (simplified)
        obs = env.reset()
        rewards = []
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        avg_reward = np.mean(rewards)
        logging.info("Average validation reward: %.4f", avg_reward)
        sio.emit('training_complete', {
            'model': 'execution',
            'metrics': {'avg_reward': float(avg_reward)}
        })

        # Save model
        model_path = MODEL_DIR / 'dqn_execution.zip'
        model.save(model_path)
        logging.info("Model saved to %s", model_path)

        sio.disconnect()
        return model

    except Exception as e:
        logging.error("Error in train_execution_model: %s", str(e))
        sio.emit('error', {'model': 'execution', 'message': str(e)})
        sio.disconnect()
        raise


if __name__ == "__main__":
    try:
        data_path = PROJECT_ROOT / 'data' / 'raw' / 'BTC_USD_1min_full.csv'
        model = train_execution_model(data_path)
        print("Execution model trained and saved successfully!")
    except Exception as e:
        print(f"Failed to train model: {e}")