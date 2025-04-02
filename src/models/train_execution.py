# train_execution.py
import pandas as pd
from stable_baselines3 import DQN
import gym

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df

def train_execution_model(df):
    env = TradingEnv(df)
    model = DQN('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=1000)
    model.save('../models/execution/dqn_execution')
    return model

if __name__ == "__main__":
    df = pd.read_csv('../../data/raw/BTC_USD_1min_full.csv')
    train_execution_model(df)
