# train_regime.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_regime_model(df):
    X = df[['volatility', 'ema_fast', 'ema_slow']]
    y = pd.cut(df['volatility'], bins=[0, 0.01, 0.03, 1], labels=[0, 1, 2])
    model = RandomForestClassifier()
    model.fit(X, y)
    import pickle
    with open('../models/regime/rf_regime.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

if __name__ == "__main__":
    df = pd.read_csv('../../data/raw/BTC_USD_1min_full.csv')
    train_regime_model(df)
