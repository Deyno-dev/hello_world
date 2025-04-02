import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# ✅ Load dataset
DATASET_PATH = "Datasets/BTC_USD_1min_expanded.csv"
df = pd.read_csv(DATASET_PATH)

# ✅ Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# ✅ Load trained XGBoost model
model = xgb.Booster()
model.load_model("ai_custom/models/xgboost_model.json")

# ✅ Prepare features
EXPECTED_FEATURES = [
    "Open", "High", "Low", "Volume",
    "hour", "minute", "dayofweek", "month",
    "is_weekend"
]
X_test = df[EXPECTED_FEATURES]

# ✅ Predict future prices
dtest = xgb.DMatrix(X_test)
df['predicted_price'] = model.predict(dtest)

# ✅ Simulate Trading Strategy
df['returns'] = df['y'].pct_change()  # Actual price change
df['predicted_returns'] = df['predicted_price'].pct_change()

# ✅ Simple Strategy: Buy if model predicts positive returns
df['position'] = np.where(df['predicted_returns'] > 0, 1, -1)  # 1 = Buy, -1 = Short
df['strategy_returns'] = df['position'].shift(1) * df['returns']

# ✅ Cumulative Returns
df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()

# ✅ Plot Performance
plt.figure(figsize=(12, 6))
plt.plot(df['cumulative_returns'], label="Strategy Returns", color='blue')
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
plt.title("XGBoost Trading Strategy Performance")
plt.xlabel("Time")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid()
plt.show()

# ✅ Save results
df.to_csv("backtest_results.csv")
print(f"✅ Backtest complete! Total Strategy Return: {df['cumulative_returns'].iloc[-1] - 1:.2%}")
