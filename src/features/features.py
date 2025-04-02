import pandas as pd
import ta

def calculate_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['supertrend'] = ta.trend.SuperTrend(df['high'], df['low'], df['close'], period=7, multiplier=3).supertrend()
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.BollingerBands(df['close']).bollinger_hband(), \
                                                     ta.volatility.BollingerBands(df['close']).bollinger_mavg(), \
                                                     ta.volatility.BollingerBands(df['close']).bollinger_lband()
    df['cci'] = ta.momentum.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    return df
