"""
indicators.py
Computes technical indicators used as ML model features.
"""

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    """

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
):
    """
    Calculate MACD indicator
    """

    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False
    ).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock dataframe
    """

    df = df.copy()

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    # RSI
    df["RSI"] = compute_rsi(close)

    # MACD
    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Hist"]
    ) = compute_macd(close)

    # Moving averages
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()

    # Price vs SMA20
    df["Price_vs_SMA20"] = (
        (close - df["SMA_20"]) / df["SMA_20"]
    )

    # Volatility
    log_returns = np.log(close / close.shift(1))

    df["Volatility"] = (
        log_returns.rolling(window=20).std()
        * np.sqrt(252)
    )

    # Daily return
    df["Daily_Return"] = close.pct_change()

    # Volume change
    df["Volume_Change"] = volume.pct_change()

    # Target variable
    # 1 if next day return positive
    df["Target"] = (
        df["Daily_Return"].shift(-1) > 0
    ).astype(int)

    # Replace inf values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Drop NaN rows
    df.dropna(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


FEATURE_COLS = [
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Price_vs_SMA20",
    "Volatility",
    "Volume_Change"
]


if __name__ == "__main__":

    from data_fetch import fetch_stock_data

    raw_df = fetch_stock_data("TCS")

    enriched_df = add_indicators(raw_df)

    print(enriched_df[FEATURE_COLS + ["Target"]].tail())