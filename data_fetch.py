"""
data_fetch.py
Fetches historical stock data from Yahoo Finance for NSE/BSE tickers.
"""

import yfinance as yf
import pandas as pd


def normalize_ticker(ticker: str) -> str:
    """
    Auto-append .NS suffix for NSE tickers if not already present.
    """
    ticker = ticker.strip().upper()

    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"

    return ticker


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given NSE ticker.

    Args:
        ticker: Stock ticker symbol
        period: 6mo, 1y, 2y etc.

    Returns:
        DataFrame with OHLCV data
    """

    ticker = normalize_ticker(ticker)

    try:
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            raise ValueError(f"No data found for {ticker}")

        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.dropna(inplace=True)

        return data

    except Exception as e:
        raise RuntimeError(f"Failed to fetch stock data: {str(e)}")


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic company metadata.
    """

    ticker = normalize_ticker(ticker)

    try:
        info = yf.Ticker(ticker).info

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", None),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSE"),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
        }

    except Exception:

        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
        }


if __name__ == "__main__":

    df = fetch_stock_data("RELIANCE")

    print(df.tail())

    info = get_stock_info("RELIANCE")

    print(info)