"""
recommendation.py
Generates BUY / HOLD / SELL recommendation.
"""

import numpy as np
import pandas as pd


CONFIDENCE_STRONG = 0.70
CONFIDENCE_WEAK = 0.55

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30


def compute_risk_metrics(
    df: pd.DataFrame
):
    """
    Compute risk metrics
    """

    close = df["Close"].squeeze()

    daily_returns = close.pct_change().dropna()

    # Total return
    total_return = (
        close.iloc[-1] / close.iloc[0]
    ) - 1

    trading_days = len(df)

    # Annualized return
    annualized_return = (
        (1 + total_return)
        ** (252 / trading_days)
    ) - 1

    # VaR
    var_95 = np.percentile(
        daily_returns,
        5
    )

    # Sharpe ratio
    risk_free_daily = 0.065 / 252

    excess_returns = (
        daily_returns - risk_free_daily
    )

    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
    ) * np.sqrt(252)

    # Max drawdown
    rolling_max = close.cummax()

    drawdown = (
        close - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()

    # Current volatility
    current_volatility = (
        df["Volatility"].iloc[-1]
    )

    return {
        "annualized_return": round(
            annualized_return * 100,
            2
        ),
        "var_95": round(
            var_95 * 100,
            2
        ),
        "sharpe_ratio": round(
            sharpe,
            2
        ),
        "max_drawdown": round(
            max_drawdown * 100,
            2
        ),
        "current_volatility": round(
            current_volatility * 100,
            2
        ),
        "total_return_1y": round(
            total_return * 100,
            2
        ),
    }


def generate_recommendation(
    prediction: dict,
    df: pd.DataFrame
):
    """
    Generate BUY/HOLD/SELL recommendation
    """

    direction = prediction["direction"]

    confidence = prediction["confidence"]

    latest = df.iloc[-1]

    rsi = latest.get("RSI", 50)

    volatility = latest.get(
        "Volatility",
        0.2
    )

    macd = latest.get("MACD", 0)

    macd_signal = latest.get(
        "MACD_Signal",
        0
    )

    # Score system
    score = 0

    # Model signal
    if direction == "UP":

        if confidence >= CONFIDENCE_STRONG:
            score += 2
        else:
            score += 1

    else:

        if confidence >= CONFIDENCE_STRONG:
            score -= 2
        else:
            score -= 1

    # RSI adjustment
    if rsi > RSI_OVERBOUGHT:
        score -= 1

    elif rsi < RSI_OVERSOLD:
        score += 1

    # MACD adjustment
    if macd > macd_signal:
        score += 0.5
    else:
        score -= 0.5

    # Final action
    if score >= 2:

        action = "BUY"
        emoji = "🟢"
        color = "green"

    elif score <= -2:

        action = "SELL"
        emoji = "🔴"
        color = "red"

    else:

        action = "HOLD"
        emoji = "🟡"
        color = "orange"

    rationale = []

    if direction == "UP":

        rationale.append(
            f"The model predicts an upward move with "
            f"{confidence:.0%} confidence."
        )

    else:

        rationale.append(
            f"The model predicts a downward move with "
            f"{confidence:.0%} confidence."
        )

    # RSI explanation
    if rsi > RSI_OVERBOUGHT:

        rationale.append(
            f"RSI ({rsi:.1f}) suggests the stock "
            f"may be overbought."
        )

    elif rsi < RSI_OVERSOLD:

        rationale.append(
            f"RSI ({rsi:.1f}) suggests the stock "
            f"may be oversold."
        )

    else:

        rationale.append(
            f"RSI ({rsi:.1f}) is in a neutral zone."
        )

    # MACD explanation
    if macd > macd_signal:

        rationale.append(
            "MACD indicates bullish momentum."
        )

    else:

        rationale.append(
            "MACD indicates bearish momentum."
        )

    # Volatility warning
    if volatility > 0.35:

        rationale.append(
            f"⚠️ High volatility detected "
            f"({volatility * 100:.1f}%)."
        )

    risk_metrics = compute_risk_metrics(df)

    return {
        "action": action,
        "emoji": emoji,
        "color": color,
        "score": score,
        "rationale": "\n\n".join(rationale),
        "risk_metrics": risk_metrics,
    }


if __name__ == "__main__":

    from data_fetch import fetch_stock_data
    from indicators import add_indicators
    from model import train_model, predict

    df = fetch_stock_data("TCS")

    df = add_indicators(df)

    model, scaler, acc, _ = train_model(df)

    pred = predict(
        model,
        scaler,
        df
    )

    rec = generate_recommendation(
        pred,
        df
    )

    print(rec)