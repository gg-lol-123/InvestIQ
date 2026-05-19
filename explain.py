"""
explain.py
SHAP explainability for the XGBoost model.
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from indicators import FEATURE_COLS


FEATURE_DESCRIPTIONS = {
    "RSI": "Momentum (RSI)",
    "MACD": "Trend Direction (MACD)",
    "MACD_Signal": "MACD Signal",
    "MACD_Hist": "MACD Histogram",
    "Price_vs_SMA20": "Price vs SMA20",
    "Volatility": "Market Volatility",
    "Volume_Change": "Volume Change",
}


def get_shap_values(
    model,
    scaler,
    df: pd.DataFrame
):
    """
    Compute SHAP values
    """

    X = df[FEATURE_COLS].values

    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_scaled).values

    return (
        shap_values,
        X_scaled,
        explainer
    )


def explain_latest(
    model,
    scaler,
    df: pd.DataFrame
):
    """
    Explain latest prediction
    """

    X = df[FEATURE_COLS].values

    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_scaled).values

    # Latest row
    last_shap = shap_values[-1]

    last_features = df[FEATURE_COLS].iloc[-1]

    contributions = dict(
        zip(FEATURE_COLS, last_shap)
    )

    # Sort by importance
    sorted_contribs = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_positive = [
        (f, v)
        for f, v in sorted_contribs
        if v > 0
    ][:3]

    top_negative = [
        (f, v)
        for f, v in sorted_contribs
        if v < 0
    ][:3]

    explanation_parts = []

    if top_positive:

        pos_text = ", ".join([
            f"{FEATURE_DESCRIPTIONS.get(f, f)} ({last_features[f]:.2f})"
            for f, _ in top_positive
        ])

        explanation_parts.append(
            f"📈 Bullish factors: {pos_text}."
        )

    if top_negative:

        neg_text = ", ".join([
            f"{FEATURE_DESCRIPTIONS.get(f, f)} ({last_features[f]:.2f})"
            for f, _ in top_negative
        ])

        explanation_parts.append(
            f"📉 Bearish factors: {neg_text}."
        )

    # Strongest feature
    top_feature, top_value = sorted_contribs[0]

    direction = (
        "positive"
        if top_value > 0
        else "negative"
    )

    explanation_parts.append(
        f"🔍 Strongest signal: "
        f"{FEATURE_DESCRIPTIONS.get(top_feature, top_feature)} "
        f"with a {direction} influence."
    )

    return {
        "shap_contributions": contributions,
        "plain_english": "\n\n".join(explanation_parts),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "sorted_contributions": sorted_contribs,
        "shap_values": shap_values,
        "X_scaled": X_scaled,
        "explainer": explainer,
    }


def plot_shap_bar(
    shap_values,
    feature_names=FEATURE_COLS
):
    """
    Global feature importance plot
    """

    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    sorted_idx = np.argsort(mean_abs_shap)

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx]
    )

    ax.set_xlabel(
        "Mean |SHAP Value|"
    )

    ax.set_title(
        "Global Feature Importance"
    )

    plt.tight_layout()

    return fig


def plot_shap_latest(
    shap_result: dict
):
    """
    Local SHAP contribution plot
    """

    contribs = shap_result[
        "sorted_contributions"
    ]

    features = [
        FEATURE_DESCRIPTIONS.get(f, f)
        for f, _ in contribs
    ]

    values = [
        v for _, v in contribs
    ]

    colors = [
        "green" if v > 0 else "red"
        for v in values
    ]

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.barh(
        features[::-1],
        values[::-1],
        color=colors[::-1]
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--"
    )

    ax.set_xlabel(
        "SHAP Contribution"
    )

    ax.set_title(
        "Latest Prediction Drivers"
    )

    plt.tight_layout()

    return fig


if __name__ == "__main__":

    from data_fetch import fetch_stock_data
    from indicators import add_indicators
    from model import train_model

    df = fetch_stock_data("RELIANCE")

    df = add_indicators(df)

    model, scaler, acc, _ = train_model(df)

    result = explain_latest(
        model,
        scaler,
        df
    )

    print(result["plain_english"])