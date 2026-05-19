"""
model.py
XGBoost classifier for stock direction prediction.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from indicators import FEATURE_COLS


def train_model(df: pd.DataFrame):
    """
    Train XGBoost model
    """

    X = df[FEATURE_COLS].values
    y = df["Target"].values

    # Time-series split
    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    # Scale features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Accuracy
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return (
        model,
        scaler,
        accuracy,
        FEATURE_COLS
    )


def predict(
    model,
    scaler,
    df: pd.DataFrame
) -> dict:
    """
    Predict latest stock movement
    """

    latest = df[FEATURE_COLS].iloc[-1:].values

    latest_scaled = scaler.transform(latest)

    pred = model.predict(latest_scaled)[0]

    probabilities = model.predict_proba(
        latest_scaled
    )[0]

    return {
        "prediction": int(pred),
        "direction": "UP" if pred == 1 else "DOWN",
        "confidence": float(max(probabilities)),
        "prob_up": float(probabilities[1]),
        "prob_down": float(probabilities[0]),
    }


def get_model_metrics(
    model,
    scaler,
    df: pd.DataFrame
):
    """
    Return model metrics
    """

    split_idx = int(len(df) * 0.8)

    X_test = df[FEATURE_COLS].iloc[split_idx:].values
    y_test = df["Target"].iloc[split_idx:].values

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    feature_importances = dict(
        zip(
            FEATURE_COLS,
            model.feature_importances_
        )
    )

    return {
        "accuracy": round(acc * 100, 2),
        "feature_importances": feature_importances,
        "test_samples": len(y_test),
    }


if __name__ == "__main__":

    from data_fetch import fetch_stock_data
    from indicators import add_indicators

    df = fetch_stock_data("INFY")

    df = add_indicators(df)

    model, scaler, acc, feats = train_model(df)

    print(f"Accuracy: {acc:.2%}")

    result = predict(model, scaler, df)

    print(result)