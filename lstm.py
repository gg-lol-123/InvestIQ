"""
lstm.py

Load saved LSTM model and perform prediction.
"""

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from indicators import FEATURE_COLS


SEQUENCE_LENGTH = 20


def load_lstm_model():
    """
    Load saved LSTM model and scaler.
    """

    model = load_model(
        "lstm_model.keras"
    )

    scaler = joblib.load(
        "lstm_scaler.pkl"
    )

    return model, scaler


def predict_lstm(
    model,
    scaler,
    df: pd.DataFrame
):
    """
    Predict latest stock direction.
    """

    X = df[FEATURE_COLS].values

    X_scaled = scaler.transform(X)

    if len(X_scaled) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} rows"
        )

    latest_sequence = X_scaled[
        -SEQUENCE_LENGTH:
    ]

    latest_sequence = np.expand_dims(
        latest_sequence,
        axis=0
    )

    prob_up = float(
        model.predict(
            latest_sequence,
            verbose=0
        )[0][0]
    )

    prob_down = 1.0 - prob_up

    prediction = (
        1 if prob_up >= 0.5 else 0
    )

    direction = (
        "UP"
        if prediction == 1
        else "DOWN"
    )

    confidence = max(
        prob_up,
        prob_down
    )

    return {
        "prediction": prediction,
        "direction": direction,
        "confidence": confidence,
        "prob_up": prob_up,
        "prob_down": prob_down,
    }


def get_lstm_metrics():
    """
    Temporary metrics function.

    Replace later with proper
    evaluation if needed.
    """

    return {
        "accuracy": 55.29
    }


if __name__ == "__main__":

    from data_fetch import fetch_stock_data
    from indicators import add_indicators

    print("Loading model...")

    model, scaler = load_lstm_model()

    print("Fetching data...")

    df = fetch_stock_data(
        "RELIANCE",
        period="2y"
    )

    df = add_indicators(df)

    result = predict_lstm(
        model,
        scaler,
        df
    )

    print()
    print("LSTM Prediction")
    print(result)