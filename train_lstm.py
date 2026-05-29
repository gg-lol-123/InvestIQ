"""
train_lstm.py
Train LSTM model for stock direction prediction.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_fetch import fetch_stock_data
from indicators import add_indicators, FEATURE_COLS


# -------------------------------
# SETTINGS
# -------------------------------

SEQUENCE_LENGTH = 20

# -------------------------------
# LOAD DATA
# -------------------------------

print("Fetching stock data...")

df = fetch_stock_data(
    "RELIANCE",
    period="2y"
)

df = add_indicators(df)

# -------------------------------
# FEATURES / TARGET
# -------------------------------

X = df[FEATURE_COLS].values
y = df["Target"].values

# -------------------------------
# SCALE FEATURES
# -------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------
# CREATE SEQUENCES
# -------------------------------

X_seq = []
y_seq = []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):

    X_seq.append(
        X_scaled[i - SEQUENCE_LENGTH:i]
    )

    y_seq.append(
        y[i]
    )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

split_idx = int(
    len(X_seq) * 0.8
)

X_train = X_seq[:split_idx]
X_test = X_seq[split_idx:]

y_train = y_seq[:split_idx]
y_test = y_seq[split_idx:]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# -------------------------------
# BUILD MODEL
# -------------------------------

model = Sequential([
    LSTM(
        64,
        input_shape=(
            SEQUENCE_LENGTH,
            len(FEATURE_COLS)
        )
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# TRAIN
# -------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=25,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# EVALUATE
# -------------------------------

probs = model.predict(
    X_test,
    verbose=0
)

preds = (
    probs > 0.5
).astype(int)

accuracy = accuracy_score(
    y_test,
    preds
)

print()
print("=" * 50)
print("LSTM Accuracy:", round(
    accuracy * 100,
    2
), "%")
print("=" * 50)

# -------------------------------
# SAVE
# -------------------------------

model.save(
    "lstm_model.keras"
)

joblib.dump(
    scaler,
    "lstm_scaler.pkl"
)

print()
print("Saved:")
print(" - lstm_model.keras")
print(" - lstm_scaler.pkl")