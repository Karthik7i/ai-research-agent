from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import get_model_version
import numpy as np
from sklearn.pipeline import Pipeline
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib



def split_time_series(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    pipeline = Pipeline([
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {
        "mae": mae,
        "rmse": rmse,
        "predictions": predictions,
    }




MODEL_VERSION = get_model_version()
MODEL_DIR = Path("models") / MODEL_VERSION
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"


def save_model(model, metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "features": [
            "open_price",
            "high_price",
            "low_price",
            "volume",
            "daily_return",
            "ma_5",
            "ma_20",
        ],
        "metrics": {
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=2)


def load_model():
    return joblib.load(MODEL_PATH)



