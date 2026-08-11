import pandas as pd

from src.model import (
    split_time_series,
    train_model,
    evaluate_model,
)


def sample_training_data():
    X = pd.DataFrame({
        "open_price": range(100, 120),
        "high_price": range(101, 121),
        "low_price": range(99, 119),
        "volume": range(1000, 1020),
        "daily_return": [0.01] * 20,
        "ma_5": range(100, 120),
        "ma_20": range(100, 120),
    })

    y = pd.Series(range(101, 121))

    return X, y


def test_split_time_series():
    X, y = sample_training_data()

    X_train, X_test, y_train, y_test = split_time_series(X, y)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_train_model():
    X, y = sample_training_data()

    X_train, X_test, y_train, y_test = split_time_series(X, y)

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)


def test_evaluate_model():
    X, y = sample_training_data()

    X_train, X_test, y_train, y_test = split_time_series(X, y)

    model = train_model(X_train, y_train)

    results = evaluate_model(model, X_test, y_test)

    assert "mae" in results
    assert "rmse" in results
    assert "predictions" in results

    assert results["mae"] >= 0
    assert results["rmse"] >= 0