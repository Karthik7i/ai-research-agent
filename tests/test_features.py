import pandas as pd

from src.features import create_features, create_training_data


def sample_data():
    return pd.DataFrame({
        "symbol": ["IBM"] * 25,
        "date": pd.date_range("2026-01-01", periods=25),
        "open_price": range(100, 125),
        "high_price": range(101, 126),
        "low_price": range(99, 124),
        "close_price": range(100, 125),
        "volume": range(1000, 1025),
    })


def test_create_features():
    df = sample_data()

    result = create_features(df)

    assert "daily_return" in result.columns
    assert "ma_5" in result.columns
    assert "ma_20" in result.columns


def test_create_training_data():
    df = sample_data()

    df = create_features(df)

    X, y = create_training_data(df)

    assert len(X) == len(y)
    assert X.shape[1] == 7