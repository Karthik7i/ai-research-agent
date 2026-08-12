import pandas as pd
import pytest

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


def multiple_symbols_data():
    dates = pd.date_range("2026-01-01", periods=25)

    alpha = pd.DataFrame({
        "symbol": ["ALPHA"] * 25,
        "date": dates,
        "open_price": range(10, 35),
        "high_price": range(11, 36),
        "low_price": range(9, 34),
        "close_price": range(10, 35),
        "volume": range(1000, 1025),
    })
    beta = pd.DataFrame({
        "symbol": ["BETA"] * 25,
        "date": dates,
        "open_price": range(1000, 1250, 10),
        "high_price": range(1001, 1251, 10),
        "low_price": range(999, 1249, 10),
        "close_price": range(1000, 1250, 10),
        "volume": range(2000, 2025),
    })

    return pd.concat([beta, alpha]).sample(frac=1, random_state=42).reset_index(drop=True)


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


def test_create_features_calculates_each_symbol_independently():
    result = create_features(multiple_symbols_data())

    assert result.groupby("symbol").size().to_dict() == {"ALPHA": 6, "BETA": 6}
    assert result.groupby("symbol")["date"].apply(lambda dates: dates.is_monotonic_increasing).all()

    first_alpha = result[result["symbol"] == "ALPHA"].iloc[0]
    first_beta = result[result["symbol"] == "BETA"].iloc[0]

    assert first_alpha["ma_20"] == 19.5
    assert first_beta["ma_20"] == 1095.0
    assert first_alpha["daily_return"] == pytest.approx(1 / 28)
    assert first_beta["daily_return"] == pytest.approx(10 / 1180)


def test_create_training_data_never_uses_another_symbols_next_close():
    features = create_features(multiple_symbols_data())

    X, y = create_training_data(features)
    targets = pd.DataFrame({
        "symbol": features.loc[y.index, "symbol"],
        "target": y,
    })

    assert X.index.equals(y.index)
    assert targets.groupby("symbol")["target"].apply(list).to_dict() == {
        "ALPHA": [30, 31, 32, 33, 34],
        "BETA": [1200, 1210, 1220, 1230, 1240],
    }
