import pandas as pd


FEATURE_COLUMNS = [
    "open_price",
    "high_price",
    "low_price",
    "volume",
    "daily_return",
    "ma_5",
    "ma_20",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["daily_return"] = df["close_price"].pct_change()

    df["ma_5"] = (
        df["close_price"]
        .rolling(window=5)
        .mean()
    )

    df["ma_20"] = (
        df["close_price"]
        .rolling(window=20)
        .mean()
    )

    return df.dropna()


def create_training_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]

    # Target: next day's closing price
    y = df["close_price"].shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]

    return X, y