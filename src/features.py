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
    df = df.sort_values(["symbol", "date"])

    grouped_close = df.groupby("symbol")["close_price"]

    df["daily_return"] = grouped_close.pct_change()

    df["ma_5"] = grouped_close.transform(
        lambda prices: prices.rolling(window=5).mean()
    )

    df["ma_20"] = grouped_close.transform(
        lambda prices: prices.rolling(window=20).mean()
    )

    return df.dropna()


def create_training_data(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    # Target: next day's closing price
    y = df.groupby("symbol")["close_price"].shift(-1)
    valid_targets = y.notna()

    X = df.loc[valid_targets, FEATURE_COLUMNS]
    y = y.loc[valid_targets]

    return X, y
