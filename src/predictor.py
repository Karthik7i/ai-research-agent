from src.data_loader import load_market_data
from src.features import create_features, FEATURE_COLUMNS


class Predictor:

    def __init__(self, model_service):
        self.model_service = model_service

    def predict_symbol(self, symbol: str):

        df = load_market_data()

        symbol = symbol.upper()

        df = df[df["symbol"].str.upper() == symbol]

        if df.empty:
            raise ValueError(f"No market data found for {symbol}")

        df = create_features(df)

        latest = df.iloc[[-1]]

        X_latest = latest[FEATURE_COLUMNS]

        prediction = self.model_service.predict(X_latest)[0]

        return {
            "symbol": symbol,
            "latest_date": str(latest["date"].iloc[0].date()),
            "latest_close": float(latest["close_price"].iloc[0]),
            "predicted_next_close": float(prediction),
        }