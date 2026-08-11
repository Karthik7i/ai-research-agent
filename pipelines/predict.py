from src.data_loader import load_market_data
from src.features import create_features, FEATURE_COLUMNS
from src.model import load_model


def main():
    # Load historical data
    df = load_market_data()

    # Create the same features used during training
    df = create_features(df)

    # Get the most recent observation
    latest = df.iloc[[-1]]

    # Select model features
    X_latest = latest[FEATURE_COLUMNS]

    # Load trained model
    model = load_model()

    # Make prediction
    prediction = model.predict(X_latest)[0]

    print("Latest date:", latest["date"].iloc[0])
    print("Latest closing price:", latest["close_price"].iloc[0])
    print("Predicted next closing price:", prediction)


if __name__ == "__main__":
    main()