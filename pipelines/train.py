from src.data_loader import load_market_data
from src.features import create_features, create_training_data
from src.model import (
    split_time_series,
    train_model,
    evaluate_model,
    save_model,
)

def main():
    # Load data
    df = load_market_data()

    # Feature engineering
    df = create_features(df)
    X, y = create_training_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = split_time_series(X, y)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Train model
    model = train_model(X_train, y_train)

    print("Model trained!")

    # Evaluate model FIRST
    results = evaluate_model(model, X_test, y_test)

    print("MAE:", results["mae"])
    print("RMSE:", results["rmse"])

    # Save model AFTER evaluation
    save_model(model, results)

    print("Model saved!")


if __name__ == "__main__":
    main()