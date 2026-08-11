from src.predictor import Predictor


class ResearchAgent:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def research(self, symbol: str) -> dict:
        prediction = self.predictor.predict_symbol(symbol)

        return {
            "symbol": symbol,
            "prediction": prediction,
        }