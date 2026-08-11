from src.predictor import Predictor


class PredictionTool:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    def run(self, symbol: str) -> dict:
        return self.predictor.predict_symbol(symbol)