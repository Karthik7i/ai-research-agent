from src.tools.prediction_tool import PredictionTool


class FakePredictor:
    def predict_symbol(self, symbol):
        return {
            "symbol": symbol,
            "predicted_price": 205.50
        }


def test_prediction_tool():
    predictor = FakePredictor()
    tool = PredictionTool(predictor)

    result = tool.run("IBM")

    assert result["symbol"] == "IBM"
    assert result["predicted_price"] == 205.50