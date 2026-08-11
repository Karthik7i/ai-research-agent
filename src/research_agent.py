from src.tools.prediction_tool import PredictionTool


class ResearchAgent:
    def __init__(self, prediction_tool: PredictionTool):
        self.prediction_tool = prediction_tool

    def research(self, symbol: str) -> dict:
        prediction = self.prediction_tool.run(symbol)

        return {
            "symbol": symbol,
            "prediction": prediction,
        }