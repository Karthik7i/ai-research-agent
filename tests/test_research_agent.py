from src.research_agent import ResearchAgent


class FakePredictionTool:
    def run(self, symbol):
        return {
            "symbol": symbol,
            "predicted_price": 205.50
        }


def test_research_agent():
    tool = FakePredictionTool()
    agent = ResearchAgent(tool)

    result = agent.research("IBM")

    assert result["symbol"] == "IBM"
    assert result["prediction"]["predicted_price"] == 205.50