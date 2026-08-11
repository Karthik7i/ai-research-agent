from fastapi.testclient import TestClient

from api.main import app, predictor


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI Research Agent API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction(monkeypatch):

    def fake_prediction(symbol):
        return {
            "symbol": symbol.upper(),
            "latest_date": "2026-01-01",
            "latest_close": 200.0,
            "predicted_next_close": 205.0,
        }

    monkeypatch.setattr(
        predictor,
        "predict_symbol",
        fake_prediction,
    )

    response = client.get("/predict/IBM")

    assert response.status_code == 200

    data = response.json()

    assert data["symbol"] == "IBM"
    assert data["latest_close"] == 200.0
    assert data["predicted_next_close"] == 205.0