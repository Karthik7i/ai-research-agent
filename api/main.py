from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model_service import ModelService
from src.predictor import Predictor


model_service = ModelService()
predictor = Predictor(model_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="AI Research Agent API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    symbol: str
    latest_date: str
    latest_close: float
    predicted_next_close: float


@app.get("/")
def root():
    return {
        "message": "AI Research Agent API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/predict/{symbol}", response_model=PredictionResponse)
def predict(symbol: str):

    try:
        return predictor.predict_symbol(symbol)

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )