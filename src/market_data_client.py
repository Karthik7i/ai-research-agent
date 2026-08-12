from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import requests

from src.config import get_api_key


class MarketDataClient(ABC):
    @abstractmethod
    def get_daily_ohlcv(self, symbol: str) -> List[Dict[str, Any]]:
        """Return daily OHLCV records for one symbol."""


class AlphaVantageMarketDataClient(MarketDataClient):
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, session=requests, timeout: int = 30):
        self.api_key = api_key or get_api_key()
        self.session = session
        self.timeout = timeout

    def get_daily_ohlcv(self, symbol: str) -> List[Dict[str, Any]]:
        normalized_symbol = symbol.strip().upper()

        if not normalized_symbol:
            raise ValueError("Symbol is required")
        if not self.api_key:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY is not configured")

        response = self.session.get(
            self.BASE_URL,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": normalized_symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, Mapping):
            raise RuntimeError("Alpha Vantage response was not a JSON object")
        if "Error Message" in payload or "Note" in payload:
            raise RuntimeError(payload.get("Error Message") or payload["Note"])

        daily_series = payload.get("Time Series (Daily)")
        if not isinstance(daily_series, dict):
            raise RuntimeError("Alpha Vantage response did not contain daily market data")

        records = []
        for record_date, values in daily_series.items():
            if not isinstance(values, Mapping):
                raise RuntimeError(
                    f"Alpha Vantage daily record for {record_date} was not a JSON object"
                )

            records.append({
                "symbol": normalized_symbol,
                "date": record_date,
                "open_price": values.get("1. open"),
                "high_price": values.get("2. high"),
                "low_price": values.get("3. low"),
                "close_price": values.get("4. close"),
                "volume": values.get("5. volume"),
            })

        return records
