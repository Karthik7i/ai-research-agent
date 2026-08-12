from copy import deepcopy
from datetime import date, timedelta

import pytest

from pipelines.ingest import ingest_symbol
from src.data_quality import validate_market_record
from src.market_data_client import AlphaVantageMarketDataClient


class FakeMarketDataClient:
    def __init__(self, records=None, error=None):
        self.records = records or []
        self.error = error
        self.requested_symbols = []

    def get_daily_ohlcv(self, symbol):
        self.requested_symbols.append(symbol)
        if self.error:
            raise self.error
        return deepcopy(self.records)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, parameters):
        self.connection.queries.append((query, parameters))
        key = (parameters["symbol"], parameters["date"])
        inserted = key not in self.connection.rows
        self.connection.rows[key] = dict(parameters)
        self.result = (inserted,)

    def fetchone(self):
        return self.result


class FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.queries = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def market_record(**overrides):
    record = {
        "symbol": "aapl",
        "date": "2026-08-09",
        "open_price": "220.00",
        "high_price": "225.00",
        "low_price": "218.00",
        "close_price": "223.50",
        "volume": "1000000",
    }
    record.update(overrides)
    return record


def test_ingest_symbol_normalizes_symbol_and_inserts_records(monkeypatch):
    connection = FakeConnection()
    client = FakeMarketDataClient([market_record()])
    monkeypatch.setattr("pipelines.ingest.get_connection", lambda: connection)

    result = ingest_symbol(" aapl ", client)

    assert client.requested_symbols == ["AAPL"]
    assert result == {"fetched": 1, "valid": 1, "inserted": 1, "updated": 0, "skipped": 0}
    assert connection.rows[("AAPL", date(2026, 8, 9))]["close_price"] == 223.5
    assert connection.commits == 1
    assert connection.closed


def test_repeated_ingestion_updates_existing_row(monkeypatch):
    connection = FakeConnection()
    client = FakeMarketDataClient([market_record()])
    monkeypatch.setattr("pipelines.ingest.get_connection", lambda: connection)

    first_result = ingest_symbol("AAPL", client)
    second_result = ingest_symbol("AAPL", client)

    assert first_result == {"fetched": 1, "valid": 1, "inserted": 1, "updated": 0, "skipped": 0}
    assert second_result == {"fetched": 1, "valid": 1, "inserted": 0, "updated": 1, "skipped": 0}
    assert len(connection.rows) == 1
    assert connection.commits == 2


def test_ingestion_updates_values_for_existing_date(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr("pipelines.ingest.get_connection", lambda: connection)

    ingest_symbol("AAPL", FakeMarketDataClient([market_record(close_price="223.50")]))
    result = ingest_symbol("AAPL", FakeMarketDataClient([market_record(close_price="224.75")]))

    assert result == {"fetched": 1, "valid": 1, "inserted": 0, "updated": 1, "skipped": 0}
    assert connection.rows[("AAPL", date(2026, 8, 9))]["close_price"] == 224.75


def test_ingestion_skips_invalid_or_wrong_symbol_records(monkeypatch):
    invalid_records = [
        market_record(volume=None),
        market_record(date="not-a-date"),
        market_record(symbol="MSFT"),
        market_record(close_price="not-a-number"),
        market_record(volume="1.5"),
        market_record(volume="-1"),
        market_record(high_price="217.00"),
        market_record(open_price="226.00"),
        market_record(close_price="217.00"),
        market_record(date=date.today() + timedelta(days=1)),
    ]
    client = FakeMarketDataClient(invalid_records)
    monkeypatch.setattr(
        "pipelines.ingest.get_connection",
        lambda: pytest.fail("No database connection should be opened for invalid records"),
    )

    result = ingest_symbol("AAPL", client)

    assert result == {"fetched": 10, "valid": 0, "inserted": 0, "updated": 0, "skipped": 10}
    assert all(
        report.invalid == 1
        for _, report in (validate_market_record(record, expected_symbol="AAPL") for record in invalid_records)
    )


def test_ingestion_uses_canonical_validator_issue_codes(monkeypatch):
    connection = FakeConnection()
    client = FakeMarketDataClient([
        market_record(symbol="MSFT"),
        market_record(volume="1.5"),
        market_record(),
    ])
    original_validator = validate_market_record
    observed_issue_codes = []

    def tracking_validator(record, expected_symbol=None):
        normalized_record, report = original_validator(record, expected_symbol)
        observed_issue_codes.extend(issue.code for issue in report.issues)
        return normalized_record, report

    monkeypatch.setattr("pipelines.ingest.get_connection", lambda: connection)
    monkeypatch.setattr("pipelines.ingest.validate_market_record", tracking_validator)

    result = ingest_symbol("AAPL", client)

    assert result == {"fetched": 3, "valid": 1, "inserted": 1, "updated": 0, "skipped": 2}
    assert {"unexpected_symbol", "non_integer_volume"} <= set(observed_issue_codes)


def test_ingestion_skips_duplicate_records_within_a_batch(monkeypatch):
    connection = FakeConnection()
    client = FakeMarketDataClient([
        market_record(close_price="223.50"),
        market_record(close_price="224.75"),
    ])
    monkeypatch.setattr("pipelines.ingest.get_connection", lambda: connection)

    result = ingest_symbol("AAPL", client)

    assert result == {"fetched": 2, "valid": 1, "inserted": 1, "updated": 0, "skipped": 1}
    assert connection.rows[("AAPL", date(2026, 8, 9))]["close_price"] == 224.75


def test_ingestion_propagates_client_failure_without_opening_database(monkeypatch):
    client = FakeMarketDataClient(error=RuntimeError("Alpha Vantage unavailable"))
    monkeypatch.setattr(
        "pipelines.ingest.get_connection",
        lambda: pytest.fail("Database should not be opened when the client fails"),
    )

    with pytest.raises(RuntimeError, match="Alpha Vantage unavailable"):
        ingest_symbol("AAPL", client)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


def test_alpha_vantage_client_translates_daily_response_without_network():
    session = FakeSession({
        "Time Series (Daily)": {
            "2026-08-09": {
                "1. open": "220.00",
                "2. high": "225.00",
                "3. low": "218.00",
                "4. close": "223.50",
                "5. volume": "1000000",
            }
        }
    })

    records = AlphaVantageMarketDataClient(api_key="test-key", session=session).get_daily_ohlcv(" aapl ")

    assert session.calls[0][1]["symbol"] == "AAPL"
    assert records == [market_record(symbol="AAPL")]


def test_alpha_vantage_client_rejects_malformed_daily_records_without_network():
    session = FakeSession({
        "Time Series (Daily)": {
            "2026-08-09": "not-a-record",
        }
    })

    client = AlphaVantageMarketDataClient(api_key="test-key", session=session)

    with pytest.raises(RuntimeError, match="2026-08-09 was not a JSON object"):
        client.get_daily_ohlcv("AAPL")
