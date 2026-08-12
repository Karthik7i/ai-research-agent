import argparse
from typing import Any, Dict, Iterable, List

from src.database import get_connection
from src.data_quality import validate_market_record
from src.market_data_client import AlphaVantageMarketDataClient, MarketDataClient


UPSERT_MARKET_DATA_SQL = """
    INSERT INTO public.market_data (
        symbol, date, open_price, high_price, low_price, close_price, volume
    )
    VALUES (%(symbol)s, %(date)s, %(open_price)s, %(high_price)s, %(low_price)s,
            %(close_price)s, %(volume)s)
    ON CONFLICT (symbol, date) DO UPDATE SET
        open_price = EXCLUDED.open_price,
        high_price = EXCLUDED.high_price,
        low_price = EXCLUDED.low_price,
        close_price = EXCLUDED.close_price,
        volume = EXCLUDED.volume
    RETURNING (xmax = 0) AS inserted;
"""


def _normalize_symbol(symbol: str) -> str:
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("Symbol is required")
    return normalized_symbol


def _valid_records(records: Iterable[Any], symbol: str) -> tuple[List[Dict[str, Any]], int]:
    valid_by_key = {}
    skipped = 0

    for record in records:
        normalized_record, quality_report = validate_market_record(
            record,
            expected_symbol=symbol,
        )
        if quality_report.invalid:
            skipped += 1
            continue

        assert normalized_record is not None
        key = (normalized_record["symbol"], normalized_record["date"])
        if key in valid_by_key:
            skipped += 1
        valid_by_key[key] = normalized_record

    return list(valid_by_key.values()), skipped


def ingest_symbol(symbol: str, client: MarketDataClient) -> Dict[str, int]:
    """Fetch, validate, and upsert one symbol's daily market data."""
    normalized_symbol = _normalize_symbol(symbol)
    fetched_records = client.get_daily_ohlcv(normalized_symbol)
    records, skipped = _valid_records(fetched_records, normalized_symbol)

    counts = {
        "fetched": len(fetched_records),
        "valid": len(records),
        "inserted": 0,
        "updated": 0,
        "skipped": skipped,
    }
    if not records:
        return counts

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            for record in records:
                cursor.execute(UPSERT_MARKET_DATA_SQL, record)
                if cursor.fetchone()[0]:
                    counts["inserted"] += 1
                else:
                    counts["updated"] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return counts


def main():
    parser = argparse.ArgumentParser(description="Ingest daily market data for one symbol")
    parser.add_argument("symbol")
    args = parser.parse_args()

    result = ingest_symbol(args.symbol, AlphaVantageMarketDataClient())
    print(result)


if __name__ == "__main__":
    main()
