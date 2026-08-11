import pandas as pd

from src.database import get_connection


def load_market_data():
    query = """
    SELECT
        symbol,
        date,
        open_price,
        high_price,
        low_price,
        close_price,
        volume
    FROM market_data
    ORDER BY date;
    """

    connection = get_connection()

    try:
        return pd.read_sql(query, connection)
    finally:
        connection.close()