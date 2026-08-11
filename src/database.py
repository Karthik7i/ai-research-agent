import psycopg

from src.config import get_database_config


def get_connection():
    config = get_database_config()
    return psycopg.connect(**config)