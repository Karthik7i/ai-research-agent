import os
from dotenv import load_dotenv

load_dotenv()


def get_database_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }


def get_api_key():
    return os.getenv("ALPHA_VANTAGE_API_KEY")


def get_model_version():
    return os.getenv("MODEL_VERSION", "v1")