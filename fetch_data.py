import os
from datetime import datetime, timedelta
import pandas as pd

from db import get_conn, insert_bars
from polygon_client import PolygonTradingClient
from symbols import DAILY_BAR_SYMBOLS


def fetch_daily_bars(tickers, start_date, end_date):
    """
    Fetch daily trading bars for a list of tickers within a date range.
    """
    client = PolygonTradingClient(api_key=os.getenv("POLYGON_API_KEY"))

    all_data = []

    for ticker in tickers:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")
        try:
            bars = client.get_daily_bars(ticker, start_date, end_date)
            all_data.extend(bars)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return all_data


def clean_bars(rows):
    """
    Basic cleaning: drop duplicates and ensure required columns exist.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            df[col] = None
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.drop_duplicates(subset=["ticker", "date"])
    df = df.sort_values(["ticker", "date"])
    return df.to_dict(orient="records")


def main():
    """
    Fetch daily bars for predefined symbols and store them in SQLite.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    data = fetch_daily_bars(DAILY_BAR_SYMBOLS, start_date, end_date)
    cleaned = clean_bars(data)

    with get_conn() as conn:
        inserted = insert_bars(conn, cleaned)

    print(f"Total bars fetched: {len(data)}")
    print(f"Cleaned and inserted into database: {inserted} rows")

    return cleaned

if __name__ == "__main__":
    main()