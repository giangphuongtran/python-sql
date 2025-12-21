import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Mapping, Optional

@contextmanager
def get_conn(db_path: Optional[str] = None):
    """
    Context manager that yields a SQLite connection and commits on success.
    """
    path = db_path
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create the daily_bars table if it does not exist.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            transactions REAL,
            volume_weighted_avg_price REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )


def insert_bars(conn: sqlite3.Connection, rows: Iterable[Mapping]) -> int:
    """
    Insert an iterable of dict-like daily bar rows into SQLite.
    """
    rows = list(rows)
    if not rows:
        return 0
    init_schema(conn)
    conn.executemany(
        """
        INSERT OR REPLACE INTO daily_bars (
            ticker, date, open, high, low, close, volume, transactions, volume_weighted_avg_price
        ) VALUES (
            :ticker, :date, :open, :high, :low, :close, :volume, :transactions, :volume_weighted_avg_price
        )
        """,
        rows,
    )
    return len(rows)

