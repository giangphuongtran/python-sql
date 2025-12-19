"""
Export existing PostgreSQL daily_bars data into SQLite for the class project.
Requires PG connection env vars: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE.
"""
import os
import sqlite3

import pandas as pd
import psycopg2

from db import DB_PATH, init_schema


def export_daily_bars(sqlite_path: str = DB_PATH, table: str = "daily_bars") -> int:
    pg_conn = psycopg2.connect(
        host="localhost",
        port="5441",
        user="phuonggiang_pgt",
        password="pgt_secret",
        dbname="trading_db",
    )

    with pg_conn, pg_conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {table}")
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)

    with sqlite3.connect(sqlite_path) as sqlite_conn:
        init_schema(sqlite_conn)
        df.to_sql("daily_bars", sqlite_conn, if_exists="replace", index=False)

    return len(df)


if __name__ == "__main__":
    count = export_daily_bars()
    print(f"Exported {count} rows into {DB_PATH}")

