# Final Project (SQLite + Streamlit)

Minimal pipeline for class submission (no Docker). Everything here lives in this folder; **do not use `app/`**.

## Setup
- Python 3.10+ recommended.
- `pip install -r requirements.txt`
- Env vars:
  - `POLYGON_API_KEY` (for fetching fresh data)
  - Optional: `SQLITE_DB_PATH` (defaults to `sql.db`)
  - Optional (for migration): `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

## 1) Get data into SQLite
- Fresh from Polygon (free tier pace):
  - `python fetch_data.py`
  - Writes daily bars for the tickers in `symbols.py` into `sql.db`.
- Or migrate existing Postgres data:
  - `python export_postgres_to_sqlite.py`

## 2) Feature engineering for clustering
- `python -c "import features; print(features.compute_features().head())"`
- Features produced (per ticker):
  - Risk/Return: `mean_daily_return`, `mean_stock_vs_market`, `beta_global`, `mean_sharpe_20d`, `worst_drawdown`
  - Momentum: `mean_momentum_20d`, `mean_close_vs_sma200`, `mean_adx_14`, `mean_price_pos_20`
  - Volatility: `mean_volatility_20d`, `mean_atr_14`, `mean_volatility_ratio`, `mean_bb_width`
  - Volume/Liquidity: `mean_liquidity_20d`, `mean_volume_ratio`
  - Technical: `mean_rsi_14`, `mean_macd_hist`, `mean_stoch_k`
  - Distributional: `return_skewness`, `return_kurtosis`

## 3) Hierarchical clustering (cosine, average linkage)
- Console run: `python clustering.py` (prints cluster assignments).
- Internals: standardizes feature matrix, computes cosine distances, average linkage dendrogram, cluster labels via distance threshold (default 0.5).

## 4) Streamlit dashboard
- `streamlit run streamlit_app.py`
- Inputs: `SQLite path` (default `sql.db`), `Cluster distance threshold`.
- Shows feature sample, cluster assignments, Sharpe table, candlesticks, and dendrogram.

## One-shot command list (for professor)
1) `cd python-sql/final-project`
2) `pip install -r requirements.txt`
3) `POLYGON_API_KEY=... python fetch_data.py`   # or `python export_postgres_to_sqlite.py`
4) `python clustering.py`
5) `streamlit run streamlit_app.py`

## File map
- `fetch_data.py` — pull Polygon daily bars -> SQLite.
- `export_postgres_to_sqlite.py` — migrate existing Postgres `daily_bars` -> SQLite.
- `features.py` — build feature set listed above from `daily_bars`.
- `clustering.py` — cosine/average linkage clustering on features.
- `streamlit_app.py` — minimal UI to view clusters and dendrogram.
- `symbols.py` — tickers to fetch.
- `requirements.txt` — deps.

## Typical flow
1) `python fetch_data.py` (or `python export_postgres_to_sqlite.py`)
2) `python clustering.py` (quick check)  
3) `streamlit run streamlit_app.py` (present)  

