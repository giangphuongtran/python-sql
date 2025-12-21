# Final Project - Stock Clustering Analysis

## Project Description
This project performs stock clustering analysis using machine learning techniques to group stocks based on their risk-return characteristics, momentum patterns, volatility, and technical indicators. The analysis helps investors identify groups of stocks with similar behavioral patterns for informed investment decision-making.

The project includes:
- Data fetching from Polygon.io API
- Feature engineering with technical indicators
- Multiple clustering methods (KMeans, PAM, Hierarchical)
- Interactive Streamlit dashboard for visualization and analysis

## Dataset
**Data Source**: https://kaggle.com/datasets/ae2daadecaecac15b060f3f6eff4cef6e866766d2759af7cdb2ebe8cdc791b9d

**Note**: 
- Dataset is fetched directly from Polygon.io API and stored in a local SQLite database
- The database file (`sql.db`) contains daily bars (open, high, low, close, volume) for 50 major US stocks
- Data files are stored in the project root directory

## Setup
- Python 3.10+ recommended
- Install dependencies: `pip install -r requirements.txt`
- Environment variables:
  - `POLYGON_API_KEY` (required for fetching fresh data)
  - Optional: `SQLITE_DB_PATH` (defaults to `sql.db`)

## How to Run

### 1. Download the dataset in the Kaggle link above

### 2. Place the data files in the project folder, same directory with the streamlit.py file

### 3. Download all the dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Streamlit Dashboard
```bash
streamlit run streamlit.py
```
- Input: SQLite database path (default: `sql.db`)
- Features:
  - Dataset overview and feature exploration
  - Correlation matrix and PCA analysis
  - Multiple clustering methods (KMeans, PAM, Hierarchical)
  - Cluster visualization with PCA
  - Standardized cluster means heatmap
  - Radar charts for cluster profiles
  - Silhouette analysis
  - Comparison of all clustering methods
  - Interactive candlestick charts with technical indicators

## File Map
- `fetch_data.py` — Fetch Polygon daily bars and store in SQLite
- `features.py` — Build feature set from `daily_bars` (technical indicators and aggregated metrics)
- `streamlit.py` — Interactive dashboard for clustering analysis and visualization
- `symbols.py` — List of stock tickers to fetch
- `polygon_client.py` — Polygon.io API client
- `db.py` — Database schema and operations
- `requirements.txt` — Python dependencies
- `sql.db` — SQLite database (created after running fetch_data.py)