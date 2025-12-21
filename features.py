from __future__ import annotations

import sqlite3
from typing import Dict, Optional

import numpy as np
import pandas as pd

def _load_bars(db_path: Optional[str] = None) -> pd.DataFrame:
    path = db_path or "sql.db"
    with sqlite3.connect(path) as conn:
        df = pd.read_sql(
            "SELECT ticker, date, open, high, low, close, volume FROM daily_bars",
            conn,
            parse_dates=["date"],
        )
    if df.empty:
        raise ValueError("No data found in daily_bars table. Run fetch_data.py first.")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # enforce numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    plus_dm = (high.diff().clip(lower=0)).where((high.diff() > low.diff()), 0.0)
    minus_dm = (low.diff().clip(upper=0).abs()).where((low.diff() > high.diff()), 0.0)

    tr = _atr(high, low, close, window=1)
    atr = tr.rolling(window=window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    return dx.ewm(alpha=1 / window).mean()


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def _stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return 100 * (close - lowest_low) / denom


def _drawdown(returns: pd.Series) -> pd.Series:
    cumulative = (1 + returns.fillna(0)).cumprod()
    peak = cumulative.cummax()
    dd = cumulative / peak - 1
    return dd


def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for a single ticker's data.

    Input columns required: date, open, high, low, close, volume
    Output: original columns + indicator columns:
      RSI, ATR, ADX, MACD_Hist, Stoch_K, SMA20, SMA60, SMA200, BB_Upper, BB_Lower, Volume
    """
    df = data.copy().sort_values("date").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI
    df["RSI"] = _rsi(close, window=14)

    # ATR
    df["ATR"] = _atr(high, low, close, window=14)

    # ADX
    df["ADX"] = _adx(high, low, close, window=14)

    # MACD Histogram
    df["MACD_Hist"] = _macd_hist(close)

    # Stochastic K
    df["Stoch_K"] = _stoch_k(close, high, low, window=14)

    # SMAs
    df["SMA20"] = close.rolling(20).mean()
    df["SMA60"] = close.rolling(60).mean()
    df["SMA200"] = close.rolling(200).mean()

    # Bollinger Bands (20, 2-sigma)
    bb_mid = df["SMA20"]
    bb_std = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std

    # Volume
    df["Volume"] = volume

    return df

def compute_features(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compute all feature groups for clustering.
    """
    bars = _load_bars(db_path)

    # For market return
    closes = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = closes.pct_change()
    market_returns = returns.mean(axis=1)

    features: Dict[str, Dict[str, float]] = {}

    for ticker, g in bars.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)

        price = g["close"]
        high = g["high"]
        low = g["low"]
        vol = g["volume"]

        if price.dropna().shape[0] < 60:
            continue

        # 1) compute indicators
        gi = compute_technical_indicators(g)

        # 2) derived time-series used for aggregation
        ret = gi["close"].pct_change()
        mkt = market_returns.reindex(gi["date"]).reset_index(drop=True)

        # Momentum and rolling stats
        momentum_20d = gi["close"] / gi["close"].shift(20) - 1
        sharpe_20d = ret.rolling(20).mean() / ret.rolling(20).std()

        # beta against market
        mkt_var = mkt.var()
        beta_global = ret.cov(mkt) / mkt_var if (mkt_var and not np.isnan(mkt_var)) else np.nan

        std20 = ret.rolling(20).std()
        std60 = ret.rolling(60).std()

        # Bollinger width (reuse BB_Upper and BB_Lower from compute_technical_indicators)
        bb_width = (gi["BB_Upper"] - gi["BB_Lower"]) / gi["SMA20"]

        # risk/return
        mean_daily_return = ret.mean()
        mean_stock_vs_market = (ret - mkt).mean()
        mean_sharpe_20d = sharpe_20d.replace([np.inf, -np.inf], np.nan).mean()
        worst_drawdown = _drawdown(ret).min()

        # momentum
        mean_momentum_20d = momentum_20d.mean()
        mean_close_vs_sma200 = (gi["close"] / gi["SMA200"] - 1).mean()
        mean_adx_14 = gi["ADX"].mean()
        mean_price_pos_20 = (gi["close"] > gi["SMA20"]).mean()

        # volatility
        mean_volatility_20d = std20.mean()
        mean_atr_14 = gi["ATR"].mean()
        mean_volatility_ratio = (std20 / std60).replace([np.inf, -np.inf], np.nan).mean()
        mean_bb_width = bb_width.replace([np.inf, -np.inf], np.nan).mean()

        # volume/liquidity
        dollar_vol = gi["close"] * gi["volume"]
        mean_liquidity_20d = dollar_vol.rolling(20).mean().mean()
        mean_volume_ratio = (gi["volume"] / gi["volume"].rolling(60).mean()).replace([np.inf, -np.inf], np.nan).mean()

        # technical
        mean_rsi_14 = gi["RSI"].mean()
        mean_macd_hist = gi["MACD_Hist"].mean()
        mean_stoch_k = gi["Stoch_K"].mean()

        # distributional
        return_skewness = ret.skew()
        return_kurtosis = ret.kurtosis()

        features[ticker] = {
            "mean_daily_return": mean_daily_return,
            "mean_stock_vs_market": mean_stock_vs_market,
            "beta_global": beta_global,
            "mean_sharpe_20d": mean_sharpe_20d,
            "worst_drawdown": worst_drawdown,
            "mean_momentum_20d": mean_momentum_20d,
            "mean_close_vs_sma200": mean_close_vs_sma200,
            "mean_adx_14": mean_adx_14,
            "mean_price_pos_20": mean_price_pos_20,
            "mean_volatility_20d": mean_volatility_20d,
            "mean_atr_14": mean_atr_14,
            "mean_volatility_ratio": mean_volatility_ratio,
            "mean_bb_width": mean_bb_width,
            "mean_liquidity_20d": mean_liquidity_20d,
            "mean_volume_ratio": mean_volume_ratio,
            "mean_rsi_14": mean_rsi_14,
            "mean_macd_hist": mean_macd_hist,
            "mean_stoch_k": mean_stoch_k,
            "return_skewness": return_skewness,
            "return_kurtosis": return_kurtosis,
        }
    if not features:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(features, orient="index").dropna(how="all")
    
    # Remove zero-variance columns
    variances = df.var(axis=0, ddof=1)
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        
    return df