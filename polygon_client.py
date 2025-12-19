import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from polygon import RESTClient
from requests.exceptions import (
    HTTPError, RetryError, ConnectionError, ReadTimeout, ConnectTimeout)

class RateLimitError(Exception):
    """Custom exception for rate limit errors."""
    pass

class PolygonTradingClient:
    """
    Polygon API client to get daily trading data.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_delay: float = 20.0,
        max_attempts: int = 5):
        """
        Initialize Polygon API client with rate limiting and retry logic.
        """
        
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or through the POLYGON_API_KEY environment variable.")
        
        self.client = RESTClient(self.api_key)
        self.rate_limit_delay = float(rate_limit_delay)
        self.max_attempts = int(max_attempts)
        self._last_call = 0.0
        
    def _pace(self) -> None:
        """
        Enforce minimum delay between API calls.
        """
        elapsed = time.time() - self._last_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_call = time.time()
        
    @staticmethod
    def _retry_backoff(attempt: int, base: float) -> float:
        return base * (2 ** attempt)
    
    def _handle_api_errors(self, e: Exception, ticker: str, attempt: int) -> None:
        """
        Handle API errors based on response status codes.
        """
        if isinstance(e, HTTPError):
            status = getattr(e.response, "status_code", None)
            if status == 429:  # Rate limit exceeded
                if attempt > self.max_attempts:
                    raise RateLimitError(f"Rate limit exceeded after {self.max_attempts} attempts for {ticker}") from e
                retry_after = e.response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._retry_backoff(attempt, self.rate_limit_delay)
                print(f"Rate limit (429) for {ticker}; sleeping {delay:.1f}s before retry #{attempt}")
                return True, delay
            raise
        raise
    
    def _retry_api_call(self, callable_func, ticker: str):
        """
        Retry wrapper for API calls with error handling.
        """
        attempt = 1
        while attempt <= self.max_attempts:
            try:
                self._pace() # Rate limiting
                return callable_func()
            except (HTTPError, RetryError, ConnectionError, ReadTimeout, ConnectionTimeout) as e:
                should_retry, delay = self._handle_api_errors(e, ticker, attempt)
                if not should_retry:
                    raise
                time.sleep(delay)
                attempt += 1
                continue
            
    def get_daily_bars(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        *,
        multiplier: int = 1,
        timespan: str = "day"
    ) -> List[Dict]:
        """
        Fetch daily trading bars for a given ticker and date range.
        """
        
        def _fetch_bars():
            it = self.client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                adjusted=True,
                sort="asc",
                limit=5000
            )
            rows: List[Dict] = []
            for agg in it:
                ts = datetime.fromtimestamp(agg.timestamp / 1000)
                row = {
                    "ticker": ticker,
                    "timestamp": ts.strftime("%Y-%m-%d"),
                    "date": ts.date(),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                    "transactions": getattr(agg, "transactions", None),
                    "volume_weighted_avg_price": getattr(agg, "vwap", None),
                }
                rows.append(row)
            print(f"Fetched {len(rows)} bars for {ticker} from {start_date} to {end_date}")
            return rows
        return self._retry_api_call(_fetch_bars, ticker)