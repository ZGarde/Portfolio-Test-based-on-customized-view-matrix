"""Download raw market and macro data for the factor project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import FRED_SERIES, RAW_DATA_DIR, START_DATE, YFINANCE_TICKERS


def download_yfinance_prices() -> pd.DataFrame:
    """Download adjusted close prices from yfinance and save them as CSV."""
    try:
        import yfinance as yf

        print("Downloading yfinance prices...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        tickers = list(YFINANCE_TICKERS.keys())
        raw = yf.download(
            tickers=tickers,
            start=START_DATE,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )

        if raw.empty:
            raise ValueError("yfinance returned an empty DataFrame.")

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = tickers

        prices = prices.rename(columns=YFINANCE_TICKERS)
        prices.index.name = "date"
        prices = prices.sort_index()
        output_path = RAW_DATA_DIR / "us_asset_prices.csv"
        prices.to_csv(output_path)
        print(f"Saved yfinance prices to {output_path}")
        return prices
    except Exception as exc:
        print(f"Failed to download yfinance prices: {exc}", file=sys.stderr)
        raise


def download_fred_rates() -> pd.DataFrame:
    """Download FRED interest-rate series and save them as CSV."""
    try:
        from pandas_datareader import data as pdr

        print("Downloading FRED rates...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        rates = pdr.DataReader(FRED_SERIES, "fred", START_DATE)
        if rates.empty:
            raise ValueError("FRED returned an empty DataFrame.")

        rates.index.name = "date"
        rates = rates.sort_index()
        output_path = RAW_DATA_DIR / "fred_rates.csv"
        rates.to_csv(output_path)
        print(f"Saved FRED rates to {output_path}")
        return rates
    except Exception as exc:
        print(f"Failed to download FRED rates: {exc}", file=sys.stderr)
        raise


def download_china_gold_sge() -> pd.DataFrame:
    """Download China SGE gold benchmark data from akshare and save it as CSV."""
    try:
        import akshare as ak

        print("Downloading China SGE gold benchmark data...")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw = ak.spot_golden_benchmark_sge()
        if raw.empty:
            raise ValueError("akshare returned an empty DataFrame.")

        df = raw.copy()
        date_col, morning_col, evening_col = _resolve_china_gold_columns(df)

        if date_col is None:
            raise ValueError(f"Cannot find date column in akshare output: {list(df.columns)}")
        if morning_col is None or evening_col is None:
            raise ValueError(
                "Cannot find morning/evening gold columns in akshare output: "
                f"{list(df.columns)}"
            )

        df = df[[date_col, morning_col, evening_col]].rename(
            columns={
                date_col: "date",
                morning_col: "china_gold_morning",
                evening_col: "china_gold_evening",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["china_gold_morning"] = pd.to_numeric(df["china_gold_morning"], errors="coerce")
        df["china_gold_evening"] = pd.to_numeric(df["china_gold_evening"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.set_index("date")
        output_path = RAW_DATA_DIR / "china_gold_sge.csv"
        df.to_csv(output_path)
        print(f"Saved China gold data to {output_path}")
        return df
    except Exception as exc:
        print(f"Failed to download China SGE gold data: {exc}", file=sys.stderr)
        raise


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first column whose lowercase name contains one candidate string."""
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for lower_name, original_name in lowered.items():
            if candidate_lower in lower_name:
                return original_name
    return None


def _resolve_china_gold_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Resolve akshare China gold date, morning price, and evening price columns."""
    exact_matches = {
        "date": ["交易时间", "日期", "交易日", "date"],
        "morning": ["早盘价", "早盘", "morning_price", "morning"],
        "evening": ["晚盘价", "晚盘", "evening_price", "evening"],
    }

    date_col = _find_column(df, exact_matches["date"])
    morning_col = _find_column(df, exact_matches["morning"])
    evening_col = _find_column(df, exact_matches["evening"])

    # akshare.spot_golden_benchmark_sge() commonly returns:
    # ["交易时间", "晚盘价", "早盘价"]. Keep this fallback simple and explicit.
    if date_col is None and len(df.columns) >= 1:
        date_col = df.columns[0]
    if evening_col is None and len(df.columns) >= 2:
        evening_col = df.columns[1]
    if morning_col is None and len(df.columns) >= 3:
        morning_col = df.columns[2]

    return date_col, morning_col, evening_col


def main() -> None:
    """Run all raw data download steps."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_yfinance_prices()
    download_fred_rates()
    download_china_gold_sge()


if __name__ == "__main__":
    main()
