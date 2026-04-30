"""Project configuration for the multi-asset factor engineering pipeline."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

START_DATE = "2018-01-01"

YFINANCE_TICKERS = {
    "GC=F": "gold_us",
    "USO": "oil",
    #"CL=F": "oil",
    "BTC-USD": "btc",
    "QQQ": "nasdaq",
    "TLT": "treasury",
    "^VIX": "vix_level",
}

FRED_SERIES = ["DGS10", "DGS2", "DFF"]
