"""China/US market analysis research pipeline.

This script is intentionally separate from the main production factor pipeline.
It downloads China and US index data, FX/rate proxies, builds a processed
research panel, creates candidate market factors through ``factors.market_factors``,
and writes charts plus CSV outputs under dedicated market-analysis folders.

Run:
    python src/market_analysis.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from factors.common import build_simple_return_panel
from factors.market_factors import build_market_research_factors


warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

RAW_MARKET_DIR = RAW_DATA_DIR / "market_analysis"
PROCESSED_MARKET_DIR = PROCESSED_DATA_DIR / "market_analysis"
PERIOD_CORR_DIR = PROCESSED_MARKET_DIR / "period_correlation_matrices"
CHARTS_DIR = PROCESSED_MARKET_DIR / "charts"

START_DATE = "2010-01-01"
END_DATE = None
ROLLING_WINDOW = 60
PERIODS = {
    "2010_2014": ("2010-01-01", "2014-12-31"),
    "2015_2019": ("2015-01-01", "2019-12-31"),
    "2020_2022": ("2020-01-01", "2022-12-31"),
    "2023_now": ("2023-01-01", None),
}

US_INDICES = {
    "^GSPC": "sp500",
    "^IXIC": "nasdaq_composite",
    "^DJI": "dow_jones",
    "^RUT": "russell_2000",
    "^VIX": "vix",
}

CN_INDICES = {
    "000001.SS": "sse_index",
    "000016.SS": "sse_50",
    "000300.SS": "csi_300",
    "000905.SS": "csi_500",
    "000906.SS": "csi_800",
    "000688.SS": "star_50",
    "399006.SZ": "chinext",
    "399001.SZ": "szse_component",
}

FX_AND_RATE = {
    "USDCNY=X": "usd_cny",
    "CNH=X": "cnh_usd",
    "^TNX": "us_10y_yield",
}

DISPLAY_NAMES = {
    "sp500": "S&P 500",
    "nasdaq_composite": "Nasdaq",
    "dow_jones": "Dow Jones",
    "russell_2000": "Russell 2000",
    "vix": "VIX",
    "sse_index": "SSE Index",
    "sse_50": "SSE 50",
    "csi_300": "CSI 300",
    "csi_500": "CSI 500",
    "csi_800": "CSI 800",
    "star_50": "STAR 50",
    "chinext": "ChiNext",
    "szse_component": "SZSE Component",
    "usd_cny": "USD/CNY",
    "cnh_usd": "CNH/USD",
    "us_10y_yield": "US 10Y Yield",
}

CHINA_LINE_COLORS = [
    "#d62728",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]

US_LINE_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
]

FX_LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
]

COMPARISON_LINE_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
    "#9467bd",
]


@dataclass(frozen=True)
class MarketAnalysisOutputs:
    raw_prices: Path
    processed_panel: Path
    normalized_panel: Path
    factor_panel: Path
    correlation_matrix: Path
    period_correlations: Path
    charts_dir: Path


def ensure_output_dirs() -> None:
    RAW_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    PERIOD_CORR_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def download_close_prices(
    tickers: dict[str, str],
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> pd.DataFrame:
    """Download close prices for one ticker group."""
    data = {}
    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if raw.empty:
                print(f"  skipped {ticker} ({name}): no data")
                continue
            data[name] = raw["Close"].squeeze()
            print(f"  downloaded {ticker} -> {name}: {len(raw)} rows")
        except Exception as exc:
            print(f"  skipped {ticker} ({name}): {exc}")
    return pd.DataFrame(data).sort_index()


def download_market_data(
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download US, China, and FX/rate data and persist raw group files."""
    ensure_output_dirs()
    print("Downloading US indices...")
    us = download_close_prices(US_INDICES, start_date, end_date)
    print("Downloading China indices...")
    cn = download_close_prices(CN_INDICES, start_date, end_date)
    print("Downloading FX and rate proxies...")
    fx = download_close_prices(FX_AND_RATE, start_date, end_date)

    us.to_csv(RAW_MARKET_DIR / "us_indices.csv")
    cn.to_csv(RAW_MARKET_DIR / "china_indices.csv")
    fx.to_csv(RAW_MARKET_DIR / "fx_and_rates.csv")

    raw_prices = pd.concat([us, cn, fx], axis=1).sort_index()
    raw_prices.to_csv(RAW_MARKET_DIR / "market_prices.csv")
    return us, cn, fx, raw_prices


def read_cached_raw_prices(path: Path) -> pd.DataFrame:
    """Read cached raw prices regardless of CSV index column casing."""
    cached = pd.read_csv(path)
    date_col = "date" if "date" in cached.columns else "Date" if "Date" in cached.columns else cached.columns[0]
    cached[date_col] = pd.to_datetime(cached[date_col])
    return cached.set_index(date_col).sort_index()


def clean_market_data(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Clean and align raw market data for research use."""
    if raw_prices.empty:
        raise ValueError("No market data available.")

    min_observations = min(500, max(1, int(len(raw_prices) * 0.5)))
    panel = raw_prices.dropna(thresh=min_observations, axis=1).copy()
    variable_cols = [col for col in panel.columns if panel[col].nunique(dropna=True) > 1]
    panel = panel[variable_cols]
    panel = panel.ffill().bfill()

    outlier_date = pd.Timestamp("2011-07-18")
    if "usd_cny" in panel.columns and outlier_date in panel.index:
        value = panel.loc[outlier_date, "usd_cny"]
        if value < 6.0:
            previous_value = panel.loc[:outlier_date, "usd_cny"].iloc[-2]
            next_value = panel.loc[outlier_date:, "usd_cny"].iloc[1]
            panel.loc[outlier_date, "usd_cny"] = (previous_value + next_value) / 2

    return panel


def minmax_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalize each column into a 0-1 range without requiring sklearn."""
    denom = panel.max() - panel.min()
    denom = denom.replace(0, np.nan)
    return ((panel - panel.min()) / denom).replace([np.inf, -np.inf], np.nan)


def correlation_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Return correlations after excluding columns with no variation."""
    variable_cols = [col for col in panel.columns if panel[col].nunique(dropna=True) > 1]
    return panel[variable_cols].corr()


def build_period_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Build period-level correlation diagnostics for key China/US relationships."""
    pairs = [
        ("sp500", "sse_index"),
        ("sp500", "csi_300"),
        ("sp500", "szse_component"),
        ("sp500", "chinext"),
        ("nasdaq_composite", "chinext"),
        ("usd_cny", "sse_index"),
        ("usd_cny", "csi_300"),
        ("usd_cny", "szse_component"),
        ("usd_cny", "chinext"),
    ]

    rows = []
    for left, right in pairs:
        if left not in panel.columns or right not in panel.columns:
            continue
        for period_name, (start, end) in PERIODS.items():
            end_date = pd.Timestamp(end) if end is not None else panel.index.max()
            subset = panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= end_date)]
            corr = subset[left].corr(subset[right]) if len(subset) > 30 else np.nan
            rows.append(
                {
                    "pair": f"{left}_vs_{right}",
                    "left": left,
                    "right": right,
                    "period": period_name,
                    "correlation": corr,
                    "observations": len(subset),
                }
            )
    return pd.DataFrame(rows)


def build_period_correlation_matrices(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build full correlation matrices for each research period."""
    matrices = {}
    for period_name, (start, end) in PERIODS.items():
        end_date = pd.Timestamp(end) if end is not None else panel.index.max()
        subset = panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= end_date)]
        if len(subset) > 30:
            matrices[period_name] = correlation_matrix(subset)
    return matrices


def _title(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def save_line_chart(
    data: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_name: str,
    cmap_name: str = "tab10",
    colors: list[str] | None = None,
) -> None:
    available = [col for col in columns if col in data.columns]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    line_colors = colors or list(plt.get_cmap(cmap_name)(np.linspace(0.15, 0.9, len(available))))
    for i, col in enumerate(available):
        color = line_colors[i % len(line_colors)]
        ax.plot(data.index, data[col], label=_title(col), linewidth=1.4, color=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", ncol=2 if len(available) > 5 else 1)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / output_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(corr: pd.DataFrame) -> None:
    if corr.empty:
        return
    show = [col for col in corr.columns if col != "vix"]
    subset = corr.loc[show, show]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(subset, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(subset.columns)))
    ax.set_yticks(range(len(subset.index)))
    ax.set_xticklabels([_title(c) for c in subset.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_title(c) for c in subset.index], fontsize=8)
    for i in range(len(subset.index)):
        for j in range(len(subset.columns)):
            ax.text(j, i, f"{subset.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Full Correlation Matrix", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "05_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_period_correlation_heatmaps(period_matrices: dict[str, pd.DataFrame]) -> None:
    """Save one heatmap per period correlation matrix."""
    for period_name, matrix in period_matrices.items():
        if matrix.empty:
            continue
        show = [col for col in matrix.columns if col != "vix"]
        subset = matrix.loc[show, show]
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(subset, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(subset.columns)))
        ax.set_yticks(range(len(subset.index)))
        ax.set_xticklabels([_title(c) for c in subset.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([_title(c) for c in subset.index], fontsize=8)
        for i in range(len(subset.index)):
            for j in range(len(subset.columns)):
                ax.text(j, i, f"{subset.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"Correlation Matrix - {period_name}", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
        fig.tight_layout()
        fig.savefig(CHARTS_DIR / f"period_correlation_matrix_{period_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def return_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Build return panel for rolling correlation charts."""
    return build_simple_return_panel(panel, suffix="")


def plot_rolling_pairs(
    ax,
    returns: pd.DataFrame,
    base_col: str,
    candidate_cols: list[str],
    title: str,
    window: int,
) -> None:
    """Plot rolling correlations for pairs with enough non-null observations."""
    line_count = 0
    for col in candidate_cols:
        if base_col not in returns.columns or col not in returns.columns:
            continue
        pair = returns[[base_col, col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < window:
            continue
        rolling = pair[base_col].rolling(window, min_periods=max(20, window // 2)).corr(pair[col])
        rolling = rolling.dropna()
        if rolling.empty:
            continue
        ax.plot(rolling.index, rolling, label=f"{_title(base_col)} vs {_title(col)}", linewidth=1.4)
        line_count += 1
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    if line_count:
        ax.legend(loc="best", ncol=2)
    else:
        ax.text(0.5, 0.5, "No pairs with enough rolling data", ha="center", va="center", transform=ax.transAxes)


def save_rolling_correlation_chart(panel: pd.DataFrame, window: int = ROLLING_WINDOW) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    returns = return_panel(panel)
    china_cols = [
        col
        for col in ["sse_index", "csi_300", "szse_component", "chinext", "csi_500", "sse_50"]
        if col in returns.columns and returns[col].nunique(dropna=True) > 1
    ]

    plot_rolling_pairs(
        axes[0],
        returns,
        "sp500",
        china_cols,
        f"S&P 500 vs China indices: {window}-day rolling return correlation",
        window,
    )
    plot_rolling_pairs(
        axes[1],
        returns,
        "usd_cny",
        china_cols,
        f"USD/CNY vs China indices: {window}-day rolling return correlation",
        window,
    )
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "06_rolling_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_charts(
    panel: pd.DataFrame,
    normalized: pd.DataFrame,
    corr: pd.DataFrame,
    period_matrices: dict[str, pd.DataFrame],
) -> None:
    """Generate market-analysis charts in the dedicated charts directory."""
    save_line_chart(
        normalized,
        list(CN_INDICES.values()),
        "China Stock Indices - Normalized Growth Trend",
        "Normalized value (0-1)",
        "01_china_indices.png",
        "Reds",
        CHINA_LINE_COLORS,
    )
    save_line_chart(
        normalized,
        list(US_INDICES.values()),
        "US Stock Indices - Normalized Growth Trend",
        "Normalized value (0-1)",
        "02_us_indices.png",
        "Blues",
        US_LINE_COLORS,
    )
    save_line_chart(
        normalized,
        list(FX_AND_RATE.values()),
        "FX And Rate Proxies - Normalized",
        "Normalized value (0-1)",
        "03_fx_rates.png",
        "tab10",
        FX_LINE_COLORS,
    )
    save_line_chart(
        normalized,
        ["sp500", "nasdaq_composite", "csi_300", "sse_index", "chinext"],
        "China vs US Market Indices - Comparison",
        "Normalized value (0-1)",
        "04_china_us_comparison.png",
        "tab10",
        COMPARISON_LINE_COLORS,
    )
    save_correlation_heatmap(corr)
    save_period_correlation_heatmaps(period_matrices)
    save_rolling_correlation_chart(panel, ROLLING_WINDOW)
    save_line_chart(
        panel,
        list(FX_AND_RATE.values()),
        "Exchange Rates And Rate Proxy - Raw Data",
        "Raw level",
        "07_fx_raw.png",
        "tab10",
        FX_LINE_COLORS,
    )


def run_market_analysis(
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
    use_cached_raw: bool = True,
) -> MarketAnalysisOutputs:
    """Run the full China/US market research pipeline."""
    ensure_output_dirs()
    raw_path = RAW_MARKET_DIR / "market_prices.csv"
    if use_cached_raw and raw_path.exists():
        raw_prices = read_cached_raw_prices(raw_path)
        raw_prices.index.name = "date"
    else:
        _, _, _, raw_prices = download_market_data(start_date, end_date)
        raw_prices.index.name = "date"

    panel = clean_market_data(raw_prices)
    normalized = minmax_normalize(panel)
    corr = correlation_matrix(panel)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        factors = build_market_research_factors(panel)
    period_corr = build_period_correlations(panel)
    period_matrices = build_period_correlation_matrices(panel)

    processed_panel_path = PROCESSED_MARKET_DIR / "market_price_panel.csv"
    normalized_path = PROCESSED_MARKET_DIR / "market_price_panel_normalized.csv"
    factor_path = PROCESSED_MARKET_DIR / "market_factor_panel.csv"
    corr_path = PROCESSED_MARKET_DIR / "market_correlation_matrix.csv"
    period_path = PROCESSED_MARKET_DIR / "market_period_correlations.csv"

    panel.to_csv(processed_panel_path)
    normalized.to_csv(normalized_path)
    factors.to_csv(factor_path)
    corr.to_csv(corr_path)
    period_corr.to_csv(period_path, index=False)
    for period_name, matrix in period_matrices.items():
        matrix.to_csv(PERIOD_CORR_DIR / f"correlation_matrix_{period_name}.csv")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        save_charts(panel, normalized, corr, period_matrices)

    return MarketAnalysisOutputs(
        raw_prices=raw_path,
        processed_panel=processed_panel_path,
        normalized_panel=normalized_path,
        factor_panel=factor_path,
        correlation_matrix=corr_path,
        period_correlations=period_path,
        charts_dir=CHARTS_DIR,
    )


def main() -> None:
    outputs = run_market_analysis()
    print("Market analysis complete.")
    print(f"Raw prices: {outputs.raw_prices}")
    print(f"Processed panel: {outputs.processed_panel}")
    print(f"Factor panel: {outputs.factor_panel}")
    print(f"Charts: {outputs.charts_dir}")


if __name__ == "__main__":
    main()
