# Multi-Asset Factor Project

This project is a first-stage data engineering and factor calculation pipeline for a multi-asset portfolio research workflow. It downloads data, builds daily returns, creates asset-specific factors, runs simple validation checks, and saves CSV outputs.

It intentionally does not implement Black-Litterman, LSTM models, or a full backtest.

## Data Sources

- `GC=F`: US gold futures from yfinance
- `CL=F`: WTI crude oil futures from yfinance
- `BTC-USD`: Bitcoin from yfinance
- `QQQ`: Nasdaq ETF from yfinance
- `TLT`: long-term US Treasury ETF from yfinance
- `^VIX`: VIX index level from yfinance
- China gold: `akshare.spot_golden_benchmark_sge()`
- US rates: FRED `DGS10`, `DGS2`, `DFF` from pandas_datareader

## Project Structure

```text
multi_asset_factor_project/
    data/
        raw/
        processed/
    src/
        config.py
        get_data.py
        build_clean_factor_panel.py
        strict_factor_review.py
        build_static_portfolios_bl.py
        run_rolling_backtest_bl.py
        factors/
            __init__.py
            common.py
            gold_factors.py
            bond_factors.py
            vix_factors.py
            btc_factors.py
            oil_factors.py
            equity_factors.py
    requirements.txt
    README.md
```

## Factor Meanings

Gold factors measure China-to-US gold linkage, US-to-China gold linkage, and short/medium-term US gold momentum.

Bond factors measure interest-rate changes, yield curve movement, and the negative relationship between rising rates and Treasury ETF returns.

VIX factors use VIX level changes, not VIX percentage returns, as a risk-aversion proxy.

Bitcoin factors measure rolling correlation with Nasdaq, VIX changes, and gold, plus Bitcoin momentum.

Oil factors measure oil sensitivity to VIX changes and oil momentum.

Equity factors measure risk-on behavior through Nasdaq returns minus Treasury returns, plus Nasdaq momentum.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline from the project root:

```bash
python src/get_data.py
python src/build_clean_factor_panel.py
python src/strict_factor_review.py
```

Run the Stage 2 static portfolio and Black-Litterman workflow:

```bash
python src/build_static_portfolios_bl.py
```

Optional user-defined low-confidence Black-Litterman factor views are controlled by:

```text
config/custom_factor_views.yaml
```

## Outputs

- `data/raw/us_asset_prices.csv`: raw yfinance price levels
- `data/raw/fred_rates.csv`: raw FRED interest-rate series
- `data/raw/china_gold_sge.csv`: raw China SGE gold data
- `data/processed/clean_factor_panel.csv`: cleaned factor panel with asset-calendar-aware returns and robust oil return fields
- `data/processed/clean_factor_correlation.csv`: correlation matrix from the clean panel
- `data/processed/clean_factor_quantile_test.csv`: quantile test output from the clean panel
- `data/processed/factor_quality_report.csv`: factor-level quality, IC, quantile spread, and pass/fail diagnostics
- `data/processed/china_gold_signal_alignment_report.csv`: daily proxy China gold alignment tests
- `data/processed/gold_daily_proxy_diagnostic_summary.csv`: strict-review note for the daily China gold proxy
- `data/processed/asset_returns.csv`: aligned asset return matrix for static allocation
- `data/processed/portfolio_weights_static.csv`: traditional and Black-Litterman static weights
- `data/processed/portfolio_ex_ante_performance.csv`: ex-ante return, volatility, Sharpe, and concentration metrics
- `data/processed/portfolio_risk_contribution.csv`: portfolio risk contribution diagnostics
- `data/processed/bl_*`: Black-Litterman views, posterior returns, and weights
- `data/processed/custom_views_matrix.csv`: enabled user-defined custom factor views
- `data/processed/custom_factor_view_log.csv`: validation and look-ahead warnings for custom views
- `data/processed/bl_weights_custom.csv`: Custom Factor Views BL weights
- `data/processed/charts/`: Stage 2 diagnostic figures

## Notes

Correlation is not causality. These checks are only first-pass diagnostics.

Do not use future data in factor construction. The target columns are created only with `shift(-1)`, while factors use current or past information.

This is not a complete trading strategy. It is the data and factor engineering layer that later research can build on.
