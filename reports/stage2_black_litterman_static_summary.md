# Stage 2 Black-Litterman Static Summary

## Scope

This stage generates static portfolio weights and ex-ante diagnostics. It does not run a rolling backtest, cumulative strategy simulation, transaction cost analysis, or turnover analysis.

## Why This Is Still Black-Litterman

Weights are optimized from equilibrium implied returns and Black-Litterman posterior expected returns. Custom factors do not directly determine weights. They only enter through low-confidence views that affect expected returns before optimization.

## Custom Factor Views

Custom factor views are user-selected research views. They are not approved high-confidence signals.

- Factors do not directly decide weights.
- Factors only influence Black-Litterman P/Q views.
- Confidence and Q strength are controlled by each custom view's YAML mapping parameters.
- Missing or invalid custom factors are skipped with warnings.
- Static BL uses the latest available custom factor value; the China gold custom view is intentionally configured with no mechanical shift(1).
- Rolling BL, when implemented, must use only factor values available before each rebalance date.

Active custom views:

- `custom_china_gold_to_us_gold`: Gold Q=-0.46%, confidence=11.6%, factor value=-0.457974.
- `custom_us_cn_nasdaq_sse_corr_60d`: Nasdaq Q=-0.13%, confidence=10.3%, factor value=-0.126649.

## China Gold Caveat

The China gold custom factor now uses `china_gold_morning_to_evening_z`, based on China morning-to-evening return. It is research-only. The intended hypothesis is same-day China trading-session information shock to US gold reaction a few hours later. US gold open/reopen intraday prices are unavailable, so validation uses daily close-to-close proxy returns and cannot strictly prove US high-open behavior.

## Model Hierarchy

Conservative BL remains the main model. Custom BL is user-defined scenario analysis, not a direct trading strategy.

## Notes

- Target volatility <= 12.00% satisfied.
- delta estimated from market proxy: 8.0974

## Ex-Ante Performance

| portfolio_name         |   expected_annual_return |   expected_annual_volatility |   expected_sharpe_ratio |   max_asset_weight |   herfindahl_concentration_index |   portfolio_variance |
|:-----------------------|-------------------------:|-----------------------------:|------------------------:|-------------------:|---------------------------------:|---------------------:|
| Equal Weight           |                0.155614  |                     0.161564 |                0.96317  |           0.2125   |                         0.203125 |            0.026103  |
| Minimum Variance       |                0.0912291 |                     0.106299 |                0.858228 |           0.468117 |                         0.335045 |            0.0112996 |
| Maximum Sharpe         |                0.181661  |                     0.144395 |                1.25808  |           0.5      |                         0.392287 |            0.02085   |
| Risk Parity            |                0.119375  |                     0.119645 |                0.997743 |           0.363341 |                         0.250737 |            0.0143148 |
| Target Volatility      |                0.143345  |                     0.12     |                1.19454  |           0.44876  |                         0.320553 |            0.0144    |
| BL No View             |                0.119364  |                     0.121412 |                0.983124 |           0.35     |                         0.25     |            0.014741  |
| BL Conservative        |                0.106632  |                     0.114759 |                0.929185 |           0.380494 |                         0.265125 |            0.0131696 |
| BL Exploratory         |                0.10871   |                     0.115863 |                0.938265 |           0.381268 |                         0.263913 |            0.0134243 |
| BL Custom Factor Views |                0.10019   |                     0.115486 |                0.867545 |           0.408679 |                         0.272475 |            0.0133371 |
