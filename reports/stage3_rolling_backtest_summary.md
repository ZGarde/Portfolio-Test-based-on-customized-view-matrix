# Stage 3 Rolling Backtest Summary

## Goal

Validate static allocation methods out of sample with rolling covariance, rolling Black-Litterman inputs, rebalance schedules, transaction costs, and net equity curves.

## Data And Settings

- Data: `asset_returns.csv`, `clean_factor_panel.csv`, and optional `custom_factor_views.yaml`.
- Initial capital: $1,000,000.
- Backtest dates: 2019-01-08 to 2026-04-28.
- Rolling windows: [252, 504].
- Rebalance frequencies: ['daily', 'monthly', 'quarterly'].
- Transaction cost: 10 bps one-way, charged on turnover.
- Risk overlay enabled: False.

## Portfolio Value And Returns

Gross return is the daily portfolio return before trading costs. Net return subtracts transaction cost on rebalance days. Gross and net portfolio values both start from the initial capital. Position weights drift daily with asset returns until the next rebalance.

## Look-Ahead Controls

Each rebalance uses only returns strictly before the rebalance date. Weights are generated on the rebalance date, and returns are calculated only from the following holding period. Custom factors use the configured availability rule and search only for values available at or before `rebalance_date - lag_days`. The China gold custom factor is intentionally configured with `lag_days: 0` to test the same-day proxy hypothesis instead of mechanically shifting the signal by one day.

## Strategy Set

Minimum Variance and Risk Parity are non-BL baselines. BL No View tests equilibrium implied returns. BL Conservative is the main model with low-confidence macro views. BL Exploratory and BL Custom Factor Views are research scenario models. Custom factors never directly determine weights; they only affect BL views.

## China Gold Caveat

China gold to US gold remains research-only. The revised custom factor uses China morning-to-evening return z-score and tests the same-day proxy hypothesis. US gold open/reopen intraday data are unavailable, so the current evidence is daily close-to-close proxy evidence and cannot strictly prove high-open behavior.

## Risk Overlay

The overlay is disabled by default. When enabled, it is a risk overlay only and does not change Black-Litterman expected returns. This run has no risk events unless explicitly enabled.

## Results

- Highest annualized net return: Risk Parity / window 252 / quarterly.
- Highest net Sharpe: Risk Parity / window 252 / quarterly.
- Lowest max drawdown: Minimum Variance / window 252 / daily.
- Highest transaction cost: BL Custom Factor Views / window 252 / daily.
- Robust candidate by drawdown then Sharpe: Minimum Variance / window 252 / daily.

Custom BL performance does not prove the China gold factor is valid. Good performance may come from portfolio construction, covariance, or macro views; poor performance may indicate weak scenario relevance or unstable factor timing.

## Outputs

Files:

- `rolling_backtest_returns.csv`
- `rolling_backtest_weights.csv`
- `rolling_backtest_turnover.csv`
- `rolling_backtest_performance_summary.csv`
- `rolling_backtest_drawdowns.csv`
- `rolling_backtest_risk_contribution.csv`
- `rolling_backtest_view_log.csv`
- `rolling_backtest_custom_factor_log.csv`
- `rolling_backtest_equity_curve.csv`
- `rolling_backtest_positions.csv`
- `rolling_backtest_trades.csv`
- `rolling_backtest_risk_events.csv`
- `rolling_backtest_config_summary.csv`

Charts:

- `drawdown_curve.png`
- `equity_curve_gross.png`
- `equity_curve_net.png`
- `monthly_returns_heatmap.png`
- `risk_events_timeline.png`
- `rolling_bl_posterior_returns.png`
- `rolling_cumulative_net_returns_comparison.png`
- `rolling_cumulative_returns_comparison.png`
- `rolling_custom_china_gold_signal_vs_Gold_weight.png`
- `rolling_custom_china_gold_to_us_gold_signal_vs_Gold_weight.png`
- `rolling_custom_factor_confidence.png`
- `rolling_custom_factor_signal_vs_gold_weight.png`
- `rolling_custom_sse_to_nasdaq_signal_z_signal_vs_Nasdaq_weight.png`
- `rolling_custom_us_cn_nasdaq_sse_corr_60d_signal_vs_Nasdaq_weight.png`
- `rolling_custom_vix_to_nasdaq_signal_vs_Nasdaq_weight.png`
- `rolling_drawdown_comparison.png`
- `rolling_sharpe_comparison.png`
- `rolling_turnover_comparison.png`
- `rolling_volatility_comparison.png`
- `rolling_weight_evolution.png`
- `with_vs_without_risk_overlay_drawdown.png`
- `with_vs_without_risk_overlay_equity_curve.png`
