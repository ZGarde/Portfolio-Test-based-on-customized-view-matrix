# Stage 3 Rolling Backtest Audit Report

## Scope

Reviewed file: `src/run_rolling_backtest_bl.py`

Reviewed outputs:

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

## Executive Summary

Audit status: **PASS after fixes**

No confirmed look-ahead bias was found in the rolling estimation, rebalance, Black-Litterman, or custom factor view logic.

Two output/accounting issues were found and fixed:

1. `rolling_backtest_trades.csv` repeated total portfolio transaction cost on every asset trade row. It now records asset-level transaction cost.
2. `rolling_backtest_positions.csv` duplicated position states on rebalance dates. It now keeps the rebalance target state and avoids writing the stale pre-rebalance actual state for the same date.

One performance reporting issue was found and fixed:

1. `average_turnover` was daily-averaged. It now uses rebalance-level turnover.

The strategy results can now be safely used for next-stage analysis, subject to the stated modeling caveats.

## Audit Checklist

| Area | Status | Notes |
|---|---:|---|
| Rolling window logic | PASS | Each rebalance uses `returns.index < rebalance_date` and `tail(window)`. No future holding-period returns enter covariance, expected return, Minimum Variance, Risk Parity, or BL inputs. |
| 252/504 window slicing | PASS | Full 252 or 504 prior observations are required before optimization. |
| Rebalance dates | PASS | Monthly/quarterly dates are resampled from actual asset return trading dates, so they are real observed trading dates. |
| Holding period returns | PASS | Returns start after the first rebalance date. Weight generation and subsequent returns are separated. |
| Custom factor lagging | PASS | Active custom views use `factor_date_used < rebalance_date`; minimum observed lag is 1 day. |
| Custom factor confidence cap | PASS | Active custom confidence max is 0.20, below the 0.30 cap. |
| Custom factor weight control | PASS | Custom factors only create BL P/Q views and never directly overwrite portfolio weights. |
| Weight constraints | PASS | Weight sums equal 1 within numerical tolerance; no negative weights; single asset cap, BTC cap, and Oil cap pass. |
| First build turnover | PASS | First rebalance old weights are zero, so initial turnover is 1.0 and transaction cost is charged. |
| Turnover formula | PASS | `turnover = sum(abs(new_weight - old_weight))`; output matches recomputation. |
| Transaction cost | PASS after fix | Per-asset trade costs now sum to `sum(abs(trade_value)) * 0.001`. |
| Gross/net separation | PASS | Gross and net returns/equity are separately recorded. Net return deducts transaction cost on rebalance holding start only. |
| Equity reconstruction | PASS | Net equity reconstructs from `rolling_backtest_returns.csv` with max final error below `1e-5`. |
| Positions accounting | PASS after fix | Position values now sum to net portfolio value within numerical tolerance. |
| Performance formulas | PASS | Annualized return, Sharpe, final equity, and max drawdown all match independent recomputation. |
| Max drawdown basis | PASS | Drawdown is based on net equity curve. |
| Fallback logging | PASS | Fallback path is implemented. No fallback was triggered in the current run. |
| Risk overlay | PASS for default disabled run | `use_risk_overlay = false`; risk events are empty by design. Enabling overlay should be audited separately when implemented/activated. |

## Detailed Findings

### 1. Rolling Window And Look-Ahead Bias

Result: **PASS**

For each rebalance date, the estimation window is:

```python
hist = returns.loc[returns.index < rebalance_date].tail(window)
```

This excludes the rebalance date return and all future returns. The rolling covariance matrix, rolling annualized return, Minimum Variance, Risk Parity, BL equilibrium returns, and BL posterior returns all use this past-only `hist`.

No future holding-period returns were found in optimization inputs.

### 2. Rebalance Timing

Result: **PASS**

The first out-of-sample return date is strictly after the first rebalance date for every strategy/window/frequency combination.

Monthly and quarterly rebalance dates are selected from actual observed return dates, not calendar-only non-trading dates.

### 3. Custom Factor Views

Result: **PASS**

Active custom factor rows satisfy:

- `factor_date_used < rebalance_date`
- `confidence <= 0.30`
- custom views create P/Q BL views only
- no direct weight override exists

For the China gold custom factor, the log confirms use of `t-1` or earlier values. Example:

```text
rebalance_date = 2019-01-31
factor_date_used = 2019-01-30
```

### 4. Transaction Costs And Turnover

Result: **PASS after fix**

Fixed issue:

Previously, each asset row in `rolling_backtest_trades.csv` repeated the full portfolio transaction cost. This was misleading and could overstate costs if the trade file were summed by asset rows.

Current behavior:

```text
asset transaction_cost = abs(weight_change) * transaction_cost_rate * portfolio_value
```

The sum of trade-row costs equals:

```text
sum(abs(trade_value)) * 0.001
```

Turnover recomputation from trade weights matches `rolling_backtest_turnover.csv`.

### 5. Portfolio Accounting

Result: **PASS after fix**

Fixed issue:

`rolling_backtest_positions.csv` previously contained two position states on rebalance dates:

- stale actual weights from the prior holding period
- new target rebalance weights

This made position values double-count on rebalance dates.

Current behavior:

The stale daily position state is skipped on rebalance dates because the new target position state is recorded by the rebalance block. Position values now sum to portfolio value within numerical tolerance.

### 6. Performance Metrics

Result: **PASS after fix**

Verified formulas:

- Annualized return uses compounded total return with 252 trading days.
- Annualized volatility uses daily return standard deviation times `sqrt(252)`.
- Sharpe ratio uses zero risk-free rate.
- Sortino ratio uses downside volatility.
- Max drawdown is based on net equity curve.
- Calmar ratio is annualized net return divided by absolute max drawdown.
- Final equity equals cumulative net return from daily net returns.

Fixed issue:

`average_turnover` previously used daily average turnover, which understated rebalance turnover. It now uses rebalance-level turnover from `rolling_backtest_turnover.csv`.

### 7. Fallback Logic

Result: **PASS**

Optimization fallback behavior is implemented:

1. Use previous valid weights if available.
2. Otherwise use feasible constrained equal weights.
3. Log fallback in `rolling_backtest_view_log.csv`.

Current run had no optimization fallback rows.

### 8. Output Consistency

Result: **PASS**

Independent checks passed:

- `rolling_backtest_returns.csv` reconstructs net equity.
- `rolling_backtest_weights.csv` satisfies constraints.
- `rolling_backtest_turnover.csv` matches trade weight changes.
- `rolling_backtest_performance_summary.csv` matches recomputed final equity, annualized return, Sharpe, and max drawdown.
- `rolling_backtest_custom_factor_log.csv` explains active BL Custom Factor Views and confirms lagged factor usage.

## Residual Caveats

1. Risk overlay is disabled by default. The current audit verifies the disabled baseline and empty risk-events output. If `USE_RISK_OVERLAY` is enabled later, that path needs a separate implementation audit.
2. Net return deducts transaction cost as `gross_return - transaction_cost_rate` on the first holding day after rebalance. This is consistent with the current project convention, but a broker-style cash ledger could model cost before market return as `(1 - cost) * (1 + gross_return) - 1`.
3. Custom factor views remain research-only. Passing this implementation audit does not validate the China gold factor economically.

## Final Assessment

The rolling backtest implementation is now safe for next-stage analysis.

No confirmed data leakage or future function was found.

The identified output/accounting issues were fixed and outputs were regenerated by rerunning:

```bash
python src/run_rolling_backtest_bl.py
```

