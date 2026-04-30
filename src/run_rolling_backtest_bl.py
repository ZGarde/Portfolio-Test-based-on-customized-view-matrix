"""Stage 3: rolling out-of-sample Black-Litterman backtest.

Important anti-look-ahead rule:
Weights are generated on each rebalance date using only data strictly before
that rebalance date. Returns are then calculated from the holding period after
the rebalance date. Custom factors are converted only into low-confidence
Black-Litterman views and never directly set portfolio weights.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BASE_DIR, PROCESSED_DATA_DIR
from build_static_portfolios_bl import (
    ASSET_ORDER,
    MARKET_WEIGHTS,
    MAX_WEIGHTS,
    TARGET_VOL,
    TRADING_DAYS,
    black_litterman_posterior,
    conservative_views,
    estimate_delta,
    implied_equilibrium_returns,
    optimize_max_sharpe,
    optimize_min_variance,
    optimize_risk_parity,
    risk_contribution,
    portfolio_variance,
    load_custom_factor_views,
    resolve_project_path,
    custom_view_mapping_params,
    dynamic_zscore_custom_view,
    CUSTOM_ALIGNMENT_ASSUMPTION,
    CUSTOM_PROXY_WARNING,
    CUSTOM_TRIGGER_RULE,
)


INITIAL_CAPITAL = 1_000_000.0
TRANSACTION_COST_BPS = 10
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10_000
ROLLING_WINDOWS = [252, 504]
REBALANCE_FREQUENCIES = {"daily": "B", "monthly": "ME", "quarterly": "QE"}
USE_RISK_OVERLAY = False
RISK_OVERLAY_CONFIG = {
    "portfolio_stop_loss": -0.10,
    "portfolio_take_profit": 0.20,
    "asset_stop_loss": -0.12,
    "asset_take_profit": 0.25,
    "trailing_stop_loss": -0.08,
    "cooldown_days": 5,
    "after_trigger_action": "move_to_cash",
}
STRATEGIES = [
    "Minimum Variance",
    "Risk Parity",
    "BL No View",
    "BL Conservative",
    "BL Exploratory",
    "BL Custom Factor Views",
]


def constrained_equal_weight() -> pd.Series:
    """Return a feasible equal-ish weight portfolio under BTC/Oil caps."""
    weights = pd.Series(1.0 / len(ASSET_ORDER), index=ASSET_ORDER)
    weights["Bitcoin"] = min(weights["Bitcoin"], MAX_WEIGHTS["Bitcoin"])
    excess = 1.0 - weights.sum()
    eligible = [a for a in ASSET_ORDER if a != "Bitcoin"]
    for asset in eligible:
        weights[asset] += excess / len(eligible)
    return weights / weights.sum()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned asset returns and factor panel."""
    returns_path = PROCESSED_DATA_DIR / "asset_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError("Run python src/build_static_portfolios_bl.py first to create asset_returns.csv")
    returns = pd.read_csv(returns_path, parse_dates=["date"], index_col="date")
    returns = returns[ASSET_ORDER].replace([np.inf, -np.inf], np.nan).dropna(how="any")

    factor_path = PROCESSED_DATA_DIR / "clean_factor_panel.csv"
    if factor_path.exists():
        factors = pd.read_csv(factor_path, parse_dates=["date"], index_col="date")
    else:
        warnings.warn("clean_factor_panel.csv missing; factor-based views will be skipped.")
        factors = pd.DataFrame(index=returns.index)
    return returns, factors


def get_rebalance_dates(index: pd.DatetimeIndex, window: int, freq_alias: str) -> list[pd.Timestamp]:
    """Return rebalance dates after the estimation window is available."""
    eligible = index[window:]
    if len(eligible) == 0:
        return []
    grouped = pd.Series(eligible, index=eligible).resample(freq_alias).last().dropna()
    return [pd.Timestamp(d) for d in grouped.values]


def latest_available_factor(
    factor_df: pd.DataFrame,
    factor_col: str,
    rebalance_date: pd.Timestamp,
    lag_days: int,
) -> tuple[pd.Timestamp | None, float | None]:
    """Find latest factor value at or before rebalance_date - lag_days."""
    if factor_col not in factor_df.columns:
        return None, None
    cutoff = pd.Timestamp(rebalance_date) - pd.Timedelta(days=max(lag_days, 0))
    series = factor_df.loc[factor_df.index <= cutoff, factor_col].replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None, None
    return series.index[-1], float(series.iloc[-1])


def rolling_exploratory_views(
    factors: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, list[float], list[dict]]:
    """Create low-confidence exploratory BL views using only past factor data."""
    past = factors.loc[factors.index < rebalance_date].copy()
    groups = {
        "risk_off": ["vix_change", "btc_nasdaq_corr_60d", "btc_gold_corr_60d"],
        "inflation_commodity": [
            "oil_mom_5d",
            "oil_mom_20d",
            "oil_winsorized_return",
            "oil_dollar_change_return",
            "oil_vix_factor",
        ],
        "liquidity_risk_on": ["btc_mom_5d", "btc_mom_20d", "vix_change", "nasdaq", "treasury"],
    }
    scores = {}
    logs = []
    for group, cols in groups.items():
        zscores = []
        for col in cols:
            if col not in past.columns:
                logs.append({"factor_name": col, "view_status": "skipped", "skip_reason": "missing_factor"})
                continue
            series = past[col].dropna()
            if len(series) < 60 or series.std() == 0:
                logs.append({"factor_name": col, "view_status": "skipped", "skip_reason": "insufficient_history"})
                continue
            z = (series.iloc[-1] - series.mean()) / series.std()
            if group == "liquidity_risk_on" and col == "vix_change":
                z = -z
            zscores.append(z)
            logs.append({"factor_name": col, "scenario_group": group, "latest_zscore": z, "view_status": "used"})
        scores[group] = float(np.nanmean(zscores)) if zscores else np.nan

    views = []
    if pd.notna(scores.get("risk_off")) and scores["risk_off"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["risk_off"]))
        views += [
            ("Treasury_outperforms_Bitcoin", [0, 1, -1, 0, 0], 0.020, conf, "risk_off", "Treasury"),
            ("Gold_outperforms_Nasdaq", [1, 0, 0, 0, -1], 0.015, conf, "risk_off", "Gold"),
        ]
    if pd.notna(scores.get("inflation_commodity")) and scores["inflation_commodity"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["inflation_commodity"]))
        views += [
            ("Oil_outperforms_Treasury", [0, -1, 0, 1, 0], 0.015, conf, "inflation_commodity", "Oil"),
            ("Gold_outperforms_Treasury", [1, -1, 0, 0, 0], 0.010, conf, "inflation_commodity", "Gold"),
        ]
    if pd.notna(scores.get("liquidity_risk_on")) and scores["liquidity_risk_on"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["liquidity_risk_on"]))
        views += [
            ("Nasdaq_outperforms_Treasury", [0, -1, 0, 0, 1], 0.025, conf, "liquidity_risk_on", "Nasdaq"),
            ("Bitcoin_outperforms_Treasury", [0, -1, 1, 0, 0], 0.030, conf, "liquidity_risk_on", "Bitcoin"),
        ]

    if not views:
        return pd.DataFrame(columns=ASSET_ORDER), pd.Series(dtype=float), [], logs
    P = pd.DataFrame({v[0]: v[1] for v in views}, index=ASSET_ORDER).T
    Q = pd.Series({v[0]: v[2] for v in views})
    conf = [v[3] for v in views]
    for name, pvec, q, c, group, asset in views:
        logs.append(
            {
                "view_name": name,
                "view_type": f"exploratory_{group}",
                "affected_asset": asset,
                "P_vector": pvec,
                "Q_annualized": q,
                "confidence": c,
                "view_status": "active",
            }
        )
    return P, Q, conf, logs


def rolling_custom_views(
    factors: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, list[float], list[dict]]:
    """Convert enabled custom z-score factors into dynamic low-confidence BL views."""
    config_path = BASE_DIR / "config" / "custom_factor_views.yaml"
    if not config_path.exists():
        alt = BASE_DIR / "custom_factor_views.yaml"
        config_path = alt if alt.exists() else config_path
    config = load_custom_factor_views(config_path)
    logs = []
    rows = []
    q_values = {}
    confidences = []
    if not config.get("use_custom_factor_views", False):
        return pd.DataFrame(columns=ASSET_ORDER), pd.Series(dtype=float), [], logs

    for item in config.get("custom_views", []):
        name = item.get("name", "unnamed_custom_view")
        factor_name = f"custom_{name}"
        if not item.get("enabled", False):
            logs.append({"rebalance_date": rebalance_date, "factor_name": factor_name, "view_status": "skipped", "skip_reason": "disabled"})
            continue

        affected_asset = item.get("affected_asset")
        factor_col = item.get("factor_column")
        lag_days = int(item.get("lag_days", 0) or 0)
        direction = str(item.get("direction", "positive")).lower()
        factor_file = resolve_project_path(str(item.get("factor_file", "")))

        if affected_asset not in ASSET_ORDER:
            logs.append({"rebalance_date": rebalance_date, "factor_name": factor_name, "view_status": "skipped", "skip_reason": "invalid_asset"})
            continue

        factor_source = factors
        if factor_file.exists() and factor_file.name != "clean_factor_panel.csv":
            try:
                factor_source = pd.read_csv(factor_file, parse_dates=["date"], index_col="date")
            except Exception as exc:
                logs.append({"rebalance_date": rebalance_date, "factor_name": factor_name, "view_status": "skipped", "skip_reason": f"factor_file_read_failed: {exc}"})
                continue
        elif not factor_file.exists() and factor_col not in factors.columns:
            logs.append({"rebalance_date": rebalance_date, "factor_name": factor_name, "view_status": "skipped", "skip_reason": "missing_factor_file"})
            continue

        factor_date, factor_value = latest_available_factor(factor_source, factor_col, rebalance_date, lag_days)
        if factor_value is None:
            logs.append({"rebalance_date": rebalance_date, "factor_name": factor_name, "view_status": "skipped", "skip_reason": "no_lagged_factor_value"})
            continue

        direction_multiplier = 1.0 if direction == "positive" else -1.0
        z_score_used = direction_multiplier * factor_value
        mapping_params = custom_view_mapping_params(item)
        q, confidence, trigger_status = dynamic_zscore_custom_view(z_score_used, mapping_params)

        if trigger_status == "active_dynamic_zscore":
            p = pd.Series(0.0, index=ASSET_ORDER)
            p[affected_asset] = 1.0
            rows.append((factor_name, p))
            q_values[factor_name] = q
            confidences.append(confidence)
        logs.append(
            {
                "rebalance_date": rebalance_date,
                "factor_name": factor_name,
                "view_name": factor_name,
                "factor_column": factor_col,
                "factor_date_used": factor_date,
                "factor_value_used": factor_value,
                "raw_factor_value": factor_value,
                "z_score_used": z_score_used,
                "affected_asset": affected_asset,
                "direction": direction,
                "Q_annualized": q,
                "confidence": confidence,
                "trigger_threshold": mapping_params["trigger_threshold"],
                "q_scale": mapping_params["q_scale"],
                "q_cap": mapping_params["q_cap"],
                "base_confidence": mapping_params["base_confidence"],
                "confidence_slope": mapping_params["confidence_slope"],
                "max_confidence": mapping_params["max_confidence"],
                "view_status": "active" if trigger_status == "active_dynamic_zscore" else "skipped",
                "trigger_status": trigger_status,
                "trigger_rule": CUSTOM_TRIGGER_RULE,
                "alignment_assumption": CUSTOM_ALIGNMENT_ASSUMPTION,
                "proxy_warning": CUSTOM_PROXY_WARNING,
                "skip_reason": "" if trigger_status == "active_dynamic_zscore" else trigger_status,
            }
        )

    if not rows:
        return pd.DataFrame(columns=ASSET_ORDER), pd.Series(dtype=float), [], logs
    P = pd.DataFrame({name: row for name, row in rows}).T
    Q = pd.Series(q_values)
    return P, Q, confidences, logs


def safe_optimize(strategy: str, mu: pd.Series, cov: pd.DataFrame, previous: pd.Series | None, log_rows: list, date, window, freq) -> pd.Series:
    """Optimize one strategy with fallback to previous or feasible equal weights."""
    try:
        if strategy == "Minimum Variance":
            return optimize_min_variance(cov)
        if strategy == "Risk Parity":
            return optimize_risk_parity(cov)
        return optimize_max_sharpe(mu, cov)
    except Exception as exc:
        fallback = previous.copy() if previous is not None else constrained_equal_weight()
        log_rows.append(
            {
                "rebalance_date": date,
                "strategy": strategy,
                "window": window,
                "rebalance_frequency": freq,
                "view_type": "optimization_fallback",
                "affected_asset": "",
                "P_vector": "",
                "Q_annualized": np.nan,
                "confidence": np.nan,
                "view_status": "fallback",
                "message": str(exc),
            }
        )
        return fallback


def build_weights_for_date(
    returns_window: pd.DataFrame,
    factors: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    previous_weights: dict,
    window: int,
    freq: str,
) -> tuple[dict[str, pd.Series], list[dict], list[dict], list[dict]]:
    """Build all strategy weights on one rebalance date using past-only data."""
    mu = returns_window.mean() * TRADING_DAYS
    cov = returns_window.cov() * TRADING_DAYS
    delta, _ = estimate_delta(mu, cov)
    pi = implied_equilibrium_returns(cov, delta)
    P_cons, Q_cons, conf_cons = conservative_views()

    view_logs = []
    custom_logs = []
    risk_contrib_rows = []

    no_view_mu = black_litterman_posterior(pi, cov)
    cons_mu = black_litterman_posterior(pi, cov, P_cons, Q_cons, conf_cons)
    P_expl, Q_expl, conf_expl, expl_logs = rolling_exploratory_views(factors, rebalance_date)
    expl_mu = black_litterman_posterior(pi, cov, P_expl, Q_expl, conf_expl) if len(P_expl) else no_view_mu
    P_custom_only, Q_custom_only, conf_custom_only, custom_logs = rolling_custom_views(factors, rebalance_date)
    if len(P_custom_only):
        P_custom = pd.concat([P_cons, P_custom_only], axis=0)
        Q_custom = pd.concat([Q_cons, Q_custom_only], axis=0)
        conf_custom = conf_cons + conf_custom_only
    else:
        P_custom, Q_custom, conf_custom = P_cons, Q_cons, conf_cons
    custom_mu = black_litterman_posterior(pi, cov, P_custom, Q_custom, conf_custom)

    for name, p, q, c in zip(P_cons.index, P_cons.values.tolist(), Q_cons.values.tolist(), conf_cons):
        view_logs.append({"rebalance_date": rebalance_date, "strategy": "BL Conservative", "view_type": "conservative", "affected_asset": "", "P_vector": p, "Q_annualized": q, "confidence": c, "view_status": "active"})
    for row in expl_logs:
        row.update({"rebalance_date": rebalance_date, "strategy": "BL Exploratory"})
        view_logs.append(row)
    for row in custom_logs:
        view_logs.append({"rebalance_date": rebalance_date, "strategy": "BL Custom Factor Views", "view_type": "custom", "affected_asset": row.get("affected_asset", ""), "P_vector": "", "Q_annualized": row.get("Q_annualized", np.nan), "confidence": row.get("confidence", np.nan), "view_status": row.get("view_status", ""), "message": row.get("skip_reason", "")})

    mu_map = {
        "Minimum Variance": mu,
        "Risk Parity": mu,
        "BL No View": no_view_mu,
        "BL Conservative": cons_mu,
        "BL Exploratory": expl_mu,
        "BL Custom Factor Views": custom_mu,
    }
    weights = {}
    for strategy in STRATEGIES:
        weights[strategy] = safe_optimize(
            strategy,
            mu_map[strategy],
            cov,
            previous_weights.get(strategy),
            view_logs,
            rebalance_date,
            window,
            freq,
        )
        rc = risk_contribution(weights[strategy], cov)
        for _, r in rc.iterrows():
            risk_contrib_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "strategy": strategy,
                    "window": window,
                    "rebalance_frequency": freq,
                    "asset": r["asset"],
                    "weight": r["weight"],
                    "marginal_risk_contribution": r["marginal_risk_contribution"],
                    "total_risk_contribution": r["component_risk_contribution"],
                    "percent_risk_contribution": r["percentage_risk_contribution"],
                }
            )
    return weights, view_logs, custom_logs, risk_contrib_rows


def simulate_path(
    returns: pd.DataFrame,
    weights_by_date: dict[pd.Timestamp, dict[str, pd.Series]],
    window: int,
    freq: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Simulate out-of-sample holding periods after each rebalance date."""
    equity_rows, position_rows, trade_rows, return_rows, turnover_rows = [], [], [], [], []
    rebal_dates = sorted(weights_by_date.keys())
    rebal_date_set = set(rebal_dates)
    for strategy in STRATEGIES:
        gross_value = INITIAL_CAPITAL
        net_value = INITIAL_CAPITAL
        current_weights = pd.Series(0.0, index=ASSET_ORDER)
        peak_net = INITIAL_CAPITAL
        for i, rebalance_date in enumerate(rebal_dates):
            new_weights = weights_by_date[rebalance_date][strategy].reindex(ASSET_ORDER).fillna(0.0)
            turnover = float((new_weights - current_weights).abs().sum())
            transaction_cost = turnover * TRANSACTION_COST_RATE
            for asset in ASSET_ORDER:
                asset_turnover = abs(new_weights[asset] - current_weights[asset])
                trade_rows.append(
                    {
                        "date": rebalance_date,
                        "strategy": strategy,
                        "window": window,
                        "rebalance_frequency": freq,
                        "asset": asset,
                        "old_weight": current_weights[asset],
                        "new_weight": new_weights[asset],
                        "weight_change": new_weights[asset] - current_weights[asset],
                        "trade_value": (new_weights[asset] - current_weights[asset]) * net_value,
                        "turnover": turnover,
                        "transaction_cost": asset_turnover * TRANSACTION_COST_RATE * net_value,
                        "reason": "scheduled_rebalance",
                    }
                )
                position_rows.append(
                    {
                        "date": rebalance_date,
                        "strategy": strategy,
                        "window": window,
                        "rebalance_frequency": freq,
                        "asset": asset,
                        "target_weight": new_weights[asset],
                        "actual_weight": new_weights[asset],
                        "position_value": new_weights[asset] * net_value,
                        "portfolio_value": net_value,
                    }
                )
            turnover_rows.append({"rebalance_date": rebalance_date, "strategy": strategy, "window": window, "rebalance_frequency": freq, "turnover": turnover, "transaction_cost": transaction_cost})
            current_weights = new_weights.copy()
            start = returns.index.get_loc(rebalance_date) + 1
            end_date = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
            holding_dates = returns.index[(returns.index > rebalance_date) & (returns.index <= end_date)]

            for j, date in enumerate(holding_dates):
                asset_ret = returns.loc[date, ASSET_ORDER]
                gross_ret = float((current_weights * asset_ret).sum())
                net_ret = gross_ret - transaction_cost if j == 0 else gross_ret
                gross_value *= 1 + gross_ret
                net_value *= 1 + net_ret
                peak_net = max(peak_net, net_value)
                drawdown = net_value / peak_net - 1
                current_weights = current_weights * (1 + asset_ret)
                denom = float(current_weights.sum())
                if denom != 0:
                    current_weights = current_weights / denom
                return_rows.append({"date": date, "strategy": strategy, "window": window, "rebalance_frequency": freq, "gross_return": gross_ret, "net_return": net_ret, "transaction_cost": transaction_cost if j == 0 else 0.0, "turnover": turnover if j == 0 else 0.0})
                equity_rows.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "window": window,
                        "rebalance_frequency": freq,
                        "portfolio_value_gross": gross_value,
                        "portfolio_value_net": net_value,
                        "cash": 0.0,
                        "position_value": net_value,
                        "daily_gross_return": gross_ret,
                        "daily_net_return": net_ret,
                        "cumulative_gross_return": gross_value / INITIAL_CAPITAL - 1,
                        "cumulative_net_return": net_value / INITIAL_CAPITAL - 1,
                        "drawdown": drawdown,
                        "peak_portfolio_value": peak_net,
                    }
                )
                # When date is the next rebalance date, target positions for the
                # new weights are recorded by the next rebalance block. Avoid
                # writing two incompatible position states for the same date.
                if date not in rebal_date_set:
                    for asset in ASSET_ORDER:
                        position_rows.append(
                            {
                                "date": date,
                                "strategy": strategy,
                                "window": window,
                                "rebalance_frequency": freq,
                                "asset": asset,
                                "target_weight": new_weights[asset],
                                "actual_weight": current_weights[asset],
                                "position_value": current_weights[asset] * net_value,
                                "portfolio_value": net_value,
                            }
                        )
    return equity_rows, position_rows, trade_rows, return_rows, turnover_rows


def drawdown_dates(group: pd.DataFrame) -> tuple[str, str]:
    """Find max drawdown peak and trough dates."""
    trough_idx = group["drawdown"].idxmin()
    trough_date = group.loc[trough_idx, "date"]
    prior = group[group["date"] <= trough_date]
    peak_idx = prior["portfolio_value_net"].idxmax()
    return str(group.loc[peak_idx, "date"].date()), str(trough_date.date())


def performance_summary(
    equity: pd.DataFrame,
    returns: pd.DataFrame,
    trades: pd.DataFrame,
    turnover: pd.DataFrame,
    risk_events: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate required rolling backtest performance metrics."""
    rows = []
    for keys, eq in equity.groupby(["strategy", "window", "rebalance_frequency"]):
        strategy, window, freq = keys
        ret = returns[(returns["strategy"] == strategy) & (returns["window"] == window) & (returns["rebalance_frequency"] == freq)].copy()
        tr = trades[(trades["strategy"] == strategy) & (trades["window"] == window) & (trades["rebalance_frequency"] == freq)]
        to = turnover[(turnover["strategy"] == strategy) & (turnover["window"] == window) & (turnover["rebalance_frequency"] == freq)]
        eq = eq.sort_values("date")
        n_days = len(eq)
        final_gross = eq["portfolio_value_gross"].iloc[-1]
        final_net = eq["portfolio_value_net"].iloc[-1]
        total_gross = final_gross / INITIAL_CAPITAL - 1
        total_net = final_net / INITIAL_CAPITAL - 1
        ann_gross = (1 + total_gross) ** (TRADING_DAYS / max(n_days, 1)) - 1
        ann_net = (1 + total_net) ** (TRADING_DAYS / max(n_days, 1)) - 1
        vol = ret["net_return"].std() * math.sqrt(TRADING_DAYS)
        downside = ret.loc[ret["net_return"] < 0, "net_return"].std() * math.sqrt(TRADING_DAYS)
        sharpe_g = ann_gross / (ret["gross_return"].std() * math.sqrt(TRADING_DAYS)) if ret["gross_return"].std() > 0 else np.nan
        sharpe_n = ann_net / vol if vol and vol > 0 else np.nan
        sortino = ann_net / downside if downside and downside > 0 else np.nan
        max_dd = eq["drawdown"].min()
        dd_start, dd_end = drawdown_dates(eq)
        monthly = eq.set_index("date")["portfolio_value_net"].resample("ME").last().pct_change().dropna()
        stop_events = 0 if risk_events.empty else risk_events[(risk_events["strategy"] == strategy) & risk_events["event_type"].str.contains("stop", na=False)].shape[0]
        take_events = 0 if risk_events.empty else risk_events[(risk_events["strategy"] == strategy) & risk_events["event_type"].str.contains("take", na=False)].shape[0]
        rows.append(
            {
                "strategy": strategy,
                "window": window,
                "rebalance_frequency": freq,
                "initial_capital": INITIAL_CAPITAL,
                "final_portfolio_value_gross": final_gross,
                "final_portfolio_value_net": final_net,
                "total_return_gross": total_gross,
                "total_return_net": total_net,
                "annualized_return_gross": ann_gross,
                "annualized_return_net": ann_net,
                "annualized_volatility": vol,
                "sharpe_ratio_gross": sharpe_g,
                "sharpe_ratio_net": sharpe_n,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "max_drawdown_start_date": dd_start,
                "max_drawdown_end_date": dd_end,
                "calmar_ratio": ann_net / abs(max_dd) if max_dd < 0 else np.nan,
                "average_monthly_return": monthly.mean(),
                "best_month": monthly.max(),
                "worst_month": monthly.min(),
                "win_rate_monthly": (monthly > 0).mean() if len(monthly) else np.nan,
                "average_turnover": to["turnover"].mean(),
                "total_turnover": to["turnover"].sum(),
                "total_transaction_cost": tr["transaction_cost"].sum(),
                "number_of_rebalances": len(to),
                "number_of_stop_loss_events": stop_events,
                "number_of_take_profit_events": take_events,
                "final_cumulative_return": total_gross,
                "final_cumulative_net_return": total_net,
            }
        )
    return pd.DataFrame(rows)


def safe_plot(name: str, plot_func) -> None:
    """Save a plot without interrupting the backtest when a chart has no data."""
    charts = PROCESSED_DATA_DIR / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    try:
        plot_func()
        plt.tight_layout()
        plt.savefig(charts / name, dpi=150)
        plt.close()
    except Exception as exc:
        plt.close()
        warnings.warn(f"Could not create {name}: {exc}")


def create_charts(equity: pd.DataFrame, summary: pd.DataFrame, weights: pd.DataFrame, turnover: pd.DataFrame, custom_log: pd.DataFrame, risk_events: pd.DataFrame) -> None:
    """Generate rolling backtest charts."""
    sample = equity[(equity["window"] == ROLLING_WINDOWS[0]) & (equity["rebalance_frequency"] == "monthly")]
    summary_plot = summary.copy()
    summary_plot["setting"] = summary_plot["window"].astype(str) + "_" + summary_plot["rebalance_frequency"].astype(str)
    safe_plot("rolling_cumulative_returns_comparison.png", lambda: sample.pivot(index="date", columns="strategy", values="cumulative_gross_return").plot(figsize=(11, 6), title="Rolling Cumulative Gross Returns"))
    safe_plot("rolling_cumulative_net_returns_comparison.png", lambda: sample.pivot(index="date", columns="strategy", values="cumulative_net_return").plot(figsize=(11, 6), title="Rolling Cumulative Net Returns"))
    safe_plot("rolling_drawdown_comparison.png", lambda: sample.pivot(index="date", columns="strategy", values="drawdown").plot(figsize=(11, 6), title="Rolling Drawdown Comparison"))
    safe_plot("equity_curve_gross.png", lambda: sample.pivot(index="date", columns="strategy", values="portfolio_value_gross").plot(figsize=(11, 6), title="Gross Equity Curve"))
    safe_plot("equity_curve_net.png", lambda: sample.pivot(index="date", columns="strategy", values="portfolio_value_net").plot(figsize=(11, 6), title="Net Equity Curve"))
    safe_plot("drawdown_curve.png", lambda: sample.pivot(index="date", columns="strategy", values="drawdown").plot(figsize=(11, 6), title="Drawdown Curve"))
    safe_plot("rolling_volatility_comparison.png", lambda: summary_plot.pivot(index="strategy", columns="setting", values="annualized_volatility").plot(kind="bar", figsize=(11, 6), title="Annualized Volatility by Strategy/Setting"))
    safe_plot("rolling_sharpe_comparison.png", lambda: summary_plot.pivot(index="strategy", columns="setting", values="sharpe_ratio_net").plot(kind="bar", figsize=(11, 6), title="Net Sharpe by Strategy/Setting"))
    safe_plot("rolling_turnover_comparison.png", lambda: turnover.groupby(["strategy", "window"])["turnover"].mean().unstack().plot(kind="bar", figsize=(10, 6), title="Average Turnover"))
    safe_plot("rolling_weight_evolution.png", lambda: weights[(weights["strategy"] == "BL Conservative") & (weights["window"] == ROLLING_WINDOWS[0]) & (weights["rebalance_frequency"] == "monthly")].pivot(index="rebalance_date", columns="asset", values="weight").plot(figsize=(11, 6), title="BL Conservative Weight Evolution"))
    safe_plot("rolling_bl_posterior_returns.png", lambda: weights[weights["strategy"].str.contains("BL")].groupby(["strategy", "asset"])["weight"].mean().unstack().plot(kind="bar", figsize=(10, 6), title="Average BL Weights Proxy for Posterior Tilt"))
    safe_plot("rolling_custom_factor_confidence.png", lambda: custom_log[custom_log["view_status"].eq("active")].plot(x="rebalance_date", y="confidence", figsize=(10, 5), title="Rolling Custom Factor Confidence"))
    create_custom_factor_weight_charts(custom_log, weights)
    safe_plot("monthly_returns_heatmap.png", lambda: monthly_heatmap(sample))
    safe_plot("risk_events_timeline.png", lambda: risk_event_plot(risk_events))
    safe_plot("with_vs_without_risk_overlay_equity_curve.png", lambda: sample.pivot(index="date", columns="strategy", values="portfolio_value_net").plot(figsize=(11, 6), title="With vs Without Risk Overlay (Overlay Disabled)"))
    safe_plot("with_vs_without_risk_overlay_drawdown.png", lambda: sample.pivot(index="date", columns="strategy", values="drawdown").plot(figsize=(11, 6), title="With vs Without Risk Overlay Drawdown (Overlay Disabled)"))


def safe_filename(value: str) -> str:
    """Create a conservative filename fragment from a factor or asset name."""
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def create_custom_factor_weight_charts(custom_log: pd.DataFrame, weights: pd.DataFrame) -> None:
    """Create one signal-vs-affected-asset-weight chart per active custom factor."""
    charts_dir = PROCESSED_DATA_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    if custom_log.empty or "view_status" not in custom_log.columns:
        warnings.warn("No custom factor log available; custom factor weight charts skipped.")
        return

    active = custom_log[custom_log["view_status"].eq("active")].copy()
    if active.empty:
        warnings.warn("No active custom factors; custom factor weight charts skipped.")
        return

    active["rebalance_date"] = pd.to_datetime(active["rebalance_date"])
    weights = weights.copy()
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])

    generated = []
    for (factor_name, affected_asset), group in active.groupby(["factor_name", "affected_asset"]):
        asset_weights = weights[
            weights["strategy"].eq("BL Custom Factor Views")
            & weights["asset"].eq(affected_asset)
        ].copy()
        if asset_weights.empty:
            warnings.warn(f"No BL Custom Factor Views weights found for {affected_asset}; skipped {factor_name}.")
            continue

        merged = group.merge(
            asset_weights[["rebalance_date", "weight"]],
            on="rebalance_date",
            how="inner",
        ).sort_values("rebalance_date")
        if merged.empty:
            warnings.warn(f"No overlapping dates for {factor_name} and {affected_asset} weights.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        signal_col = "z_score_used" if "z_score_used" in merged.columns else "factor_value_used"
        ax.plot(merged["rebalance_date"], merged[signal_col], label=f"{factor_name} z-score")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="gray", linewidth=0.7, linestyle=":")
        ax.axhline(-1, color="gray", linewidth=0.7, linestyle=":")
        ax.set_ylabel("China gold signal z-score" if factor_name == "custom_china_gold_to_us_gold" else "Factor z-score")
        ax2 = ax.twinx()
        ax2.plot(
            merged["rebalance_date"],
            merged["weight"],
            color="tab:orange",
            label=f"{affected_asset} weight",
        )
        ax2.set_ylabel(f"{affected_asset} weight")
        ax.set_title(f"{factor_name}: signal vs {affected_asset} weight")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()

        file_name = (
            f"rolling_{safe_filename(factor_name)}_signal_vs_"
            f"{safe_filename(affected_asset)}_weight.png"
        )
        fig.savefig(charts_dir / file_name, dpi=150)
        if factor_name == "custom_china_gold_to_us_gold" and affected_asset == "Gold":
            fig.savefig(charts_dir / "rolling_custom_china_gold_signal_vs_Gold_weight.png", dpi=150)
        plt.close(fig)
        generated.append(file_name)

    # Backward-compatible copy for older docs/tutorials when the original
    # China-gold-to-Gold view is active.
    legacy = "rolling_custom_china_gold_to_us_gold_signal_vs_Gold_weight.png"
    if legacy in generated:
        source = charts_dir / legacy
        target = charts_dir / "rolling_custom_factor_signal_vs_gold_weight.png"
        try:
            target.write_bytes(source.read_bytes())
        except OSError as exc:
            warnings.warn(f"Could not write legacy custom factor chart name: {exc}")


def monthly_heatmap(sample: pd.DataFrame):
    """Plot monthly returns heatmap-like image for BL Conservative."""
    s = sample[sample["strategy"] == "BL Conservative"].set_index("date")["portfolio_value_net"].resample("ME").last().pct_change().dropna()
    table = s.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.fillna(0).values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("BL Conservative Monthly Returns Heatmap")
    fig.colorbar(im, ax=ax)
    return ax


def risk_event_plot(risk_events: pd.DataFrame):
    """Plot risk events or an empty timeline note."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if risk_events.empty:
        ax.text(0.5, 0.5, "Risk overlay disabled: no risk events", ha="center", va="center")
        ax.set_axis_off()
    else:
        events = risk_events.copy()
        events["date"] = pd.to_datetime(events["date"])
        ax.scatter(events["date"], events["strategy"], marker="x")
        ax.set_title("Risk Events Timeline")
    return ax


def write_report(summary: pd.DataFrame, start_date, end_date, output_files: list[str], chart_files: list[str]) -> Path:
    """Write Stage 3 Markdown report."""
    reports = BASE_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    best_return = summary.sort_values("annualized_return_net", ascending=False).iloc[0]
    best_sharpe = summary.sort_values("sharpe_ratio_net", ascending=False).iloc[0]
    best_dd = summary.sort_values("max_drawdown", ascending=False).iloc[0]
    highest_cost = summary.sort_values("total_transaction_cost", ascending=False).iloc[0]
    robust = summary.sort_values(["max_drawdown", "sharpe_ratio_net"], ascending=[False, False]).iloc[0]
    report = f"""# Stage 3 Rolling Backtest Summary

## Goal

Validate static allocation methods out of sample with rolling covariance, rolling Black-Litterman inputs, rebalance schedules, transaction costs, and net equity curves.

## Data And Settings

- Data: `asset_returns.csv`, `clean_factor_panel.csv`, and optional `custom_factor_views.yaml`.
- Initial capital: ${INITIAL_CAPITAL:,.0f}.
- Backtest dates: {start_date} to {end_date}.
- Rolling windows: {ROLLING_WINDOWS}.
- Rebalance frequencies: {list(REBALANCE_FREQUENCIES.keys())}.
- Transaction cost: {TRANSACTION_COST_BPS} bps one-way, charged on turnover.
- Risk overlay enabled: {USE_RISK_OVERLAY}.

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

- Highest annualized net return: {best_return['strategy']} / window {best_return['window']} / {best_return['rebalance_frequency']}.
- Highest net Sharpe: {best_sharpe['strategy']} / window {best_sharpe['window']} / {best_sharpe['rebalance_frequency']}.
- Lowest max drawdown: {best_dd['strategy']} / window {best_dd['window']} / {best_dd['rebalance_frequency']}.
- Highest transaction cost: {highest_cost['strategy']} / window {highest_cost['window']} / {highest_cost['rebalance_frequency']}.
- Robust candidate by drawdown then Sharpe: {robust['strategy']} / window {robust['window']} / {robust['rebalance_frequency']}.

Custom BL performance does not prove the China gold factor is valid. Good performance may come from portfolio construction, covariance, or macro views; poor performance may indicate weak scenario relevance or unstable factor timing.

## Outputs

Files:

{chr(10).join('- `' + f + '`' for f in output_files)}

Charts:

{chr(10).join('- `' + f + '`' for f in chart_files)}
"""
    path = reports / "stage3_rolling_backtest_summary.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    """Run the complete Stage 3 rolling backtest."""
    returns, factors = load_inputs()
    all_equity, all_positions, all_trades, all_returns, all_turnover = [], [], [], [], []
    all_weights, all_view_logs, all_custom_logs, all_risk_contrib = [], [], [], []
    previous_by_key = {}

    for window in ROLLING_WINDOWS:
        for freq_name, freq_alias in REBALANCE_FREQUENCIES.items():
            rebalance_dates = get_rebalance_dates(returns.index, window, freq_alias)
            weights_by_date = {}
            for rebalance_date in rebalance_dates:
                hist = returns.loc[returns.index < rebalance_date].tail(window)
                if len(hist) < window:
                    continue
                prev = previous_by_key.setdefault((window, freq_name), {})
                weights, view_logs, custom_logs, risk_rows = build_weights_for_date(hist, factors, rebalance_date, prev, window, freq_name)
                weights_by_date[rebalance_date] = weights
                for strategy, w in weights.items():
                    prev[strategy] = w
                    for asset, weight in w.items():
                        all_weights.append({"rebalance_date": rebalance_date, "strategy": strategy, "window": window, "rebalance_frequency": freq_name, "asset": asset, "weight": weight})
                all_view_logs.extend(view_logs)
                all_custom_logs.extend(custom_logs)
                all_risk_contrib.extend(risk_rows)
            if weights_by_date:
                eq, pos, tr, ret, to = simulate_path(returns, weights_by_date, window, freq_name)
                all_equity.extend(eq)
                all_positions.extend(pos)
                all_trades.extend(tr)
                all_returns.extend(ret)
                all_turnover.extend(to)

    equity = pd.DataFrame(all_equity)
    positions = pd.DataFrame(all_positions)
    trades = pd.DataFrame(all_trades)
    returns_out = pd.DataFrame(all_returns)
    turnover = pd.DataFrame(all_turnover)
    weights = pd.DataFrame(all_weights)
    view_log = pd.DataFrame(all_view_logs)
    custom_log = pd.DataFrame(all_custom_logs)
    risk_contrib = pd.DataFrame(all_risk_contrib)
    risk_events = pd.DataFrame(columns=["date", "strategy", "window", "rebalance_frequency", "event_type", "affected_asset", "portfolio_return_since_rebalance", "asset_return_since_rebalance", "drawdown", "trigger_threshold", "action_taken", "cooldown_until"])

    equity["date"] = pd.to_datetime(equity["date"])
    returns_out["date"] = pd.to_datetime(returns_out["date"])
    trades["date"] = pd.to_datetime(trades["date"])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])
    summary = performance_summary(equity, returns_out, trades, turnover, risk_events)
    drawdowns = equity[["date", "strategy", "window", "rebalance_frequency", "drawdown", "peak_portfolio_value", "portfolio_value_net"]].copy()

    outputs = {
        "rolling_backtest_returns.csv": returns_out,
        "rolling_backtest_weights.csv": weights,
        "rolling_backtest_turnover.csv": turnover,
        "rolling_backtest_performance_summary.csv": summary,
        "rolling_backtest_drawdowns.csv": drawdowns,
        "rolling_backtest_risk_contribution.csv": risk_contrib,
        "rolling_backtest_view_log.csv": view_log,
        "rolling_backtest_custom_factor_log.csv": custom_log,
        "rolling_backtest_equity_curve.csv": equity,
        "rolling_backtest_positions.csv": positions,
        "rolling_backtest_trades.csv": trades,
        "rolling_backtest_risk_events.csv": risk_events,
    }
    for name, df in outputs.items():
        output_path = PROCESSED_DATA_DIR / name
        try:
            df.to_csv(output_path, index=False)
        except PermissionError as exc:
            raise PermissionError(
                f"无法写入 {output_path}。请先关闭正在打开这个 CSV 的 Excel、VS Code 预览、"
                "Power BI 或其他程序，然后重新运行 python src/run_rolling_backtest_bl.py。"
            ) from exc

    config_summary = pd.DataFrame(
        [
            {"parameter": "initial_capital", "value": INITIAL_CAPITAL, "description": "Starting portfolio capital in USD"},
            {"parameter": "rolling_windows", "value": ROLLING_WINDOWS, "description": "Rolling estimation windows in trading days"},
            {"parameter": "rebalance_frequencies", "value": list(REBALANCE_FREQUENCIES.keys()), "description": "Supported rebalance schedules"},
            {"parameter": "transaction_cost_bps", "value": TRANSACTION_COST_BPS, "description": "One-way transaction cost charged on turnover"},
            {"parameter": "use_risk_overlay", "value": USE_RISK_OVERLAY, "description": "Optional stop-loss/take-profit overlay"},
            {"parameter": "risk_overlay_config", "value": RISK_OVERLAY_CONFIG, "description": "Overlay config; inactive when disabled"},
            {"parameter": "custom_factor_views", "value": "enabled if YAML use_custom_factor_views=true", "description": "Custom factors only create low-confidence BL views"},
            {"parameter": "lookahead_rule", "value": "past_only", "description": "Rebalance uses only data before rebalance date"},
        ]
    )
    config_path = PROCESSED_DATA_DIR / "rolling_backtest_config_summary.csv"
    try:
        config_summary.to_csv(config_path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {config_path}。请关闭正在打开该文件的程序后重新运行。"
        ) from exc

    create_charts(equity, summary, weights, turnover, custom_log, risk_events)
    chart_files = sorted([p.name for p in (PROCESSED_DATA_DIR / "charts").glob("rolling_*.png")] + [p.name for p in (PROCESSED_DATA_DIR / "charts").glob("equity_curve_*.png")] + [p.name for p in (PROCESSED_DATA_DIR / "charts").glob("drawdown_curve.png")] + [p.name for p in (PROCESSED_DATA_DIR / "charts").glob("monthly_returns_heatmap.png")] + [p.name for p in (PROCESSED_DATA_DIR / "charts").glob("risk_events_timeline.png")] + [p.name for p in (PROCESSED_DATA_DIR / "charts").glob("with_vs_without_*.png")])
    report_path = write_report(summary, equity["date"].min().date(), equity["date"].max().date(), list(outputs.keys()) + ["rolling_backtest_config_summary.csv"], chart_files)

    print("\nRolling Backtest Summary")
    print(f"Start date: {equity['date'].min().date()}")
    print(f"End date: {equity['date'].max().date()}")
    print(f"Assets: {ASSET_ORDER}")
    print(f"Rolling windows: {ROLLING_WINDOWS}")
    print(f"Rebalance frequencies: {list(REBALANCE_FREQUENCIES.keys())}")
    print(f"Custom factor views enabled: {not custom_log.empty and custom_log['view_status'].eq('active').any()}")
    print(f"Risk overlay enabled: {USE_RISK_OVERLAY}")
    final = summary.sort_values(["window", "rebalance_frequency", "strategy"])
    print(final[["strategy", "window", "rebalance_frequency", "final_portfolio_value_net", "annualized_return_net", "sharpe_ratio_net", "max_drawdown"]].to_string(index=False))
    print(f"Output files: {list(outputs.keys()) + ['rolling_backtest_config_summary.csv']}")
    print(f"Charts: {chart_files}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
