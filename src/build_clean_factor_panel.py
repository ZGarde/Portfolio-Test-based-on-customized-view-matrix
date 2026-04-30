"""Build the clean daily factor panel and factor quality reports.

This is the single Stage 1 factor pipeline. It uses only the data frequency we
actually have: daily prices plus China gold morning/evening fields. There is no
hour-level US gold reopen/open check here because the raw data cannot support it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from factors.bond_factors import add_bond_factors
from factors.btc_factors import add_btc_factors
from factors.equity_factors import add_equity_factors
from factors.gold_factors import add_china_gold_returns, add_gold_factors
from factors.common import next_observed_return, simple_return, winsorize_series
from factors.oil_factors import add_oil_factors
from factors.vix_factors import add_vix_factors
from multi_asset_factor_project.src.factors.market_factors import add_market_cross_asset_factors


DATA_SOURCE_WARNING = (
    "China gold data is treated as a daily proxy. Raw data does not contain "
    "hour-level US gold reopen/open prices."
)
CHINA_GOLD_PROXY_NOTE = (
    "Daily proxy test only: same-day and next-observed close-to-close returns "
    "are used because the raw data has no hour-level US gold prices."
)


@dataclass(frozen=True)
class FactorSpec:
    factor_name: str
    target_col: str
    target_asset: str


FACTOR_SPECS = [
    FactorSpec("gold_china_to_us_factor", "gold_next_ret", "gold_us"),


    FactorSpec("vix_change", "nasdaq_next_ret", "nasdaq"),

    FactorSpec("oil_vix_factor", "oil_next_ret", "oil"),
    FactorSpec("oil_vix_interaction", "oil_next_ret", "oil"),
    FactorSpec("oil_mom_5d", "oil_next_ret", "oil"),
    FactorSpec("oil_mom_20d", "oil_next_ret", "oil"),
    FactorSpec("oil_winsorized_return", "oil_next_ret", "oil"),
    FactorSpec("oil_dollar_change_return", "oil_next_ret", "oil"),
    FactorSpec("btc_nasdaq_corr_60d", "btc_next_ret", "btc"),
    FactorSpec("btc_vix_corr_60d", "btc_next_ret", "btc"),
    FactorSpec("btc_vix_corr_20d", "btc_next_ret", "btc"),
    FactorSpec("btc_vix_corr_60d_z", "btc_next_ret", "btc"),
    FactorSpec("btc_gold_corr_60d", "btc_next_ret", "btc"),
    FactorSpec("vix_change_to_btc_next_factor", "btc_next_ret", "btc"),
    FactorSpec("vix_change_to_btc_next_z", "btc_next_ret", "btc"),
    FactorSpec("btc_vix_stress_factor", "btc_next_ret", "btc"),
    FactorSpec("btc_mom_5d", "btc_next_ret", "btc"),
    FactorSpec("btc_mom_20d", "btc_next_ret", "btc"),
    FactorSpec("risk_on_factor", "nasdaq_next_ret", "nasdaq"),
    FactorSpec("nasdaq_mom_5d", "nasdaq_next_ret", "nasdaq"),
    FactorSpec("nasdaq_mom_20d", "nasdaq_next_ret", "nasdaq"),


    FactorSpec("sse_to_nasdaq_signal_z", "nasdaq_next_ret", "nasdaq")

]


def read_date_csv(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()


def build_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    returns = pd.DataFrame(index=prices.index)
    for col in ["gold_us", "nasdaq", "treasury", "btc"]:
        if col in prices.columns:
            returns[col] = simple_return(prices[col])

    if "oil" in prices.columns:
        observed_oil = prices["oil"].dropna()
        returns["oil"] = simple_return(prices["oil"])
        returns["oil_price"] = prices["oil"]
        returns["oil_dollar_change"] = observed_oil.diff().reindex(prices.index)
        returns["oil_dollar_change_return"] = (
            observed_oil.diff() / observed_oil.shift(1).abs()
        ).reindex(prices.index)
        returns["oil_winsorized_return"] = winsorize_series(returns["oil"])

    if "vix_level" in prices.columns:
        returns["vix_level"] = prices["vix_level"]
    return returns


def add_fred_levels_and_changes(df: pd.DataFrame, fred_rates: pd.DataFrame) -> pd.DataFrame:
    fred_levels = fred_rates.ffill().reindex(df.index).ffill()
    for col in fred_rates.columns:
        df[col] = fred_levels[col]
        observed_changes = fred_rates[col].dropna().diff()
        df[f"{col}_observed_change"] = observed_changes.reindex(df.index)
    if {"DGS10", "DGS2"}.issubset(fred_rates.columns):
        curve = (fred_rates["DGS10"] - fred_rates["DGS2"]).dropna().diff()
        df["yield_curve_observed_change"] = curve.reindex(df.index)
    return df


def paired_stats(data: pd.DataFrame, factor_col: str, target_col: str) -> dict:
    clean = data[[factor_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "non_null_count": len(clean),
        "pearson_corr": np.nan,
        "spearman_corr": np.nan,
        "IC_mean": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "rolling_252d_IC_mean": np.nan,
        "rolling_252d_IC_positive_ratio": np.nan,
        "q1_return": np.nan,
        "q2_return": np.nan,
        "q3_return": np.nan,
        "q4_return": np.nan,
        "q5_return": np.nan,
        "q5_minus_q1_spread": np.nan,
        "q5_minus_q1_t_stat": np.nan,
        "monotonic_quantile_pass": False,
    }
    if len(clean) < 3 or clean[factor_col].nunique() < 2:
        return out

    pearson = stats.pearsonr(clean[factor_col], clean[target_col])
    spearman = stats.spearmanr(clean[factor_col], clean[target_col])
    out["pearson_corr"] = pearson.statistic
    out["spearman_corr"] = spearman.statistic
    out["p_value"] = spearman.pvalue
    out["IC_mean"] = spearman.statistic
    out["t_stat"] = pearson.statistic * math.sqrt(
        (len(clean) - 2) / max(1e-12, 1 - pearson.statistic**2)
    )

    rolling_ic = clean[factor_col].rolling(252).corr(clean[target_col])
    out["rolling_252d_IC_mean"] = rolling_ic.mean()
    out["rolling_252d_IC_positive_ratio"] = (rolling_ic > 0).mean()

    try:
        q_data = clean.copy()
        q_data["quantile"] = pd.qcut(q_data[factor_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return out

    grouped = q_data.groupby("quantile", observed=True)[target_col]
    means = grouped.mean()
    for q in range(5):
        if q in means.index:
            out[f"q{q + 1}_return"] = means.loc[q]
    if len(means) >= 2:
        out["q5_minus_q1_spread"] = means.iloc[-1] - means.iloc[0]
    if len(means) == 5:
        out["monotonic_quantile_pass"] = all(means.iloc[i] <= means.iloc[i + 1] for i in range(4))
    if 0 in q_data["quantile"].values and q_data["quantile"].max() in q_data["quantile"].values:
        q1 = q_data.loc[q_data["quantile"] == 0, target_col]
        q5 = q_data.loc[q_data["quantile"] == q_data["quantile"].max(), target_col]
        if len(q1) >= 5 and len(q5) >= 5:
            out["q5_minus_q1_t_stat"] = stats.ttest_ind(q5, q1, equal_var=False, nan_policy="omit").statistic
    return out


def out_of_sample_stats(data: pd.DataFrame, factor_col: str, target_col: str) -> tuple[float, float]:
    clean = data[[factor_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 504:
        return np.nan, np.nan
    test = clean.iloc[int(len(clean) * 0.70) :]
    if len(test) < 30 or test[factor_col].nunique() < 2:
        return np.nan, np.nan
    ic = stats.spearmanr(test[factor_col], test[target_col]).statistic
    try:
        q_data = test.copy()
        q_data["quantile"] = pd.qcut(q_data[factor_col], 5, labels=False, duplicates="drop")
        means = q_data.groupby("quantile", observed=True)[target_col].mean()
        spread = means.iloc[-1] - means.iloc[0] if len(means) >= 2 else np.nan
    except ValueError:
        spread = np.nan
    return ic, spread


def build_clean_factor_panel() -> pd.DataFrame:
    prices = read_date_csv(RAW_DATA_DIR / "us_asset_prices.csv")
    fred = read_date_csv(RAW_DATA_DIR / "fred_rates.csv")
    china_gold = add_china_gold_returns(read_date_csv(RAW_DATA_DIR / "china_gold_sge.csv"))
    #new 

    market_panel=read_date_csv(PROCESSED_DATA_DIR / "market_analysis" / "market_factor_panel.csv")

    panel = build_returns_from_prices(prices)
    panel = add_fred_levels_and_changes(panel, fred)

    market_cols = [
    "us_cn_nasdaq_sse_corr_60d",
    "us_cn_sp500_csi300_corr_60d",
    "china_us_equity_relative_strength_20d",
    "usdcny_csi300_corr_60d",
    "usdcny_sp500_corr_60d",
    "usdcny_mom_20d",
    "usdcny_z_252d",
    "china_equity_mom_20d",
    "china_equity_z_252d",
    "us_equity_mom_20d",
    "us_equity_z_252d",
    "us10y_china_equity_corr_60d",
    "sse_to_nasdaq_signal",
    "sse_to_nasdaq_signal_z",
]
    market_cols = [col for col in market_cols if col in market_panel.columns]
    panel = panel.join(market_panel[market_cols], how="left")

    china_cols = [
        col
        for col in [
            "china_gold_morning",
            "china_gold_evening",
            "china_gold_morning_ret",
            "china_gold_evening_ret",
            "china_gold_evening_to_evening_momentum",
            "china_gold_morning_to_evening_ret",
            "china_gold_morning_to_evening_z",
        ]
        if col in china_gold.columns
    ]
    panel = panel.join(china_gold[china_cols], how="left")

    panel = add_bond_factors(panel)
    panel = add_vix_factors(panel)
    panel = add_gold_factors(panel)
    panel = add_btc_factors(panel)
    panel = add_oil_factors(panel)
    panel = add_equity_factors(panel)

    for asset in ["gold_us", "treasury", "btc", "oil", "nasdaq"]:
        if asset in panel.columns:
            target_name = "gold_next_ret" if asset == "gold_us" else f"{asset}_next_ret"
            panel[target_name] = next_observed_return(panel[asset])
    if "gold_us" in panel.columns:
        panel["gold_same_day_ret"] = panel["gold_us"]

    return panel


def build_quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FACTOR_SPECS:
        if spec.factor_name not in panel.columns or spec.target_col not in panel.columns:
            rows.append(
                {
                    "factor_name": spec.factor_name,
                    "target_asset": spec.target_asset,
                    "non_null_count": 0,
                    "missing_rate": 1.0,
                    "pass_or_fail": "fail",
                    "failure_reason": "missing_factor_or_target_column",
                }
            )
            continue

        factor = panel[spec.factor_name]
        data = panel[[spec.factor_name, spec.target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        stat = paired_stats(panel, spec.factor_name, spec.target_col)
        missing_rate = 1 - factor.notna().mean()
        failure = []
        if factor.notna().sum() == 0:
            failure.append("factor_all_nan")
        if missing_rate > 0.75:
            failure.append("missing_rate_above_75pct")
        if stat["non_null_count"] < 252:
            failure.append("less_than_252_valid_observations")
        if pd.notna(stat["q5_minus_q1_spread"]) and stat["q5_minus_q1_spread"] < 0:
            failure.append("negative_q5_minus_q1_spread_direction_may_need_reverse")

        rows.append(
            {
                "factor_name": spec.factor_name,
                "target_asset": spec.target_asset,
                "non_null_count": int(factor.notna().sum()),
                "missing_rate": missing_rate,
                "start_date": data.index.min().date().isoformat() if len(data) else "",
                "end_date": data.index.max().date().isoformat() if len(data) else "",
                "pearson_corr": stat["pearson_corr"],
                "spearman_corr": stat["spearman_corr"],
                "IC_mean": stat["IC_mean"],
                "rolling_252d_IC_mean": stat["rolling_252d_IC_mean"],
                "rolling_252d_IC_positive_ratio": stat["rolling_252d_IC_positive_ratio"],
                "q5_minus_q1_spread": stat["q5_minus_q1_spread"],
                "t_stat": stat["t_stat"],
                "p_value": stat["p_value"],
                "pass_or_fail": "pass" if not failure else "fail",
                "failure_reason": ";".join(failure),
                "lookahead_bias_check": CHINA_GOLD_PROXY_NOTE
                if spec.factor_name == "gold_china_to_us_factor"
                else "",
                "tradability_check": DATA_SOURCE_WARNING
                if spec.factor_name == "gold_china_to_us_factor"
                else "",
                "gold_alignment_method": "daily_same_day_and_next_observed_proxy"
                if spec.factor_name == "gold_china_to_us_factor"
                else "",
            }
        )
    return pd.DataFrame(rows)


def build_clean_quantile_test(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in FACTOR_SPECS:
        if spec.factor_name not in panel.columns or spec.target_col not in panel.columns:
            continue
        data = panel[[spec.factor_name, spec.target_col]].dropna()
        if len(data) < 20 or data[spec.factor_name].nunique() < 2:
            continue
        try:
            data = data.copy()
            data["quantile"] = pd.qcut(data[spec.factor_name], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        grouped = data.groupby("quantile", observed=True)[spec.target_col].agg(["mean", "count"]).reset_index()
        grouped.insert(0, "target_col", spec.target_col)
        grouped.insert(0, "factor_col", spec.factor_name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_china_gold_signal_alignment_report(panel: pd.DataFrame) -> pd.DataFrame:
    tests = [
        (
            "china_gold_morning_to_evening_z",
            "gold_same_day_ret",
            "same_day_close_to_close_proxy",
            "China morning-to-evening signal tested against same-day US gold daily return",
        ),
        (
            "china_gold_morning_to_evening_z",
            "gold_next_ret",
            "next_observed_close_to_close_proxy",
            "China morning-to-evening signal tested against next observed US gold daily return",
        ),
        (
            "china_gold_evening_ret",
            "gold_same_day_ret",
            "same_day_close_to_close_proxy",
            "China evening-to-evening momentum benchmark",
        ),
        (
            "china_gold_evening_ret",
            "gold_next_ret",
            "next_observed_close_to_close_proxy",
            "China evening-to-evening momentum benchmark",
        ),
    ]
    rows = []
    for factor_name, target_name, alignment_type, note in tests:
        if factor_name not in panel.columns or target_name not in panel.columns:
            rows.append(
                {
                    "factor_name": factor_name,
                    "target_name": target_name,
                    "non_null_count": 0,
                    "pearson_corr": np.nan,
                    "spearman_corr": np.nan,
                    "q5_minus_q1_spread": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "alignment_type": alignment_type,
                    "note": f"missing factor or target column. {CHINA_GOLD_PROXY_NOTE}",
                }
            )
            continue
        stat = paired_stats(panel, factor_name, target_name)
        rows.append(
            {
                "factor_name": factor_name,
                "target_name": target_name,
                "non_null_count": stat["non_null_count"],
                "pearson_corr": stat["pearson_corr"],
                "spearman_corr": stat["spearman_corr"],
                "q5_minus_q1_spread": stat["q5_minus_q1_spread"],
                "t_stat": stat["t_stat"],
                "p_value": stat["p_value"],
                "alignment_type": alignment_type,
                "note": f"{note}. {CHINA_GOLD_PROXY_NOTE}",
            }
        )
    return pd.DataFrame(rows)


def print_conclusions(quality: pd.DataFrame, china_gold_signal_alignment: pd.DataFrame) -> None:
    keep = quality.loc[quality["pass_or_fail"] == "pass", "factor_name"].tolist()
    drop = quality.loc[quality["pass_or_fail"] == "fail", "factor_name"].tolist()
    reverse = quality[
        quality["failure_reason"].fillna("").str.contains("direction_may_need_reverse", regex=False)
    ]["factor_name"].tolist()

    print("\nClean factor quality conclusions")
    print(f"Keep candidates: {keep}")
    print(f"Delete or quarantine candidates: {drop}")
    print(f"Direction may need reverse: {reverse}")
    print(f"China gold note: {DATA_SOURCE_WARNING}")
    if not china_gold_signal_alignment.empty:
        print("China gold daily proxy alignment:")
        for _, row in china_gold_signal_alignment.iterrows():
            print(
                f"  {row['factor_name']} vs {row['target_name']}: "
                f"Spearman={row['spearman_corr']:.4f}, "
                f"q5-q1={row['q5_minus_q1_spread']:.6f}, "
                f"type={row['alignment_type']}."
            )


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_clean_factor_panel()
    quality = build_quality_report(panel)
    quantile = build_clean_quantile_test(panel)
    china_gold_signal_alignment = build_china_gold_signal_alignment_report(panel)

    panel.to_csv(PROCESSED_DATA_DIR / "clean_factor_panel.csv")
    panel.corr(numeric_only=True).to_csv(PROCESSED_DATA_DIR / "clean_factor_correlation.csv")
    quantile.to_csv(PROCESSED_DATA_DIR / "clean_factor_quantile_test.csv", index=False)
    quality.to_csv(PROCESSED_DATA_DIR / "factor_quality_report.csv", index=False)
    china_gold_signal_alignment.to_csv(
        PROCESSED_DATA_DIR / "china_gold_signal_alignment_report.csv",
        index=False,
    )

    print(f"Saved clean outputs to {PROCESSED_DATA_DIR}")
    print_conclusions(quality, china_gold_signal_alignment)


if __name__ == "__main__":
    main()
