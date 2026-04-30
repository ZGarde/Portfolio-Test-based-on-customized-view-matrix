"""Strict factor review based on the clean daily factor panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from build_clean_factor_panel import FACTOR_SPECS, out_of_sample_stats, paired_stats
from config import PROCESSED_DATA_DIR


FOCUS_FACTORS = {
    "oil_vix_factor",
    "oil_mom_20d",
    "btc_nasdaq_corr_60d",
    "btc_vix_corr_20d",
    "btc_vix_corr_60d_z",
    "btc_gold_corr_60d",
    "vix_change_to_btc_next_factor",
    "vix_change_to_btc_next_z",
    "btc_vix_stress_factor",
    "btc_mom_5d",
    "btc_mom_20d",
    "vix_change",
    "oil_winsorized_return",
    "oil_dollar_change_return",
}


@dataclass(frozen=True)
class EconomicExpectation:
    expected_spread_sign: int
    rationale: str
    reversible: bool = True


EXPECTED_SIGNS = {
    "gold_china_to_us_factor": EconomicExpectation(1, "China gold daily proxy strength should map to US gold strength.", False),
    "gold_us_to_china_factor": EconomicExpectation(1, "US gold strength should map to future gold strength."),
    "gold_mom_5d": EconomicExpectation(1, "Gold momentum is expected to continue."),
    "gold_mom_20d": EconomicExpectation(1, "Gold medium-term momentum is expected to continue."),
    "bond_rate_factor": EconomicExpectation(1, "Falling rates should support Treasury ETF returns."),
    "bond_curve_factor": EconomicExpectation(1, "Falling curve pressure should support Treasury ETF returns."),
    "vix_change": EconomicExpectation(-1, "Rising VIX should pressure next Nasdaq return."),
    "vix_mom_5d": EconomicExpectation(-1, "Persistent VIX rise should pressure next Nasdaq return."),
    "oil_vix_factor": EconomicExpectation(-1, "Rising VIX should usually pressure oil risk appetite."),
    "oil_vix_interaction": EconomicExpectation(-1, "Oil return under rising VIX should be fragile."),
    "oil_mom_5d": EconomicExpectation(1, "Oil short-horizon momentum is expected to continue."),
    "oil_mom_20d": EconomicExpectation(1, "Oil medium-horizon momentum is expected to continue."),
    "oil_winsorized_return": EconomicExpectation(1, "Robust oil return is treated as a momentum proxy."),
    "oil_dollar_change_return": EconomicExpectation(1, "Oil dollar-change return is treated as a robust momentum proxy."),
    "btc_nasdaq_corr_60d": EconomicExpectation(1, "Higher BTC/Nasdaq coupling is a risk-on proxy for BTC."),
    "btc_vix_corr_60d": EconomicExpectation(-1, "Higher BTC/VIX coupling should be adverse for BTC."),
    "btc_vix_corr_20d": EconomicExpectation(-1, "Higher recent BTC/VIX coupling should be adverse for BTC."),
    "btc_vix_corr_60d_z": EconomicExpectation(-1, "Unusually high BTC/VIX coupling should be adverse for BTC."),
    "btc_gold_corr_60d": EconomicExpectation(1, "BTC/gold coupling is treated as a cross-asset support proxy."),
    "vix_change_to_btc_next_factor": EconomicExpectation(-1, "Rising VIX today is expected to pressure next observed BTC return."),
    "vix_change_to_btc_next_z": EconomicExpectation(-1, "Unusually large VIX increases are expected to pressure next observed BTC return."),
    "btc_vix_stress_factor": EconomicExpectation(-1, "VIX shocks under strong BTC/VIX coupling should be adverse for BTC."),
    "btc_mom_5d": EconomicExpectation(1, "BTC short-horizon momentum is expected to continue."),
    "btc_mom_20d": EconomicExpectation(1, "BTC medium-horizon momentum is expected to continue."),
    "risk_on_factor": EconomicExpectation(1, "Risk-on strength should support Nasdaq."),
    "nasdaq_mom_5d": EconomicExpectation(1, "Nasdaq short-horizon momentum is expected to continue."),
    "nasdaq_mom_20d": EconomicExpectation(1, "Nasdaq medium-horizon momentum is expected to continue."),
}


def sign(value: float) -> int:
    if pd.isna(value) or value == 0:
        return 0
    return 1 if value > 0 else -1


def q5_q1_t_stat(panel: pd.DataFrame, factor_col: str, target_col: str) -> float:
    data = panel[[factor_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30 or data[factor_col].nunique() < 2:
        return np.nan
    try:
        data = data.copy()
        data["quantile"] = pd.qcut(data[factor_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    q1 = data.loc[data["quantile"] == 0, target_col].dropna()
    q5 = data.loc[data["quantile"] == data["quantile"].max(), target_col].dropna()
    if len(q1) < 5 or len(q5) < 5:
        return np.nan
    return stats.ttest_ind(q5, q1, equal_var=False, nan_policy="omit").statistic


def monotonic_label(panel: pd.DataFrame, factor_col: str, target_col: str) -> tuple[bool, str]:
    data = panel[[factor_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 30 or data[factor_col].nunique() < 2:
        return False, "insufficient_data"
    try:
        data = data.copy()
        data["quantile"] = pd.qcut(data[factor_col], 5, labels=False, duplicates="drop")
    except ValueError:
        return False, "qcut_failed"
    means = data.groupby("quantile", observed=True)[target_col].mean().tolist()
    if len(means) < 5:
        return False, "less_than_5_quantiles"
    increasing = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    decreasing = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    if increasing:
        return True, "monotonic_increasing"
    if decreasing:
        return True, "monotonic_decreasing"
    return False, "non_monotonic"


def has_reasonable_missing_explanation(factor_name: str, missing_rate: float) -> tuple[bool, str]:
    if missing_rate <= 0.40:
        return True, ""
    us_calendar_factors = ("gold_", "bond_", "vix", "oil_", "nasdaq_", "risk_on")
    if factor_name.startswith(us_calendar_factors) and missing_rate <= 0.50:
        return True, "missingness explained by US trading calendar"
    if factor_name.startswith("btc_") and "corr" in factor_name and missing_rate <= 0.50:
        return True, "rolling cross-asset correlation requires overlapping trading days"
    return False, ""


def build_gold_daily_diagnostic(signal_alignment: pd.DataFrame) -> pd.DataFrame:
    same = signal_alignment[
        signal_alignment["factor_name"].eq("china_gold_morning_to_evening_z")
        & signal_alignment["target_name"].eq("gold_same_day_ret")
    ]
    nxt = signal_alignment[
        signal_alignment["factor_name"].eq("china_gold_morning_to_evening_z")
        & signal_alignment["target_name"].eq("gold_next_ret")
    ]
    same_row = same.iloc[0] if len(same) else pd.Series(dtype="object")
    next_row = nxt.iloc[0] if len(nxt) else pd.Series(dtype="object")

    return pd.DataFrame(
        [
            {
                "factor_name": "gold_china_to_us_factor",
                "gold_data_frequency": "daily",
                "uses_hour_level_timing_check": False,
                "target_return_is_daily_proxy": True,
                "primary_same_day_proxy_spearman_corr": same_row.get("spearman_corr", np.nan),
                "primary_same_day_proxy_q5_minus_q1": same_row.get("q5_minus_q1_spread", np.nan),
                "primary_next_observed_proxy_spearman_corr": next_row.get("spearman_corr", np.nan),
                "primary_next_observed_proxy_q5_minus_q1": next_row.get("q5_minus_q1_spread", np.nan),
                "lookahead_risk_level": "daily_proxy_only",
                "recommendation": (
                    "Use only as research-only daily proxy evidence; do not claim intraday "
                    "US gold open/reopen behavior from this dataset."
                ),
            }
        ]
    )


def strict_decision_for_factor(
    factor_name: str,
    old_row: pd.Series,
    panel: pd.DataFrame,
    target_col: str,
    gold_diag: pd.DataFrame,
) -> dict:
    stat = paired_stats(panel, factor_name, target_col)
    oos_ic, oos_spread = out_of_sample_stats(panel, factor_name, target_col)
    q_t = q5_q1_t_stat(panel, factor_name, target_col)
    monotonic_ok, monotonic_status = monotonic_label(panel, factor_name, target_col)

    non_null_count = int(old_row.get("non_null_count", stat["non_null_count"]))
    missing_rate = float(old_row.get("missing_rate", np.nan))
    old_pass = old_row.get("pass_or_fail", "")
    expected = EXPECTED_SIGNS.get(factor_name, EconomicExpectation(1, "No explicit rationale."))
    q_spread = stat["q5_minus_q1_spread"]
    q_sign = sign(q_spread)
    ic_sign = sign(stat["spearman_corr"])
    expected_sign = expected.expected_spread_sign

    missing_ok, missing_explanation = has_reasonable_missing_explanation(factor_name, missing_rate)
    direction_matches = q_sign == expected_sign if q_sign != 0 else False
    direction_unstable = ic_sign != 0 and q_sign != 0 and ic_sign != q_sign
    should_reverse = q_sign == -expected_sign and ic_sign == -expected_sign and expected.reversible
    p_value = stat["p_value"]
    significant = (
        (pd.notna(q_t) and abs(q_t) > 1.96)
        or (pd.notna(stat["t_stat"]) and abs(stat["t_stat"]) > 1.96)
    ) and pd.notna(p_value) and p_value < 0.05
    weak_p = pd.isna(p_value) or p_value > 0.10

    oos_direction_ok = True
    if pd.notna(oos_ic) and ic_sign != 0:
        oos_direction_ok = sign(oos_ic) in (0, ic_sign)
    oos_spread_ok = pd.notna(oos_spread) and abs(oos_spread) > 1e-8
    rolling_ok = pd.notna(stat["rolling_252d_IC_positive_ratio"]) and stat["rolling_252d_IC_positive_ratio"] >= 0.55

    reasons = []
    tags = []
    if non_null_count < 1500:
        reasons.append("non_null_count_below_1500")
    if not missing_ok:
        reasons.append("missing_rate_above_40pct_without_sufficient_explanation")
    elif missing_explanation:
        tags.append(missing_explanation)
    if direction_unstable:
        reasons.append("direction_unstable")
    if not direction_matches:
        reasons.append(
            "raw_direction_opposite_economic_meaning_reverse_required"
            if should_reverse
            else "q5_minus_q1_direction_inconsistent_with_economic_meaning"
        )
    if weak_p:
        reasons.append("p_value_above_0.10")
    elif not significant:
        reasons.append("not_significant_at_5pct")
    if not rolling_ok:
        reasons.append("rolling_252d_IC_positive_ratio_below_0.55")
    if not oos_direction_ok:
        reasons.append("out_of_sample_IC_direction_reversal")
    if not oos_spread_ok:
        reasons.append("out_of_sample_q5_minus_q1_spread_disappears")
    if not monotonic_ok and direction_matches:
        reasons.append("non_monotonic_candidate")

    gold_research_only = factor_name == "gold_china_to_us_factor"
    gd = gold_diag.iloc[0] if gold_research_only else pd.Series(dtype="object")
    if gold_research_only:
        reasons.append("research_only_daily_proxy_evidence")

    strict_pass = len(reasons) == 0
    weak_candidate = (not strict_pass) and (not weak_p) and significant and direction_matches
    research_only = gold_research_only or weak_candidate or (factor_name in FOCUS_FACTORS and not strict_pass)
    delete = not strict_pass and not research_only and (
        non_null_count < 1500 or (not missing_ok) or (not direction_matches and not should_reverse) or weak_p
    )

    if strict_pass:
        label = "strict_pass"
    elif gold_research_only:
        label = "research_only_daily_proxy"
    elif weak_candidate:
        label = "weak_candidate"
    else:
        label = "strict_fail"

    return {
        "factor_name": factor_name,
        "target_col": target_col,
        "target_asset": old_row.get("target_asset", ""),
        "old_pass_or_fail": old_pass,
        "strict_pass_or_fail": label,
        "downgrade_reason": ";".join(dict.fromkeys(reasons + tags)),
        "non_null_count": non_null_count,
        "missing_rate": missing_rate,
        "spearman_corr": stat["spearman_corr"],
        "q5_minus_q1_spread": q_spread,
        "IC_t_stat": stat["t_stat"],
        "q5_minus_q1_t_stat": q_t,
        "p_value": p_value,
        "rolling_252d_IC_positive_ratio": stat["rolling_252d_IC_positive_ratio"],
        "out_of_sample_IC": oos_ic,
        "out_of_sample_q5_minus_q1_spread": oos_spread,
        "expected_spread_sign": expected_sign,
        "economic_rationale": expected.rationale,
        "direction_unstable": direction_unstable,
        "quantile_monotonic_status": monotonic_status,
        "whether_can_enter_black_litterman": strict_pass,
        "whether_can_enter_trading_signal": strict_pass,
        "whether_should_be_reversed": should_reverse,
        "whether_should_be_deleted": delete,
        "whether_should_be_kept_for_research_only": research_only,
        "focus_factor_reviewed": factor_name in FOCUS_FACTORS,
        "lookahead_bias_risk": gd.get("lookahead_risk_level", "not_detected_from_available_daily_files"),
        "gold_data_frequency": gd.get("gold_data_frequency", ""),
        "gold_uses_hour_level_timing_check": gd.get("uses_hour_level_timing_check", np.nan),
        "primary_same_day_proxy_spearman_corr": gd.get("primary_same_day_proxy_spearman_corr", np.nan),
        "primary_next_observed_proxy_spearman_corr": gd.get("primary_next_observed_proxy_spearman_corr", np.nan),
        "gold_recommendation": gd.get("recommendation", ""),
    }


def build_strict_reports() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(PROCESSED_DATA_DIR / "clean_factor_panel.csv", parse_dates=["date"], index_col="date")
    old_quality = pd.read_csv(PROCESSED_DATA_DIR / "factor_quality_report.csv")
    signal_path = PROCESSED_DATA_DIR / "china_gold_signal_alignment_report.csv"
    signal_alignment = pd.read_csv(signal_path) if signal_path.exists() else pd.DataFrame()
    gold_diag = build_gold_daily_diagnostic(signal_alignment)

    old_by_factor = old_quality.set_index("factor_name")
    rows = []
    for spec in FACTOR_SPECS:
        if spec.factor_name not in old_by_factor.index:
            continue
        if spec.factor_name not in panel.columns or spec.target_col not in panel.columns:
            continue
        rows.append(
            strict_decision_for_factor(
                spec.factor_name,
                old_by_factor.loc[spec.factor_name],
                panel,
                spec.target_col,
                gold_diag,
            )
        )
    strict = pd.DataFrame(rows)
    decision = strict[
        [
            "factor_name",
            "old_pass_or_fail",
            "strict_pass_or_fail",
            "downgrade_reason",
            "whether_can_enter_black_litterman",
            "whether_can_enter_trading_signal",
            "whether_should_be_reversed",
            "whether_should_be_deleted",
            "whether_should_be_kept_for_research_only",
        ]
    ].copy()
    return strict, decision, gold_diag


def write_outputs(strict: pd.DataFrame, decision: pd.DataFrame, gold_diag: pd.DataFrame) -> None:
    strict.to_csv(PROCESSED_DATA_DIR / "strict_factor_quality_report.csv", index=False)
    decision.to_csv(PROCESSED_DATA_DIR / "strict_factor_decision_summary.csv", index=False)
    gold_diag.to_csv(PROCESSED_DATA_DIR / "gold_daily_proxy_diagnostic_summary.csv", index=False)

    strict[strict["whether_should_be_deleted"]].to_csv(PROCESSED_DATA_DIR / "rejected_factors.csv", index=False)
    strict[strict["whether_should_be_kept_for_research_only"]].to_csv(
        PROCESSED_DATA_DIR / "research_only_factors.csv",
        index=False,
    )
    strict[strict["strict_pass_or_fail"].eq("strict_pass")].to_csv(
        PROCESSED_DATA_DIR / "approved_candidate_factors.csv",
        index=False,
    )


def print_final_conclusions(strict: pd.DataFrame, gold_diag: pd.DataFrame) -> None:
    approved = strict.loc[strict["strict_pass_or_fail"].eq("strict_pass"), "factor_name"].tolist()
    research = strict.loc[strict["whether_should_be_kept_for_research_only"], "factor_name"].tolist()
    rejected = strict.loc[strict["whether_should_be_deleted"], "factor_name"].tolist()
    reverse = strict.loc[strict["whether_should_be_reversed"], "factor_name"].tolist()

    print("\nStrict factor review conclusions")
    print(f"1. Factors that can enter the next stage: {approved if approved else 'None under strict rules.'}")
    print(f"2. Research-only factors: {research}")
    print(f"3. Factors that should be deleted/quarantined: {rejected}")
    print(f"4. Factors that need sign reversal before reuse: {reverse}")
    print("5. China gold is reviewed only as daily proxy evidence; hour-level timing checks were removed.")
    print(f"Gold diagnostic recommendation: {gold_diag.iloc[0]['recommendation']}")


def main() -> None:
    strict, decision, gold_diag = build_strict_reports()
    write_outputs(strict, decision, gold_diag)
    print(f"Saved strict review outputs to {PROCESSED_DATA_DIR}")
    print_final_conclusions(strict, gold_diag)


if __name__ == "__main__":
    main()
