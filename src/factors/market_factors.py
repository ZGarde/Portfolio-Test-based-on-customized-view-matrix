"""中美市场、汇率和利率相关的研究因子。

这个文件服务于 market_analysis.py，当前不直接进入主线组合回测。
它用于研究中美权益指数、人民币汇率和美国利率之间的联动关系。
"""

from __future__ import annotations

import pandas as pd

from factors.common import build_simple_return_panel, safe_rolling_corr, safe_rolling_sum, safe_rolling_zscore


def build_return_panel(price_panel: pd.DataFrame) -> pd.DataFrame:
    """把价格面板转换成收益率面板。"""
    return build_simple_return_panel(price_panel)


def add_market_cross_asset_factors(
    df: pd.DataFrame,
    corr_window: int = 60,
    momentum_window: int = 20,
    zscore_window: int = 252,
) -> pd.DataFrame:
    """添加中美市场和汇率相关的候选研究因子。

    输入应是收益率列，例如 sp500_ret、csi_300_ret、usd_cny_ret。
    如果某些列不存在，对应因子会被跳过，方便早期研究阶段逐步扩展。
    """
    out = df.copy()

    if {"sp500_ret", "csi_300_ret"}.issubset(out.columns):
        out["us_cn_sp500_csi300_corr_60d"] = safe_rolling_corr(
            out["sp500_ret"], out["csi_300_ret"], corr_window
        )
        out["china_us_equity_relative_strength_20d"] = safe_rolling_sum(
            out["csi_300_ret"] - out["sp500_ret"], momentum_window
        )

    if {"nasdaq_composite_ret", "chinext_ret"}.issubset(out.columns):
        out["us_cn_nasdaq_chinext_corr_60d"] = safe_rolling_corr(
            out["nasdaq_composite_ret"], out["chinext_ret"], corr_window
        )
        out["china_growth_vs_us_growth_relative_strength_20d"] = safe_rolling_sum(
            out["chinext_ret"] - out["nasdaq_composite_ret"], momentum_window
        )

    if {"nasdaq_composite_ret", "sse_index_ret"}.issubset(out.columns):
        out["us_cn_nasdaq_sse_corr_60d"] = safe_rolling_corr(
            out["nasdaq_composite_ret"], out["sse_index_ret"], corr_window
        )
        out["sse_to_nasdaq_signal"] = (
            out["sse_index_ret"] * out["us_cn_nasdaq_sse_corr_60d"]
        )
        out["sse_to_nasdaq_signal_z"] = safe_rolling_zscore(
            out["sse_to_nasdaq_signal"]
        )


        

    if {"usd_cny_ret", "csi_300_ret"}.issubset(out.columns):
        out["usdcny_csi300_corr_60d"] = safe_rolling_corr(
            out["usd_cny_ret"], out["csi_300_ret"], corr_window
        )

    if {"usd_cny_ret", "sp500_ret"}.issubset(out.columns):
        out["usdcny_sp500_corr_60d"] = safe_rolling_corr(
            out["usd_cny_ret"], out["sp500_ret"], corr_window
        )

    if "usd_cny_ret" in out.columns:
        out["usdcny_mom_20d"] = safe_rolling_sum(out["usd_cny_ret"], momentum_window)
        out["usdcny_z_252d"] = safe_rolling_zscore(out["usd_cny_ret"], zscore_window)

    if "csi_300_ret" in out.columns:
        out["china_equity_mom_20d"] = safe_rolling_sum(out["csi_300_ret"], momentum_window)
        out["china_equity_z_252d"] = safe_rolling_zscore(out["csi_300_ret"], zscore_window)

    if "sp500_ret" in out.columns:
        out["us_equity_mom_20d"] = safe_rolling_sum(out["sp500_ret"], momentum_window)
        out["us_equity_z_252d"] = safe_rolling_zscore(out["sp500_ret"], zscore_window)

    if {"us_10y_yield_ret", "csi_300_ret"}.issubset(out.columns):
        out["us10y_china_equity_corr_60d"] = safe_rolling_corr(
            out["us_10y_yield_ret"], out["csi_300_ret"], corr_window
        )

    return out


def build_market_research_factors(price_panel: pd.DataFrame) -> pd.DataFrame:
    """从价格面板生成 market analysis 专用研究因子面板。"""
    returns = build_return_panel(price_panel)
    factors = add_market_cross_asset_factors(returns)
    return pd.concat([price_panel, factors], axis=1)
