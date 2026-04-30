"""黄金相关因子。

黄金既受自身动量影响，也可能受到中国/美国交易时段信息传导影响。
当前中国黄金信号使用 morning-to-evening 日内代理收益的滚动 z-score，
用于描述中国交易时段内的黄金信息冲击。
"""

import numpy as np
import pandas as pd

from factors.common import check_required_columns, safe_rolling_sum, safe_rolling_zscore, simple_return


def add_china_gold_returns(china_gold: pd.DataFrame) -> pd.DataFrame:
    """根据中国黄金 morning/evening 价格构造日频代理信号。"""
    china = china_gold.copy()
    if "china_gold_morning" in china.columns:
        china["china_gold_morning_ret"] = simple_return(china["china_gold_morning"])
    if "china_gold_evening" in china.columns:
        china["china_gold_evening_ret"] = simple_return(china["china_gold_evening"])
        china["china_gold_evening_to_evening_momentum"] = china["china_gold_evening_ret"]
    if {"china_gold_morning", "china_gold_evening"}.issubset(china.columns):
        intraday = china["china_gold_evening"] / china["china_gold_morning"] - 1
        china["china_gold_morning_to_evening_ret"] = intraday.replace([np.inf, -np.inf], np.nan)
        china["china_gold_morning_to_evening_z"] = safe_rolling_zscore(
            china["china_gold_morning_to_evening_ret"]
        )
    return china


def add_gold_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加黄金跨市场联动和美国黄金动量因子。"""
    if "china_gold_morning_to_evening_z" in df.columns:
        df["gold_china_to_us_factor"] = df["china_gold_morning_to_evening_z"]
    elif "china_gold_evening_ret" in df.columns:
        df["gold_china_to_us_factor"] = df["china_gold_evening_ret"]
    else:
        check_required_columns(df, ["china_gold_morning_to_evening_z"], "add_gold_factors")

    if "china_gold_evening_ret" in df.columns:
        df["china_gold_evening_to_evening_momentum"] = df["china_gold_evening_ret"]

    if "gold_us" in df.columns:
        df["gold_us_to_china_factor"] = df["gold_us"]
        df["gold_mom_5d"] = safe_rolling_sum(df["gold_us"], 5)
        df["gold_mom_20d"] = safe_rolling_sum(df["gold_us"], 20)
    else:
        check_required_columns(df, ["gold_us"], "add_gold_factors")

    return df
