"""原油相关因子。

原油既有自身趋势，也会受到风险情绪和商品周期影响。VIX 上升通常代表
风险偏好下降，可能影响大宗商品风险溢价。
"""

import pandas as pd

from factors.common import check_required_columns, safe_rolling_sum


def add_oil_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加原油动量、VIX 风险情绪和交互项因子。"""
    if "vix_change" in df.columns:
        df["oil_vix_factor"] = df["vix_change"]
    else:
        check_required_columns(df, ["vix_change"], "add_oil_factors")

    if "oil" in df.columns:
        df["oil_mom_5d"] = safe_rolling_sum(df["oil"], 5)
        df["oil_mom_20d"] = safe_rolling_sum(df["oil"], 20)
    else:
        check_required_columns(df, ["oil"], "add_oil_factors")

    if check_required_columns(df, ["oil", "vix_change"], "add_oil_factors"):
        df["oil_vix_interaction"] = df["oil"] * df["vix_change"]

    return df
