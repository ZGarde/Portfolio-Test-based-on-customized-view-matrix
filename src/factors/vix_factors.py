"""VIX 相关因子。

VIX 常被视为市场恐慌指数。VIX 上升通常表示风险偏好下降，
对 Nasdaq、BTC 等风险资产可能不利。
"""

import pandas as pd

from factors.common import check_required_columns, safe_rolling_sum


def add_vix_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加 VIX 变化和短期 VIX 动量因子。"""
    if not check_required_columns(df, ["vix_level"], "add_vix_factors"):
        return df

    df["vix_change"] = df["vix_level"].diff()
    df["vix_mom_5d"] = safe_rolling_sum(df["vix_change"], 5)
    return df
