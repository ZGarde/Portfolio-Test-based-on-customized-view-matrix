"""权益和 risk-on 相关因子。

Nasdaq 代表成长股/风险资产的一部分。如果 Nasdaq 相对 Treasury 表现更强，
通常可以理解为风险偏好更强。
"""

import pandas as pd

from factors.common import check_required_columns, safe_rolling_sum


def add_equity_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加 Nasdaq 动量和风险偏好因子。"""
    if check_required_columns(df, ["nasdaq", "treasury"], "add_equity_factors"):
        df["risk_on_factor"] = df["nasdaq"] - df["treasury"]

    if "nasdaq" in df.columns:
        df["nasdaq_mom_5d"] = safe_rolling_sum(df["nasdaq"], 5)
        df["nasdaq_mom_20d"] = safe_rolling_sum(df["nasdaq"], 20)
    else:
        check_required_columns(df, ["nasdaq"], "add_equity_factors")

    return df
