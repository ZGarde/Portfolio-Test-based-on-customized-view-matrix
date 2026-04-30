"""债券和利率相关因子。

债券价格通常和利率方向相反：利率上升时，已有债券价格往往下跌；
利率下降时，债券价格往往受益。
"""

import pandas as pd

from factors.common import check_required_columns


def add_bond_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加债券/利率因子。

    生成的主要因子：
    - yield_curve_10y_2y：10 年期利率减 2 年期利率。
    - rate_10y_change / rate_2y_change：利率变化。
    - yield_curve_change：期限利差变化。
    - bond_rate_factor：利率下降对债券更友好，因此取 10 年期利率变化的负号。
    - bond_curve_factor：利差变化的负号，作为债券环境代理。
    """
    if not check_required_columns(df, ["DGS10", "DGS2"], "add_bond_factors"):
        return df

    df["yield_curve_10y_2y"] = df["DGS10"] - df["DGS2"]

    if "DGS10_observed_change" in df.columns:
        df["rate_10y_change"] = df["DGS10_observed_change"]
    else:
        df["rate_10y_change"] = df["DGS10"].dropna().diff().reindex(df.index)

    if "DGS2_observed_change" in df.columns:
        df["rate_2y_change"] = df["DGS2_observed_change"]
    else:
        df["rate_2y_change"] = df["DGS2"].dropna().diff().reindex(df.index)

    if "yield_curve_observed_change" in df.columns:
        df["yield_curve_change"] = df["yield_curve_observed_change"]
    else:
        df["yield_curve_change"] = df["yield_curve_10y_2y"].dropna().diff().reindex(df.index)

    df["bond_rate_factor"] = -df["rate_10y_change"]
    df["bond_curve_factor"] = -df["yield_curve_change"]
    return df
