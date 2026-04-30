"""比特币相关因子。

BTC 既可能表现得像高风险资产，也可能在某些阶段表现得像另类资产。
这里主要观察 BTC 与 Nasdaq、VIX、黄金之间的滚动关系，以及 BTC 自身动量。
"""

import pandas as pd

from factors.common import (
    check_required_columns,
    safe_rolling_corr,
    safe_rolling_sum,
    safe_rolling_zscore,
)


def add_btc_factors(df: pd.DataFrame) -> pd.DataFrame:
    """添加 BTC 跨资产相关性、VIX 敏感度和动量因子。

    生成的主要因子：
    - btc_nasdaq_corr_60d：BTC 与 Nasdaq 的 60 日滚动相关。
    - btc_vix_corr_60d / btc_vix_corr_20d：BTC 与 VIX 变化的滚动相关。
    - vix_change_to_btc_next_factor：用当天 VIX 变化研究下一期 BTC 收益。
    - btc_vix_same_day_stress：同日解释变量，不应直接当作可交易预测信号。
    - btc_mom_5d / btc_mom_20d：BTC 短期和中期动量。
    """
    if "btc" not in df.columns:
        check_required_columns(df, ["btc"], "add_btc_factors")
        return df

    if "nasdaq" in df.columns:
        df["btc_nasdaq_corr_60d"] = safe_rolling_corr(df["btc"], df["nasdaq"], 60)
    else:
        check_required_columns(df, ["nasdaq"], "add_btc_factors")

    if "vix_change" in df.columns:
        # 预测型研究因子：今天的 VIX 冲击，用来检验下一期 BTC 收益。
        df["vix_change_to_btc_next_factor"] = df["vix_change"]
        df["vix_change_to_btc_next_z"] = safe_rolling_zscore(df["vix_change"])

        # 状态因子：最近一段时间 BTC 和 VIX 的耦合程度。
        df["btc_vix_corr_60d"] = safe_rolling_corr(df["btc"], df["vix_change"], 60)
        df["btc_vix_corr_20d"] = safe_rolling_corr(df["btc"], df["vix_change"], 20)
        df["btc_vix_corr_60d_z"] = safe_rolling_zscore(df["btc_vix_corr_60d"])

        # 同日压力代理变量，用于解释风险冲击环境，不是纯预测信号。
        df["btc_vix_same_day_stress"] = df["btc"] * df["vix_change"]
        df["btc_vix_stress_factor"] = df["vix_change"] * df["btc_vix_corr_60d"]
    else:
        check_required_columns(df, ["vix_change"], "add_btc_factors")

    if "gold_us" in df.columns:
        df["btc_gold_corr_60d"] = safe_rolling_corr(df["btc"], df["gold_us"], 60)
    else:
        check_required_columns(df, ["gold_us"], "add_btc_factors")

    df["btc_mom_5d"] = safe_rolling_sum(df["btc"], 5)
    df["btc_mom_20d"] = safe_rolling_sum(df["btc"], 20)
    return df
