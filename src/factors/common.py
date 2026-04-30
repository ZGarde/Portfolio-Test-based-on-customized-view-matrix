"""因子模块共用工具函数。

这里放所有跨资产都会用到的基础计算，例如列检查、滚动相关、
滚动动量、滚动 z-score、收益率计算、下一期收益和极值裁剪。
各资产因子文件只保留具体经济逻辑。
"""

import warnings

import numpy as np
import pandas as pd


def check_required_columns(df: pd.DataFrame, columns: list[str], module_name: str) -> bool:
    """检查构造因子所需的输入列是否存在。

    如果缺少列，函数只发出 warning 并返回 False，让调用方可以跳过
    对应因子，而不是让整个流水线因为 KeyError 中断。
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        warnings.warn(f"{module_name}: missing required columns: {missing}", stacklevel=2)
        return False
    return True


def safe_rolling_corr(
    left: pd.Series,
    right: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """安全计算两个序列的滚动相关系数。

    先按日期对齐并去掉缺失值，再在真实观测点上计算 rolling corr，
    最后 reindex 回原始日期，避免非交易日和缺失值污染窗口。
    """
    if min_periods is None:
        min_periods = max(2, window // 2)

    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.empty:
        return pd.Series(index=left.index, dtype="float64")

    corr = aligned.iloc[:, 0].rolling(window=window, min_periods=min_periods).corr(
        aligned.iloc[:, 1]
    )
    return corr.reindex(left.index)


def safe_rolling_sum(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """安全计算滚动求和，常用于多日动量因子。"""
    if min_periods is None:
        min_periods = window

    observed = series.dropna()
    if observed.empty:
        return pd.Series(index=series.index, dtype="float64")

    result = observed.rolling(window=window, min_periods=min_periods).sum()
    return result.reindex(series.index)


def safe_rolling_zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 126,
) -> pd.Series:
    """计算滚动 z-score，用来衡量当前值相对近期历史是否异常。

    z-score = (当前值 - 滚动均值) / 滚动标准差。
    例如 z-score 接近 2，表示当前值约高于近期均值 2 个标准差。
    """
    observed = series.replace([np.inf, -np.inf], np.nan).dropna()
    if observed.empty:
        return pd.Series(index=series.index, dtype="float64")

    rolling_mean = observed.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = observed.rolling(window=window, min_periods=min_periods).std()
    zscore = ((observed - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
    return zscore.reindex(series.index)


def simple_return(series: pd.Series) -> pd.Series:
    """根据价格序列计算简单收益率，并对齐回原日期。"""
    observed = series.replace([np.inf, -np.inf], np.nan).dropna()
    if observed.empty:
        return pd.Series(index=series.index, dtype="float64")
    return observed.pct_change(fill_method=None).reindex(series.index)


def build_simple_return_panel(price_panel: pd.DataFrame, suffix: str = "_ret") -> pd.DataFrame:
    """对价格面板的每一列计算简单收益率。"""
    returns = pd.DataFrame(index=price_panel.index)
    for col in price_panel.columns:
        returns[f"{col}{suffix}"] = simple_return(price_panel[col])
    return returns


def next_observed_return(series: pd.Series) -> pd.Series:
    """把下一次可观测收益对齐到当前日期，用作下一期预测目标。"""
    observed = series.replace([np.inf, -np.inf], np.nan).dropna()
    return observed.shift(-1).reindex(series.index)


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位数裁剪极端值，降低异常点对因子检验的影响。"""
    clean = series.replace([np.inf, -np.inf], np.nan)
    if clean.dropna().empty:
        return clean
    lo, hi = clean.quantile([lower, upper])
    return clean.clip(lo, hi)


def safe_quantile_test(
    df: pd.DataFrame,
    factor_col: str,
    target_col: str,
    q: int = 5,
) -> pd.DataFrame:
    """做一个简单的因子分位数组测试。

    输出每个分位数组的目标收益均值和样本数。这个检验只能作为初步
    研究证据，不能单独证明因子可交易。
    """
    if not check_required_columns(df, [factor_col, target_col], "safe_quantile_test"):
        return pd.DataFrame()

    data = df[[factor_col, target_col]].dropna().copy()
    if data.empty or data[factor_col].nunique() < 2:
        warnings.warn(
            f"safe_quantile_test: insufficient variation for {factor_col}",
            stacklevel=2,
        )
        return pd.DataFrame()

    try:
        data["quantile"] = pd.qcut(
            data[factor_col],
            q=q,
            labels=False,
            duplicates="drop",
        )
    except ValueError as exc:
        warnings.warn(f"safe_quantile_test failed for {factor_col}: {exc}", stacklevel=2)
        return pd.DataFrame()

    result = (
        data.groupby("quantile", observed=True)[target_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    result.insert(0, "target_col", target_col)
    result.insert(0, "factor_col", factor_col)
    return result
