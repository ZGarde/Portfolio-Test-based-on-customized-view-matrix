"""Stage 2: 静态组合优化和 Black-Litterman 权重生成。

This stage intentionally does not run a rolling backtest. It creates static
weights, ex-ante return/risk estimates, Black-Litterman prior/posterior returns,
risk contributions, and diagnostic charts.

初学者理解：
    这个脚本不是回测，而是站在“当前整段样本”上做一次静态资产配置。
    它会先估计资产的年化收益/协方差，然后生成几类组合：
    - 等权组合
    - 最小方差组合
    - 最大夏普组合
    - 风险平价组合
    - 目标波动率组合
    - Black-Litterman 不带观点/保守观点/探索观点/custom factor 观点组合
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from config import BASE_DIR, PROCESSED_DATA_DIR


# 组合里允许配置的资产顺序。后面的权重向量、P 矩阵都按这个顺序排列。
ASSET_ORDER = ["Gold", "Treasury", "Bitcoin", "Oil", "Nasdaq"]
# clean_factor_panel.csv 中每个资产对应的收益率列名。
RETURN_COLUMNS = {
    "Gold": "gold_us",
    "Treasury": "treasury",
    "Bitcoin": "btc",
    "Oil": "oil",
    "Nasdaq": "nasdaq",
}
# 用作 Black-Litterman 均衡收益反推的“市场组合”代理权重。
MARKET_WEIGHTS = pd.Series(
    {"Gold": 0.20, "Treasury": 0.35, "Bitcoin": 0.05, "Oil": 0.15, "Nasdaq": 0.25}
)
# 每个资产的最大权重上限，避免优化器把仓位全部压到单一资产。
MAX_WEIGHTS = {"Gold": 0.50, "Treasury": 0.50, "Bitcoin": 0.15, "Oil": 0.25, "Nasdaq": 0.50}
TRADING_DAYS = 252
TARGET_VOL = 0.12
# TAU 是 Black-Litterman 里控制先验不确定性的参数；这里用常见的小值。
TAU = 0.05
RISK_FREE_RATE = 0.0
CUSTOM_FACTOR_CONFIG_PATH = BASE_DIR / "config" / "custom_factor_views.yaml"
CUSTOM_ALIGNMENT_ASSUMPTION = (
    "same-day China morning-to-evening signal; no mechanical shift(1) because hypothesis "
    "assumes only a few hours between China evening and US gold reaction."
)
CUSTOM_PROXY_WARNING = (
    "US gold open/reopen intraday price is unavailable; current test uses daily "
    "close-to-close proxy and cannot strictly prove US high-open behavior."
)
CUSTOM_TRIGGER_RULE = (
    "If z is NaN skip; if abs(z)<trigger_threshold set Q=0/confidence=inactive_confidence; "
    "otherwise set Q=clip(q_scale*z, -q_cap, q_cap), confidence=min(base_confidence + "
    "confidence_slope*(abs(z)-trigger_threshold), max_confidence)."
)
DEFAULT_CUSTOM_VIEW_PARAMS = {
    "trigger_threshold": 1.0,
    "inactive_confidence": 0.01,
    "q_scale": 0.01,
    "q_cap": 0.03,
    "base_confidence": 0.10,
    "confidence_slope": 0.10,
    "max_confidence": 0.30,
}


@dataclass(frozen=True)
class PortfolioResult:
    """保存一个静态组合的结果。

    字段:
        name: 组合名称。
        weights: 每个资产的权重。
        expected_returns: 这个组合使用的预期收益向量。
    """

    name: str
    weights: pd.Series
    expected_returns: pd.Series


def read_asset_returns() -> pd.DataFrame:
    """读取 clean factor panel，并提取组合优化需要的资产收益矩阵。

    返回:
        行是日期、列是资产的收益率 DataFrame。
    """
    path = PROCESSED_DATA_DIR / "clean_factor_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing clean factor panel: {path}")
    panel = pd.read_csv(path, parse_dates=["date"], index_col="date")
    returns = panel[[RETURN_COLUMNS[a] for a in ASSET_ORDER]].rename(
        columns={v: k for k, v in RETURN_COLUMNS.items()}
    )
    # Use common observed days for covariance optimization. BTC still comes from
    # calendar-day data upstream; optimization needs a common return matrix.
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    returns.to_csv(PROCESSED_DATA_DIR / "asset_returns.csv")
    return returns


def annualized_stats(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """计算年化收益、年化波动率、年化协方差矩阵和相关矩阵。

    初学者理解:
        - mean * 252 得到年化平均收益。
        - std * sqrt(252) 得到年化波动率。
        - cov * 252 得到年化协方差。
    """
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * math.sqrt(TRADING_DAYS)
    cov = returns.cov() * TRADING_DAYS
    corr = returns.corr()

    pd.DataFrame(
        {"annualized_return": ann_return, "annualized_volatility": ann_vol}
    ).to_csv(PROCESSED_DATA_DIR / "annualized_return_volatility.csv")
    cov.to_csv(PROCESSED_DATA_DIR / "covariance_matrix.csv")
    corr.to_csv(PROCESSED_DATA_DIR / "correlation_matrix.csv")
    return ann_return, ann_vol, cov, corr


def portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    """计算组合预期年化收益。

    公式:
        组合收益 = 权重向量 dot 资产预期收益向量。
    """
    return float(np.dot(weights, mu.values))


def portfolio_variance(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """计算组合年化方差。

    公式:
        w.T @ Cov @ w
    """
    return float(weights @ cov.values @ weights)


def portfolio_vol(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """计算组合年化波动率，也就是方差的平方根。"""
    return math.sqrt(max(portfolio_variance(weights, cov), 0.0))


def portfolio_sharpe(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame) -> float:
    """计算组合预期夏普比率。

    这里默认无风险利率为 0，所以夏普约等于收益 / 波动率。
    """
    vol = portfolio_vol(weights, cov)
    if vol <= 0:
        return np.nan
    return (portfolio_return(weights, mu) - RISK_FREE_RATE) / vol


def constraints() -> tuple[list[tuple[float, float]], dict]:
    """生成组合优化约束。

    返回:
        bounds: 每个资产的最小/最大权重。
        cons: 权重和必须等于 1 的约束。
    """
    # long-only: 每个资产权重不能小于 0，且不能超过 MAX_WEIGHTS。
    bounds = [(0.0, MAX_WEIGHTS[a]) for a in ASSET_ORDER]
    # full-investment: 所有权重加起来必须等于 1。
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    return bounds, cons


def feasible_initial_weights() -> np.ndarray:
    """生成一个满足约束的初始权重，用作 scipy 优化器起点。"""
    w = MARKET_WEIGHTS.reindex(ASSET_ORDER).values.astype(float)
    return w / w.sum()


def optimize_min_variance(cov: pd.DataFrame) -> pd.Series:
    """求最小方差组合。

    目标:
        在满足权重约束的情况下，让组合波动率/方差尽量低。
    """
    bounds, cons = constraints()
    x0 = feasible_initial_weights()
    result = minimize(lambda w: portfolio_variance(w, cov), x0, bounds=bounds, constraints=[cons])
    if not result.success:
        raise RuntimeError(f"Minimum variance optimization failed: {result.message}")
    return pd.Series(result.x, index=ASSET_ORDER)


def optimize_max_sharpe(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """求最大夏普组合。

    目标:
        在满足权重约束的情况下，让预期收益 / 风险 尽量高。
    """
    bounds, cons = constraints()
    x0 = feasible_initial_weights()
    result = minimize(
        lambda w: -portfolio_sharpe(w, mu, cov),
        x0,
        bounds=bounds,
        constraints=[cons],
    )
    if not result.success:
        raise RuntimeError(f"Maximum Sharpe optimization failed: {result.message}")
    return pd.Series(result.x, index=ASSET_ORDER)


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """计算每个资产对组合风险的贡献。

    输出字段:
        marginal_risk_contribution: 边际风险贡献。
        component_risk_contribution: 权重乘以边际贡献。
        percentage_risk_contribution: 占总组合风险的比例。
    """
    w = weights.reindex(ASSET_ORDER).values
    sigma_w = cov.values @ w
    variance = portfolio_variance(w, cov)
    vol = math.sqrt(max(variance, 0.0))
    mrc = sigma_w / vol if vol > 0 else np.full_like(w, np.nan)
    crc = w * sigma_w
    prc = crc / variance if variance > 0 else np.full_like(w, np.nan)
    return pd.DataFrame(
        {
            "asset": ASSET_ORDER,
            "weight": w,
            "marginal_risk_contribution": mrc,
            "component_risk_contribution": crc,
            "percentage_risk_contribution": prc,
        }
    )


def optimize_risk_parity(cov: pd.DataFrame) -> pd.Series:
    """求近似风险平价组合。

    目标:
        让每个资产贡献的风险比例尽量接近相等，而不是让资金权重相等。
    """
    bounds, cons = constraints()
    x0 = feasible_initial_weights()

    def objective(w: np.ndarray) -> float:
        rc = risk_contribution(pd.Series(w, index=ASSET_ORDER), cov)["percentage_risk_contribution"].values
        target = np.ones(len(ASSET_ORDER)) / len(ASSET_ORDER)
        return float(np.sum((rc - target) ** 2))

    result = minimize(objective, x0, bounds=bounds, constraints=[cons])
    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed: {result.message}")
    return pd.Series(result.x, index=ASSET_ORDER)


def optimize_target_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float = TARGET_VOL) -> tuple[pd.Series, str]:
    """求目标波动率组合。

    目标:
        在组合波动率不超过 target_vol 的前提下，最大化预期收益。
        如果目标波动率太低、连最小方差组合都达不到，就退回最小方差组合。
    """
    bounds, cons = constraints()
    x0 = optimize_min_variance(cov).values
    min_vol = portfolio_vol(x0, cov)
    if min_vol > target_vol:
        note = f"Target volatility {target_vol:.2%} infeasible; min variance vol is {min_vol:.2%}."
        return pd.Series(x0, index=ASSET_ORDER), note

    cons_list = [
        cons,
        {"type": "ineq", "fun": lambda w: target_vol**2 - portfolio_variance(w, cov)},
    ]
    result = minimize(
        lambda w: -portfolio_return(w, mu),
        x0,
        bounds=bounds,
        constraints=cons_list,
    )
    if not result.success:
        note = f"Target volatility optimization failed; using min variance. Reason: {result.message}"
        return pd.Series(x0, index=ASSET_ORDER), note
    return pd.Series(result.x, index=ASSET_ORDER), f"Target volatility <= {target_vol:.2%} satisfied."


def estimate_delta(mu: pd.Series, cov: pd.DataFrame) -> tuple[float, str]:
    """估计 Black-Litterman 风险厌恶系数 delta。

    作用:
        Black-Litterman 会用 delta、协方差和市场权重反推出均衡收益 Pi。
        如果估计值异常，就使用 2.5 作为稳健 fallback。
    """
    w_mkt = MARKET_WEIGHTS.reindex(ASSET_ORDER).values
    market_return = portfolio_return(w_mkt, mu)
    market_var = portfolio_variance(w_mkt, cov)
    if market_var <= 0:
        return 2.5, "delta fallback: non-positive market variance"
    delta = (market_return - RISK_FREE_RATE) / market_var
    if not np.isfinite(delta) or delta <= 0 or delta > 10:
        return 2.5, f"delta fallback: unstable estimate {delta:.4f}"
    return float(delta), f"delta estimated from market proxy: {delta:.4f}"


def implied_equilibrium_returns(cov: pd.DataFrame, delta: float) -> pd.Series:
    """计算 Black-Litterman 隐含均衡收益 Pi。

    公式:
        Pi = delta * Sigma * w_market
    """
    pi = delta * cov.values @ MARKET_WEIGHTS.reindex(ASSET_ORDER).values
    return pd.Series(pi, index=ASSET_ORDER, name="equilibrium_return")


def omega_from_confidence(P: pd.DataFrame, cov: pd.DataFrame, confidences: list[float]) -> pd.DataFrame:
    """根据 view confidence 构造 Omega 矩阵。

    初学者理解:
        Omega 表示“我们对观点 Q 有多不确定”。
        confidence 越高，Omega 越小，观点对后验收益影响越大。
    """
    view_var = np.diag(P.values @ (TAU * cov.values) @ P.values.T)
    omega_diag = []
    for var, conf in zip(view_var, confidences):
        conf = min(max(conf, 0.01), 0.99)
        omega_diag.append(var * (1.0 - conf) / conf)
    return pd.DataFrame(np.diag(omega_diag), index=P.index, columns=P.index)


def black_litterman_posterior(
    pi: pd.Series,
    cov: pd.DataFrame,
    P: pd.DataFrame | None = None,
    Q: pd.Series | None = None,
    confidences: list[float] | None = None,
) -> pd.Series:
    """计算 Black-Litterman 后验预期收益。

    输入:
        pi: 市场均衡收益，即没有观点时的先验。
        P: 观点矩阵，表示观点影响哪些资产。
        Q: 每条观点对应的预期收益影响。
        confidences: 每条观点的置信度。

    返回:
        融合先验和观点后的后验预期收益。
    """
    if P is None or Q is None or len(P) == 0:
        return pi.copy()

    tau_sigma = TAU * cov.values
    omega = omega_from_confidence(P, cov, confidences or [0.2] * len(P))
    inv_tau_sigma = np.linalg.pinv(tau_sigma)
    inv_omega = np.linalg.pinv(omega.values)
    middle = np.linalg.pinv(inv_tau_sigma + P.values.T @ inv_omega @ P.values)
    posterior = middle @ (inv_tau_sigma @ pi.values + P.values.T @ inv_omega @ Q.values)
    return pd.Series(posterior, index=ASSET_ORDER, name="posterior_return")


def conservative_views() -> tuple[pd.DataFrame, pd.Series, list[float]]:
    """构造固定的低置信度宏观观点。

    P 矩阵例子:
        Gold_outperforms_Nasdaq = [1, 0, 0, 0, -1]
        表示看多 Gold 相对于 Nasdaq。
    """
    rows = {
        "Gold_outperforms_Nasdaq": [1, 0, 0, 0, -1],
        "Treasury_outperforms_Bitcoin": [0, 1, -1, 0, 0],
        "Oil_outperforms_Treasury": [0, -1, 0, 1, 0],
    }
    P = pd.DataFrame(rows, index=ASSET_ORDER).T
    Q = pd.Series(
        {
            "Gold_outperforms_Nasdaq": 0.015,
            "Treasury_outperforms_Bitcoin": 0.020,
            "Oil_outperforms_Treasury": 0.015,
        }
    )
    return P, Q, [0.20, 0.20, 0.20]


def zscore_latest(panel: pd.DataFrame, col: str) -> float:
    """计算某个因子最新值相对历史均值的 z-score。

    z-score = (最新值 - 历史均值) / 历史标准差。
    """
    if col not in panel.columns:
        return np.nan
    series = panel[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 50 or series.std() == 0:
        return np.nan
    return float((series.iloc[-1] - series.mean()) / series.std())


def build_exploratory_views() -> tuple[pd.DataFrame, pd.Series, list[float], pd.DataFrame, pd.DataFrame]:
    """根据 research-only 因子组构造探索性 BL views。

    作用:
        这些因子没有通过严格审查，所以不直接交易。
        这里只让它们低置信度地影响 BL 观点方向和置信度。
    """
    panel = pd.read_csv(PROCESSED_DATA_DIR / "clean_factor_panel.csv", parse_dates=["date"], index_col="date")
    strict = pd.read_csv(PROCESSED_DATA_DIR / "research_only_factors.csv")

    groups = {
        "risk_off": ["vix_change", "btc_nasdaq_corr_60d", "btc_gold_corr_60d"],
        "inflation_commodity": [
            "oil_mom_5d",
            "oil_mom_20d",
            "oil_winsorized_return",
            "oil_dollar_change_return",
            "oil_vix_factor",
        ],
        "liquidity_risk_on": ["btc_mom_5d", "btc_mom_20d", "vix_change", "nasdaq", "treasury"],
    }

    mapping_rows = []
    scores = {}
    for group, factors in groups.items():
        zscores = []
        for factor in factors:
            z = zscore_latest(panel, factor)
            if group == "liquidity_risk_on" and factor == "vix_change":
                z = -z if pd.notna(z) else z
            zscores.append(z)
            mapping_rows.append(
                {
                    "scenario_group": group,
                    "factor_name": factor,
                    "latest_zscore": z,
                    "used_as": "research_only_direction_magnitude_confidence_helper",
                }
            )
        scores[group] = float(np.nanmean(zscores)) if np.isfinite(np.nanmean(zscores)) else np.nan

    views = []
    if scores["risk_off"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["risk_off"]))
        views.extend(
            [
                ("Treasury_outperforms_Bitcoin", [0, 1, -1, 0, 0], 0.020, conf, "risk_off"),
                ("Gold_outperforms_Nasdaq", [1, 0, 0, 0, -1], 0.015, conf, "risk_off"),
            ]
        )
    if scores["inflation_commodity"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["inflation_commodity"]))
        views.extend(
            [
                ("Oil_outperforms_Treasury", [0, -1, 0, 1, 0], 0.015, conf, "inflation_commodity"),
                ("Gold_outperforms_Treasury", [1, -1, 0, 0, 0], 0.010, conf, "inflation_commodity"),
            ]
        )
    if scores["liquidity_risk_on"] > 0:
        conf = min(0.30, 0.10 + 0.05 * abs(scores["liquidity_risk_on"]))
        views.extend(
            [
                ("Nasdaq_outperforms_Treasury", [0, -1, 0, 0, 1], 0.025, conf, "liquidity_risk_on"),
                ("Bitcoin_outperforms_Treasury", [0, -1, 1, 0, 0], 0.030, conf, "liquidity_risk_on"),
            ]
        )

    if not views:
        views.append(("No_active_factor_view", [0, 0, 0, 0, 0], 0.0, 0.01, "no_score_positive"))

    P = pd.DataFrame({v[0]: v[1] for v in views}, index=ASSET_ORDER).T
    Q = pd.Series({v[0]: v[2] for v in views})
    confidences = [v[3] for v in views]
    factor_views = pd.DataFrame(
        [
            {
                "view_name": name,
                "scenario_group": group,
                "view_return_annualized": q,
                "confidence": conf,
                "exploratory": True,
                "note": "research-only factors inform direction/magnitude/confidence; they do not directly set weights",
            }
            for name, _, q, conf, group in views
        ]
    )
    mapping = pd.DataFrame(mapping_rows)

    appendix = strict.copy()
    appendix["appendix_note"] = np.where(
        appendix["factor_name"].eq("gold_china_to_us_factor"),
        "Gold factor kept as research-only daily proxy evidence because raw data has no hour-level US open/reopen prices.",
        "Research-only factor; allowed only as low-confidence scenario context.",
    )
    appendix.to_csv(PROCESSED_DATA_DIR / "research_only_factor_appendix.csv", index=False)
    return P, Q, confidences, mapping, factor_views


def portfolio_performance_rows(portfolios: list[PortfolioResult], cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为所有组合生成静态绩效表和风险贡献表。

    ex-ante 表示“事前估计”，这里用样本均值和协方差估计未来表现，
    不是实际回测收益。
    """
    perf_rows = []
    risk_rows = []
    for pf in portfolios:
        w = pf.weights.reindex(ASSET_ORDER)
        variance = portfolio_variance(w.values, cov)
        vol = math.sqrt(max(variance, 0.0))
        expected_return = portfolio_return(w.values, pf.expected_returns)
        sharpe = expected_return / vol if vol > 0 else np.nan
        perf_rows.append(
            {
                "portfolio_name": pf.name,
                "expected_annual_return": expected_return,
                "expected_annual_volatility": vol,
                "expected_sharpe_ratio": sharpe,
                "max_asset_weight": w.max(),
                "herfindahl_concentration_index": float(np.sum(w.values**2)),
                "portfolio_variance": variance,
            }
        )
        rc = risk_contribution(w, cov)
        rc.insert(0, "portfolio_name", pf.name)
        risk_rows.append(rc)
    return pd.DataFrame(perf_rows), pd.concat(risk_rows, ignore_index=True)


def save_views(P: pd.DataFrame, Q: pd.Series, confidences: list[float], path) -> None:
    """保存 BL view 矩阵。

    输出文件会包含:
        - P 矩阵每个资产的暴露。
        - Q_annualized 年化观点收益。
        - confidence 观点置信度。
    """
    out = P.copy()
    out["Q_annualized"] = Q.reindex(out.index)
    out["confidence"] = confidences
    out.to_csv(path)


def resolve_project_path(path_value: str):
    """把 YAML 里的路径解析成项目内路径。

    例如:
        data/processed/clean_factor_panel.csv
    会被解析到项目根目录下的真实文件。
    """
    path = BASE_DIR / path_value
    if path.exists():
        return path
    return Path(path_value) if Path(path_value).is_absolute() else path


def load_custom_factor_views(config_path=CUSTOM_FACTOR_CONFIG_PATH) -> dict:
    """读取 custom_factor_views.yaml 配置。

    Missing YAML defaults to disabled custom views. Invalid entries are handled
    downstream as warnings in the view log instead of stopping the BL workflow.

    初学者理解:
        YAML 里定义的是“哪些因子要转成 Black-Litterman 观点”，
        不是直接定义权重。
    """
    if not config_path.exists():
        return {"use_custom_factor_views": False, "custom_views": []}
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config.setdefault("use_custom_factor_views", False)
    config.setdefault("custom_views", [])
    return config


def latest_lagged_factor_value(factor_df: pd.DataFrame, factor_col: str, lag_days: int) -> tuple[pd.Timestamp | None, float | None]:
    """取得最新可用因子值，并按 lag_days 控制可用日期。

    静态版本里 decision_date 使用因子序列最后一天。
    如果 lag_days=1，就使用最后一天之前至少 1 天的最近值。
    中国黄金 custom signal 当前配置为 lag_days=0，因为假设是同日 proxy。
    """
    if "date" in factor_df.columns:
        factor_df = factor_df.copy()
        factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
        factor_df = factor_df.dropna(subset=["date"]).set_index("date")
    elif not isinstance(factor_df.index, pd.DatetimeIndex):
        return None, None

    series = factor_df[factor_col].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if series.empty:
        return None, None
    decision_date = series.index.max()
    cutoff = decision_date - pd.Timedelta(days=max(int(lag_days), 0))
    available = series.loc[series.index <= cutoff]
    if available.empty:
        return None, None
    return available.index[-1], float(available.iloc[-1])


def custom_view_mapping_params(item: dict | None = None) -> dict:
    """Read custom BL view mapping parameters from one YAML item."""
    item = item or {}
    params = DEFAULT_CUSTOM_VIEW_PARAMS.copy()
    for key, default in params.items():
        try:
            params[key] = float(item.get(key, default))
        except (TypeError, ValueError):
            params[key] = default

    params["trigger_threshold"] = max(params["trigger_threshold"], 0.0)
    params["inactive_confidence"] = float(np.clip(params["inactive_confidence"], 0.0, 1.0))
    params["q_cap"] = abs(params["q_cap"])
    params["max_confidence"] = float(np.clip(params["max_confidence"], 0.0, 1.0))
    params["base_confidence"] = float(np.clip(params["base_confidence"], 0.0, params["max_confidence"]))
    params["confidence_slope"] = max(params["confidence_slope"], 0.0)
    return params


def dynamic_zscore_custom_view(z: float | None, params: dict | None = None) -> tuple[float, float, str]:
    """把 z-score 因子动态映射成 BL 的 Q 和 confidence。

    规则:
        - z 缺失: 跳过。
        - |z| < 1: 不触发强观点，Q=0，confidence=0.01。
        - |z| >= 1: Q 随 z 线性变化，并限制在 +/-3% 年化以内。
        - |z| 越大，confidence 越高，上限由 YAML 的 max_confidence 控制。
    """
    params = custom_view_mapping_params(params)
    if z is None or pd.isna(z) or not np.isfinite(z):
        return np.nan, np.nan, "skipped_nan_z"
    if abs(z) < params["trigger_threshold"]:
        return 0.0, params["inactive_confidence"], "not_triggered_near_zero"
    q_annualized = float(np.clip(params["q_scale"] * z, -params["q_cap"], params["q_cap"]))
    confidence = float(
        min(
            params["base_confidence"]
            + params["confidence_slope"] * max(abs(z) - params["trigger_threshold"], 0.0),
            params["max_confidence"],
        )
    )
    return q_annualized, confidence, "active_dynamic_zscore"


def build_custom_factor_views(config_path=CUSTOM_FACTOR_CONFIG_PATH) -> tuple[pd.DataFrame, pd.Series, list[float], pd.DataFrame]:
    """把启用的 custom factors 转换成低置信度 BL views。

    重要:
        因子不会直接决定资产权重。
        它先被映射成 P/Q/confidence，然后进入 Black-Litterman posterior，
        最后再由优化器决定权重。
    """
    config = load_custom_factor_views(config_path)
    logs = []
    rows = []
    q_values = {}
    confidences = []

    if not config.get("use_custom_factor_views", False):
        # 如果总开关关闭，返回空 P/Q，并在日志中说明。
        log = pd.DataFrame(
            [
                {
                    "view_name": "",
                    "status": "disabled",
                    "message": "use_custom_factor_views is false or config file is missing",
                }
            ]
        )
        return pd.DataFrame(columns=ASSET_ORDER), pd.Series(dtype="float64"), [], log

    for item in config.get("custom_views", []):
        # YAML 中每个 item 对应一个候选 custom view。
        name = item.get("name", "unnamed_custom_view")
        if not item.get("enabled", False):
            logs.append({"view_name": name, "status": "skipped", "message": "view disabled"})
            continue

        factor_file = resolve_project_path(str(item.get("factor_file", "")))
        factor_col = item.get("factor_column")
        affected_asset = item.get("affected_asset")
        direction = str(item.get("direction", "positive")).lower()
        lag_days = int(item.get("lag_days", 0) or 0)

        if not factor_file.exists():
            logs.append({"view_name": name, "status": "skipped", "message": f"missing factor_file: {factor_file}"})
            continue
        if affected_asset not in ASSET_ORDER:
            logs.append({"view_name": name, "status": "skipped", "message": f"affected_asset not in asset pool: {affected_asset}"})
            continue

        factor_df = pd.read_csv(factor_file)
        if factor_col not in factor_df.columns:
            logs.append({"view_name": name, "status": "skipped", "message": f"missing factor_column: {factor_col}"})
            continue

        # 找到当前静态样本中最新可用的因子值。
        factor_time, factor_value = latest_lagged_factor_value(factor_df, factor_col, lag_days)
        if factor_value is None:
            logs.append({"view_name": name, "status": "skipped", "message": "no lagged factor value available"})
            continue

        # direction=negative 时，先把 z-score 方向反过来，统一成“z 越高越看多 affected_asset”。
        sign_multiplier = 1.0 if direction == "positive" else -1.0
        z_score_used = sign_multiplier * factor_value
        # 使用统一规则，把 z-score 映射成 Q、confidence 和触发状态。
        mapping_params = custom_view_mapping_params(item)
        view_return, confidence, trigger_status = dynamic_zscore_custom_view(z_score_used, mapping_params)
        view_name = f"custom_{name}"
        if trigger_status == "active_dynamic_zscore":
            # P 行是一个资产暴露向量；absolute view 只影响 affected_asset。
            p_row = pd.Series(0.0, index=ASSET_ORDER)
            p_row[affected_asset] = 1.0
            rows.append((view_name, p_row))
            q_values[view_name] = view_return
            confidences.append(confidence)

        logs.append(
            {
                "view_name": view_name,
                "status": "active" if trigger_status == "active_dynamic_zscore" else "skipped",
                "factor_file": str(factor_file),
                "factor_column": factor_col,
                "raw_factor_value": factor_value,
                "z_score_used": z_score_used,
                "affected_asset": affected_asset,
                "direction": direction,
                "lag_days": lag_days,
                "factor_time_used": factor_time,
                "factor_value_used": factor_value,
                "Q_annualized": view_return,
                "confidence": confidence,
                "trigger_threshold": mapping_params["trigger_threshold"],
                "q_scale": mapping_params["q_scale"],
                "q_cap": mapping_params["q_cap"],
                "base_confidence": mapping_params["base_confidence"],
                "confidence_slope": mapping_params["confidence_slope"],
                "max_confidence": mapping_params["max_confidence"],
                "trigger_status": trigger_status,
                "trigger_rule": CUSTOM_TRIGGER_RULE,
                "alignment_assumption": CUSTOM_ALIGNMENT_ASSUMPTION,
                "proxy_warning": CUSTOM_PROXY_WARNING,
                "research_only": True,
                "lookahead_warning": CUSTOM_ALIGNMENT_ASSUMPTION,
                "message": "dynamic z-score custom BL view" if trigger_status == "active_dynamic_zscore" else trigger_status,
                "description": item.get("description", ""),
            }
        )

    if rows:
        P = pd.DataFrame({name: row for name, row in rows}).T
        Q = pd.Series(q_values)
    else:
        P = pd.DataFrame(columns=ASSET_ORDER)
        Q = pd.Series(dtype="float64")
    return P, Q, confidences, pd.DataFrame(logs)


def save_charts(
    corr: pd.DataFrame,
    weights: pd.DataFrame,
    performance: pd.DataFrame,
    risk_contrib: pd.DataFrame,
    prior: pd.Series,
    post_cons: pd.Series,
    post_expl: pd.Series,
    post_custom: pd.Series,
    custom_view_log: pd.DataFrame,
) -> None:
    """生成静态阶段图表。

    包括:
        - 资产相关性热力图。
        - 各组合权重柱状图。
        - 预期收益/风险散点图。
        - BL 先验和后验收益对比。
        - custom factor 置信度和影响图。
    """
    charts_dir = PROCESSED_DATA_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Asset Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(charts_dir / "asset_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    weights.T.plot(kind="bar", figsize=(10, 6))
    plt.title("Portfolio Weights")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(charts_dir / "portfolio_weights_bar_chart.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(performance["expected_annual_volatility"], performance["expected_annual_return"])
    for _, row in performance.iterrows():
        ax.annotate(row["portfolio_name"], (row["expected_annual_volatility"], row["expected_annual_return"]), fontsize=8)
    ax.set_xlabel("Expected Annual Volatility")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Expected Return vs Volatility")
    fig.tight_layout()
    fig.savefig(charts_dir / "expected_return_vs_volatility_scatter.png", dpi=150)
    plt.close(fig)

    risk_pivot = risk_contrib.pivot(index="portfolio_name", columns="asset", values="percentage_risk_contribution")
    risk_pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Percentage Risk Contribution")
    plt.ylabel("Risk Contribution")
    plt.tight_layout()
    plt.savefig(charts_dir / "risk_contribution_chart.png", dpi=150)
    plt.close()

    bl_ret = pd.DataFrame(
        {
            "Equilibrium": prior,
            "Conservative posterior": post_cons,
            "Exploratory posterior": post_expl,
            "Custom posterior": post_custom,
        }
    )
    bl_ret.plot(kind="bar", figsize=(10, 6))
    plt.title("Black-Litterman Prior vs Posterior Expected Returns")
    plt.ylabel("Annualized Expected Return")
    plt.tight_layout()
    plt.savefig(charts_dir / "bl_prior_vs_posterior_returns.png", dpi=150)
    plt.close()

    weights[["BL Conservative", "BL Exploratory"]].plot(kind="bar", figsize=(9, 6))
    plt.title("Conservative vs Exploratory BL Weights")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(charts_dir / "bl_conservative_vs_exploratory_weights.png", dpi=150)
    plt.close()

    weights[["BL No View", "BL Conservative", "BL Exploratory", "BL Custom Factor Views"]].plot(
        kind="bar",
        figsize=(10, 6),
    )
    plt.title("BL No View vs Conservative vs Exploratory vs Custom Weights")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(charts_dir / "bl_no_view_conservative_exploratory_custom_weights.png", dpi=150)
    plt.close()

    if "status" in custom_view_log.columns:
        active = custom_view_log[custom_view_log["status"].eq("active")].copy()
        impact_source = custom_view_log[
            custom_view_log.get("affected_asset", pd.Series(index=custom_view_log.index, dtype=object)).notna()
            & custom_view_log.get("Q_annualized", pd.Series(index=custom_view_log.index, dtype=float)).notna()
        ].copy()
    else:
        active = pd.DataFrame()
        impact_source = pd.DataFrame()
    if not active.empty:
        active.plot(x="view_name", y="confidence", kind="bar", legend=False, figsize=(8, 5))
        plt.title("Custom Factor View Confidence")
        plt.ylabel("Confidence")
        plt.tight_layout()
        plt.savefig(charts_dir / "custom_factor_view_confidence.png", dpi=150)
        plt.close()

    if not impact_source.empty:
        impact = impact_source.groupby("affected_asset")["Q_annualized"].sum()
        impact.plot(kind="bar", figsize=(8, 5))
        plt.title("Custom Factor Impact on Affected Assets")
        plt.ylabel("Annualized View Impact")
        plt.tight_layout()
        plt.savefig(charts_dir / "custom_factor_impact_on_affected_assets.png", dpi=150)
        plt.close()

    panel_path = PROCESSED_DATA_DIR / "clean_factor_panel.csv"
    if panel_path.exists():
        panel = pd.read_csv(panel_path, parse_dates=["date"], index_col="date")
        signal_cols = ["china_gold_morning_to_evening_ret", "china_gold_morning_to_evening_z"]
        if all(col in panel.columns for col in signal_cols):
            signal = panel[signal_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
            if not signal.empty:
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(signal.index, signal["china_gold_morning_to_evening_ret"], label="morning-to-evening return")
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax.set_ylabel("Return")
                ax2 = ax.twinx()
                ax2.plot(signal.index, signal["china_gold_morning_to_evening_z"], color="tab:orange", label="z-score")
                ax2.axhline(1, color="tab:orange", linewidth=0.7, linestyle=":")
                ax2.axhline(-1, color="tab:orange", linewidth=0.7, linestyle=":")
                ax2.set_ylabel("Z-score")
                ax.set_title("China Gold Morning-to-Evening Signal")
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc="best")
                fig.tight_layout()
                fig.savefig(charts_dir / "china_gold_morning_to_evening_signal.png", dpi=150)
                plt.close(fig)


def write_stage2_report(
    performance: pd.DataFrame,
    custom_view_log: pd.DataFrame,
    target_note: str,
    delta_note: str,
) -> None:
    """写出 Stage 2 静态 Black-Litterman 配置报告。

    报告用于解释:
        - 为什么这些组合仍然是 BL。
        - custom factor 如何进入 P/Q views。
        - 中国黄金信号为什么仍是 research-only proxy。
    """
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    active = (
        custom_view_log[custom_view_log["status"].eq("active")]
        if "status" in custom_view_log.columns
        else pd.DataFrame()
    )
    if active.empty:
        custom_lines = ["- No active custom factor views were applied."]
    else:
        custom_lines = [
            (
                f"- `{row['view_name']}`: {row['affected_asset']} Q={row['Q_annualized']:.2%}, "
                f"confidence={row['confidence']:.1%}, factor value={row['factor_value_used']:.6f}."
            )
            for _, row in active.iterrows()
        ]

    report = f"""# Stage 2 Black-Litterman Static Summary

## Scope

This stage generates static portfolio weights and ex-ante diagnostics. It does not run a rolling backtest, cumulative strategy simulation, transaction cost analysis, or turnover analysis.

## Why This Is Still Black-Litterman

Weights are optimized from equilibrium implied returns and Black-Litterman posterior expected returns. Custom factors do not directly determine weights. They only enter through low-confidence views that affect expected returns before optimization.

## Custom Factor Views

Custom factor views are user-selected research views. They are not approved high-confidence signals.

- Factors do not directly decide weights.
- Factors only influence Black-Litterman P/Q views.
- Confidence and Q strength are controlled by each custom view's YAML mapping parameters.
- Missing or invalid custom factors are skipped with warnings.
- Static BL uses the latest available custom factor value; the China gold custom view is intentionally configured with no mechanical shift(1).
- Rolling BL, when implemented, must use only factor values available before each rebalance date.

Active custom views:

{chr(10).join(custom_lines)}

## China Gold Caveat

The China gold custom factor now uses `china_gold_morning_to_evening_z`, based on China morning-to-evening return. It is research-only. The intended hypothesis is same-day China trading-session information shock to US gold reaction a few hours later. US gold open/reopen intraday prices are unavailable, so validation uses daily close-to-close proxy returns and cannot strictly prove US high-open behavior.

## Model Hierarchy

Conservative BL remains the main model. Custom BL is user-defined scenario analysis, not a direct trading strategy.

## Notes

- {target_note}
- {delta_note}

## Ex-Ante Performance

{performance.to_markdown(index=False)}
"""
    (reports_dir / "stage2_black_litterman_static_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    """运行 Stage 2 静态组合优化和 Black-Litterman 全流程。"""
    # 1. 读取资产收益，并计算样本均值、波动率、协方差和相关矩阵。
    returns = read_asset_returns()
    sample_mu, ann_vol, cov, corr = annualized_stats(returns)

    # 2. 构造几个传统基准组合。
    equal_weight = pd.Series(1.0 / len(ASSET_ORDER), index=ASSET_ORDER)
    equal_weight["Bitcoin"] = min(equal_weight["Bitcoin"], MAX_WEIGHTS["Bitcoin"])
    excess = (1.0 - equal_weight.sum()) / (len(ASSET_ORDER) - 1)
    for asset in ASSET_ORDER:
        if asset != "Bitcoin":
            equal_weight[asset] += excess

    min_var = optimize_min_variance(cov)
    max_sharpe = optimize_max_sharpe(sample_mu, cov)
    risk_parity = optimize_risk_parity(cov)
    target_vol, target_note = optimize_target_vol(sample_mu, cov)

    # 3. 反推 Black-Litterman 均衡收益 Pi。
    delta, delta_note = estimate_delta(sample_mu, cov)
    pi = implied_equilibrium_returns(cov, delta)
    pi.to_frame("equilibrium_return").to_csv(PROCESSED_DATA_DIR / "bl_equilibrium_returns.csv")

    # 4. BL No View：不加入主观观点，只使用均衡收益。
    no_view_mu = black_litterman_posterior(pi, cov)
    no_view_w = optimize_max_sharpe(no_view_mu, cov)

    # 5. BL Conservative：加入固定保守宏观 views。
    P_cons, Q_cons, conf_cons = conservative_views()
    post_cons = black_litterman_posterior(pi, cov, P_cons, Q_cons, conf_cons)
    cons_w = optimize_max_sharpe(post_cons, cov)

    # 6. BL Exploratory：由 research-only 因子组辅助生成低置信度 views。
    P_expl, Q_expl, conf_expl, scenario_mapping, factor_views = build_exploratory_views()
    post_expl = black_litterman_posterior(pi, cov, P_expl, Q_expl, conf_expl)
    expl_w = optimize_max_sharpe(post_expl, cov)

    # 7. BL Custom Factor Views：读取 YAML，把用户指定因子转换成动态 views。
    P_custom_only, Q_custom_only, conf_custom_only, custom_view_log = build_custom_factor_views()
    if len(P_custom_only) > 0:
        P_custom = pd.concat([P_cons, P_custom_only], axis=0)
        Q_custom = pd.concat([Q_cons, Q_custom_only], axis=0)
        conf_custom = conf_cons + conf_custom_only
    else:
        P_custom = P_cons.copy()
        Q_custom = Q_cons.copy()
        conf_custom = conf_cons.copy()
    post_custom = black_litterman_posterior(pi, cov, P_custom, Q_custom, conf_custom)
    custom_w = optimize_max_sharpe(post_custom, cov)

    # 8. 保存 BL views、posterior returns、weights 和日志。
    save_views(P_cons, Q_cons, conf_cons, PROCESSED_DATA_DIR / "bl_views_matrix_conservative.csv")
    save_views(P_expl, Q_expl, conf_expl, PROCESSED_DATA_DIR / "bl_views_matrix_exploratory.csv")
    save_views(P_custom_only, Q_custom_only, conf_custom_only, PROCESSED_DATA_DIR / "custom_views_matrix.csv")
    post_cons.to_frame("posterior_return").to_csv(PROCESSED_DATA_DIR / "bl_posterior_returns_conservative.csv")
    post_expl.to_frame("posterior_return").to_csv(PROCESSED_DATA_DIR / "bl_posterior_returns_exploratory.csv")
    post_custom.to_frame("posterior_return").to_csv(PROCESSED_DATA_DIR / "bl_posterior_returns_custom.csv")
    no_view_w.to_frame("weight").to_csv(PROCESSED_DATA_DIR / "bl_weights_no_view.csv")
    cons_w.to_frame("weight").to_csv(PROCESSED_DATA_DIR / "bl_weights_conservative.csv")
    expl_w.to_frame("weight").to_csv(PROCESSED_DATA_DIR / "bl_weights_exploratory.csv")
    custom_w.to_frame("weight").to_csv(PROCESSED_DATA_DIR / "bl_weights_custom.csv")
    scenario_mapping.to_csv(PROCESSED_DATA_DIR / "scenario_factor_mapping.csv", index=False)
    factor_views.to_csv(PROCESSED_DATA_DIR / "factor_informed_views.csv", index=False)
    custom_view_log.to_csv(PROCESSED_DATA_DIR / "custom_factor_view_log.csv", index=False)

    # 9. 汇总所有组合，生成权重表、绩效表和风险贡献表。
    portfolios = [
        PortfolioResult("Equal Weight", equal_weight, sample_mu),
        PortfolioResult("Minimum Variance", min_var, sample_mu),
        PortfolioResult("Maximum Sharpe", max_sharpe, sample_mu),
        PortfolioResult("Risk Parity", risk_parity, sample_mu),
        PortfolioResult("Target Volatility", target_vol, sample_mu),
        PortfolioResult("BL No View", no_view_w, no_view_mu),
        PortfolioResult("BL Conservative", cons_w, post_cons),
        PortfolioResult("BL Exploratory", expl_w, post_expl),
        PortfolioResult("BL Custom Factor Views", custom_w, post_custom),
    ]
    weights = pd.DataFrame({pf.name: pf.weights.reindex(ASSET_ORDER) for pf in portfolios})
    weights.to_csv(PROCESSED_DATA_DIR / "portfolio_weights_static.csv")

    performance, risk = portfolio_performance_rows(portfolios, cov)
    performance.to_csv(PROCESSED_DATA_DIR / "portfolio_ex_ante_performance.csv", index=False)
    risk.to_csv(PROCESSED_DATA_DIR / "portfolio_risk_contribution.csv", index=False)

    # 10. 保存约束说明，方便审查优化器到底受哪些规则限制。
    constraints_summary = pd.DataFrame(
        [
            {"constraint": "sum_weights", "value": "sum(w)=1"},
            {"constraint": "long_only", "value": "0 <= w_i"},
            {"constraint": "max_single_asset_weight", "value": "0.50"},
            {"constraint": "btc_weight_cap", "value": "0.15"},
            {"constraint": "oil_weight_cap", "value": "0.25"},
            {"constraint": "target_volatility", "value": "12% annualized"},
            {"constraint": "target_volatility_status", "value": target_note},
            {"constraint": "market_weights", "value": MARKET_WEIGHTS.to_dict()},
            {"constraint": "delta", "value": delta},
            {"constraint": "delta_note", "value": delta_note},
            {"constraint": "rolling_backtest", "value": "not performed in Stage 2"},
            {"constraint": "custom_factor_view_mapping", "value": "controlled by custom_factor_views.yaml"},
            {
                "constraint": "custom_factor_view_rule",
                "value": "custom factors only modify BL views and never directly determine weights",
            },
            {
                "constraint": "custom_factor_lookahead_guard",
                "value": "static run uses configured lag_days; rolling run must use data before rebalance date",
            },
        ]
    )
    constraints_summary.to_csv(PROCESSED_DATA_DIR / "optimization_constraints_summary.csv", index=False)

    # 11. 生成图表和 Markdown 报告。
    save_charts(corr, weights, performance, risk, pi, post_cons, post_expl, post_custom, custom_view_log)
    write_stage2_report(performance, custom_view_log, target_note, delta_note)

    best_stability = performance.sort_values(["expected_annual_volatility", "expected_sharpe_ratio"], ascending=[True, False]).iloc[0]
    risk_balance = (
        risk.groupby("portfolio_name")["percentage_risk_contribution"]
        .std()
        .sort_values()
        .index[0]
    )

    print(f"Saved Stage 2 outputs to {PROCESSED_DATA_DIR}")
    print("\nStage 2 conclusions")
    print("1. This is still Black-Litterman because weights are optimized from equilibrium implied returns and BL posterior returns, not directly from factors.")
    print("2. Research-only factors only shape custom BL views; Q/confidence mapping is controlled by YAML.")
    print("3. Factors cannot directly decide weights because strict review found no approved high-confidence factors and several look-ahead/stability risks.")
    print("4. Conservative BL uses fixed low-confidence macro views; Exploratory BL lets research-only factor groups tilt view direction and confidence.")
    print(
        "5. More stable ex-ante risk/return by low volatility screen: "
        f"{best_stability['portfolio_name']} with vol {best_stability['expected_annual_volatility']:.2%}."
    )
    print(f"6. Most balanced risk contribution: {risk_balance}.")
    active_custom_count = (
        int(custom_view_log["status"].eq("active").sum()) if "status" in custom_view_log.columns else 0
    )
    print(f"7. Custom BL active custom factor views: {active_custom_count}; factors do not directly set weights.")
    print("8. Stage 3 should rolling-test these static methods with re-estimated covariance/BL inputs, lagged factor availability, transaction costs, turnover, and out-of-sample performance.")


if __name__ == "__main__":
    main()
