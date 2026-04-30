# 项目代码导读

当前项目只保留一条因子主线：

`get_data.py` -> `build_clean_factor_panel.py` -> `strict_factor_review.py` -> `build_static_portfolios_bl.py` -> `run_rolling_backtest_bl.py`

## `build_clean_factor_panel.py`

构建核心日频因子面板 `data/processed/clean_factor_panel.csv`。

- `build_returns_from_prices`: 从原始价格直接计算收益率。
- `add_china_gold_returns`: 只基于中国黄金 morning/evening 日频字段计算代理信号。
- `build_clean_factor_panel`: 合并价格、FRED、中国黄金数据，并调用 `src/factors/*.py` 中的因子公式。
- `build_quality_report`: 生成因子质量报告。
- `build_china_gold_signal_alignment_report`: 只做日频 same-day / next-observed proxy 检查，不做小时级开盘/reopen 检查。

## `src/factors/*.py`

这些文件保存具体因子公式：

- `gold_factors.py`: 黄金和中国黄金代理因子。
- `bond_factors.py`: 债券/利率因子。
- `vix_factors.py`: VIX 因子。
- `btc_factors.py`: BTC 相关性和动量因子。
- `oil_factors.py`: 原油因子。
- `equity_factors.py`: 权益/风险偏好因子。
- `common.py`: 滚动相关、滚动求和、列检查等通用工具。

## `strict_factor_review.py`

基于 `clean_factor_panel.csv` 和日频代理检查做严格因子审查，输出：

- `strict_factor_quality_report.csv`
- `strict_factor_decision_summary.csv`
- `approved_candidate_factors.csv`
- `research_only_factors.csv`
- `rejected_factors.csv`
- `gold_daily_proxy_diagnostic_summary.csv`

## `build_static_portfolios_bl.py`

构建静态组合和 Black-Litterman 组合。它读取 `clean_factor_panel.csv` 和严格审查结果。

## `run_rolling_backtest_bl.py`

滚动窗口回测 Black-Litterman 策略，使用滞后后的可用因子值。

## 已删除的旧流程

以下旧脚本已删除，避免和 clean 主线重复：

- `build_returns.py`
- `build_factor_panel.py`
- `validate_factors.py`

小时级黄金检查也已删除，因为当前 raw 数据没有美国黄金小时级 open/reopen 价格。
