# 多资产因子工程、Black-Litterman 组合优化与滚动回测项目
(仍在更新改善，新添加美中指数关系因子，可与data/market analysis查看）
## 一、项目简介

本项目围绕五类核心资产展开：黄金、美国国债、比特币、原油和纳斯达克风险资产。项目从数据获取、收益率构建、因子工程、因子质量审查，到 Black-Litterman 静态组合构建，再到 rolling out-of-sample 回测。。

这个项目的核心研究逻辑是：

1. 多资产组合比单一资产更适合表达宏观环境变化；
2. 不同资产对利率、通胀、风险偏好、流动性和地缘政治风险的反应不同；
3. 单纯使用历史收益均值做均值-方差优化非常不稳定；
4. Black-Litterman 可以把市场均衡收益和研究观点结合起来；
5. 因子可以帮助解释观点，但不能未经验证就当作高置信度 alpha；
6. rolling out-of-sample backtest 用于检查组合方法是否在样本外仍然稳健。


本项目不适合直接作为实盘交易系统。原因是数据源、交易成本、滑点、期货连续合约、保证金、税费、盘中价格和执行细节都仍有简化。

## 二、资产池说明

| 资产 | 项目代理 | 经济含义 | 主要风险 |
|---|---|---|---|
| 黄金 | `GC=F` 或 Gold proxy | 避险资产、通胀预期、美元实际利率敏感资产 | 利率上行、美元走强、期货连续合约误差 |
| 美国国债 | `TLT` 或 Treasury proxy | 防御资产、利率周期、风险偏好下降时的避险配置 | 利率上行、久期风险 |
| 比特币 | `BTC-USD` | 高波动另类资产、流动性和风险偏好代理 | 极高波动、监管、交易时间不同 |
| 原油 | `CL=F` 或 Oil proxy | 通胀、供需、商品周期、地缘政治 | 期货换月、极端价格、2020 负油价事件 |
| 纳斯达克 | `QQQ` | 成长股、科技股、风险资产、利率敏感资产 | 高估值、利率冲击、风险偏好反转 |

这些资产之间可能存在动态相关性。例如，风险偏好上升时纳斯达克和比特币可能同涨；风险偏好下降时黄金和国债可能提供防御；通胀或供给冲击下原油可能独立走强。多资产组合的价值就在于利用这些不完全同步的风险来源实现分散化。

## 三、项目整体流程

### Stage 1：数据处理与因子构建

Stage 1 负责把原始数据变成可用于研究的因子面板。主要步骤包括：

- 下载或读取 `yfinance` 价格数据；
- 下载 FRED 利率数据；
- 下载中国黄金数据；
- 构建资产收益率；
- 构建黄金、债券、VIX、比特币、原油、权益相关因子；
- 处理中美黄金时差信号，并生成时区对齐检查；
- 处理 WTI 负价格附近的稳健原油收益；
- 生成 `clean_factor_panel.csv`；
- 检查因子缺失率、IC、rank IC、分组收益、滚动 IC；
- 生成 `approved_candidate_factors.csv`、`research_only_factors.csv`、`rejected_factors.csv`。

第一阶段严格审查的结论是：当前没有高置信度 approved factor。部分因子只能作为 `research only`，用于低置信度情景分析。

### Stage 2：静态 Black-Litterman 组合构建

Stage 2 生成静态组合权重和 ex-ante 指标。主要步骤包括：

- 从 `asset_returns.csv` 计算年化收益、波动、协方差、相关性；
- 构建 Equal Weight、Minimum Variance、Maximum Sharpe、Risk Parity、Target Volatility；
- 使用假设市场权重计算 Black-Litterman 隐含均衡收益；
- 构建 No-view BL、Conservative BL、Exploratory BL、Custom Factor Views BL；
- 将低置信度 views 转换成 P/Q/Omega；
- 生成 posterior expected returns；
- 在约束下优化权重；
- 输出静态权重、风险贡献、预期收益、预期波动、Sharpe ratio。

Conservative BL 是主模型，因为它使用低置信度、可解释的宏观情景观点。Exploratory 和 Custom BL 用于研究场景，不是直接交易模型。

### Stage 3：Rolling Backtest

Stage 3 使用 rolling out-of-sample 方法验证不同组合方法的历史表现。主要设置：

- 初始资金：1,000,000 美元；
- rolling window：252 日、504 日；
- 调仓频率：monthly、quarterly；
- 每个调仓日只使用该日之前的数据；
- 权重生成后，收益从调仓日之后的 holding period 开始计算；
- 计入 10 bps 单边交易成本；
- 输出 gross return 和 net return；
- 记录权重、持仓、交易、换手率、风险贡献、回撤和净值曲线。

回测策略包括：

- Minimum Variance
- Risk Parity
- BL No View
- BL Conservative
- BL Exploratory
- BL Custom Factor Views

### Stage 4：结果分析与策略诊断

当前项目已经生成了 Stage 3 结果和审查报告。后续可以继续扩展：

- 策略排名；
- Sharpe ratio 对比；
- 年化收益对比；
- 最大回撤对比；
- 风险贡献分析；
- 换手率和交易成本分析；
- custom factor 是否改善结果；
- 不同市场阶段表现；
- 最终推荐模型。

## 四、Black-Litterman 模型解释

传统均值-方差优化的问题是：它非常依赖预期收益输入。历史均值通常噪声很大，稍微变化就可能导致权重极端集中。

Black-Litterman 的核心思想是：

> 不直接相信历史均值，而是先从市场组合反推出一个较稳健的“均衡预期收益”，再把研究者的观点以可控置信度加入进去。

### 关键符号

| 符号 | 含义 |
|---|---|
| `Sigma` | 资产协方差矩阵 |
| `w_market` | 假设市场权重 |
| `delta` | 风险厌恶系数 |
| `Pi` | 市场隐含均衡收益 |
| `P` | view 暴露矩阵，说明观点涉及哪些资产 |
| `Q` | view 的预期收益差或绝对收益 |
| `Omega` | view 不确定性矩阵 |
| `tau` | 对均衡收益不确定性的缩放参数 |

项目中使用：

```text
Pi = delta * Sigma * w_market
```

如果有 view，例如“黄金年化跑赢纳斯达克 1.5%”，可以写成：

```text
P = [1, 0, 0, 0, -1]
Q = 0.015
```

含义是：

```text
Gold - Nasdaq = 1.5%
```

Black-Litterman 不是直接改权重，而是先生成 posterior expected returns，再用组合优化器生成权重。低置信度 view 对 posterior 的影响较小，高置信度 view 影响更大。本项目中 custom factor view 被限制在 30% 以内，因此更像情景分析，而不是高置信度交易信号。

## 五、因子系统说明

| 因子类型 | 示例 | 经济逻辑 | 目标资产 | 风险 |
|---|---|---|---|---|
| 黄金中美时差因子 | `gold_china_to_us_factor` | 中国黄金夜盘可能领先美国黄金交易时段 | 
| 黄金动量 | `gold_mom_5d`, `gold_mom_20d` | 趋势延续 | Gold | 动量可能反转 |
| 原油动量 | `oil_mom_5d`, `oil_mom_20d` | 商品趋势和供需冲击延续 | Oil | 期货换月和极端价格 |
| 原油 VIX 交互 | `oil_vix_interaction` | 风险冲击下油价行为不同 | Oil | 方向不稳定 |
| VIX 变化 | `vix_change` | 风险偏好变化 | Nasdaq, Oil | VIX 当日变化未必预测下一日 |
| BTC 相关性 | `btc_nasdaq_corr_60d`, `btc_gold_corr_60d` | 比特币风险属性变化 | Bitcoin | 相关性不是因果 |
| BTC 动量 | `btc_mom_5d`, `btc_mom_20d` | 高波动资产趋势延续 | Bitcoin | 换手和回撤风险 |
| 利率因子 | `bond_rate_factor`, `bond_curve_factor` | 利率变化影响债券 ETF | Treasury | FRED 公布频率和交易日对齐 |
| 权益风险偏好 | `risk_on_factor` | 纳斯达克相对国债强弱 | Nasdaq | 风险偏好可能反转 |

所有因子都需要考虑滞后和可观测时间。不能使用未来收益或未来因子值。

## 六、重要原则

1. 不允许未来函数；
2. 每个调仓日只能使用当时已经可获得的数据；
3. custom factor 必须 lag；
4. 因子不能直接强制改变权重；
5. 因子只能通过 Black-Litterman views 影响 posterior expected returns；
6. 回测结果必须区分 gross 和 net；
7. 交易成本必须计入；
8. 第一次建仓也应该计入 turnover；
9. 表现好的模型不一定是最可解释的模型；
10. Minimum Variance 或 Risk Parity 表现好，不代表 Black-Litterman 项目失败；
11. Custom factor 如果统计检验不足，只能作为低置信度情景分析。

## 七、主要输出文件说明

| 文件 | 作用 | 主要字段 | 如何解读 |
|---|---|---|---|
| `data/raw/us_asset_prices.csv` | 原始美国资产价格 | `date`, `gold_us`, `oil`, `btc`, `nasdaq`, `treasury`, `vix_level` | 原始价格层，用于构建收益 |
| `data/raw/fred_rates.csv` | FRED 利率数据 | `DGS10`, `DGS2`, `DFF` | 利率 level 数据 |
| `data/raw/china_gold_sge.csv` | 中国黄金代理数据 | `china_gold_morning`, `china_gold_evening` | 当前更像 SGE proxy，不应过度解释为 SHFE AU |
| `clean_price_panel.csv` | 清洗价格面板 | 当前未生成 | 可作为后续扩展目标 |
| `clean_return_panel.csv` | 清洗收益面板 | 当前未生成 | 当前对应文件是 `returns.csv` 和 `asset_returns.csv` |
| `clean_factor_panel.csv` | 核心因子面板 | 因子列、收益列、target 列 | Stage 1、Stage 2、Stage 3 都会使用 |
| `approved_candidate_factors.csv` | 严格通过因子 | 因子名称、统计指标 | 当前为空，说明没有高置信度因子 |
| `research_only_factors.csv` | 研究型因子 | 因子名称、降级原因 | 可作为低置信度 BL scenario 辅助 |
| `rejected_factors.csv` | 拒绝因子 | 因子名称、失败原因 | 默认不进入模型 |
| `custom_views_matrix.csv` | 自定义 BL views | P 矩阵、Q、confidence | Stage 2 custom view 输入 |
| `custom_factor_view_log.csv` | 自定义 view 日志 | factor value、Q、confidence | 检查 custom view 是否生效 |
| `bl_posterior_returns_custom.csv` | Custom BL 后验收益 | asset, posterior return | 用于 Custom BL 权重优化 |
| `bl_weights_custom.csv` | Custom BL 权重 | asset, weight | 静态 Custom BL 组合 |
| `portfolio_weights_static.csv` | 静态组合权重 | 各组合权重 | 比较静态组合配置 |
| `portfolio_ex_ante_performance.csv` | 静态组合预期表现 | expected return, volatility, Sharpe | Ex-ante，不是回测收益 |
| `portfolio_risk_contribution.csv` | 静态风险贡献 | marginal/component/percentage risk | 检查风险集中度 |
| `optimization_constraints_summary.csv` | 优化约束说明 | constraint, value | 查看 BTC/Oil/单资产上限 |
| `rolling_backtest_returns.csv` | 滚动回测日收益 | gross/net return, turnover, transaction cost | 可复原净值曲线 |
| `rolling_backtest_weights.csv` | 调仓权重 | rebalance date, asset, weight | 检查每次调仓配置 |
| `rolling_backtest_turnover.csv` | 换手率 | turnover, transaction cost | 检查交易成本来源 |
| `rolling_backtest_performance_summary.csv` | 回测绩效汇总 | annual return, Sharpe, drawdown | 最重要的回测结果表 |
| `rolling_backtest_drawdowns.csv` | 回撤曲线 | drawdown, peak value | 分析最大回撤路径 |
| `rolling_backtest_risk_contribution.csv` | 滚动风险贡献 | percent risk contribution | 检查风险是否集中 |
| `rolling_backtest_view_log.csv` | BL view 日志 | P, Q, confidence, status | 审查 BL views 是否正确 |
| `rolling_backtest_custom_factor_log.csv` | Custom factor 滚动日志 | factor date/value, Q, confidence | 检查 custom factor 是否 lag |
| `rolling_backtest_config_summary.csv` | 回测配置 | 参数、值、说明 | 复现实验设置 |
| `rolling_backtest_equity_curve.csv` | 净值曲线 | gross/net value, drawdown | 绘制回测表现 |
| `rolling_backtest_positions.csv` | 每日持仓 | actual weight, position value | 检查持仓漂移 |
| `rolling_backtest_trades.csv` | 交易记录 | old/new weight, trade value, cost | 检查调仓和交易成本 |
| `rolling_backtest_risk_events.csv` | 风控事件 | stop loss/take profit | 当前默认 risk overlay 关闭，因此为空 |

## 八、主要脚本说明

| 脚本 | 作用 | 输入 | 输出 | 命令 | 常见错误 |
|---|---|---|---|---|---|
| `src/get_data.py` | 下载原始数据 | 网络数据源 | `data/raw/*.csv` | `python src/get_data.py` | 网络失败、akshare 字段变化 |
| `src/build_clean_factor_panel.py` | 构建清洗后的核心因子面板和质量报告 | 原始价格/FRED/中国黄金 | `clean_factor_panel.csv`, `factor_quality_report.csv`, `china_gold_signal_alignment_report.csv` | `python src/build_clean_factor_panel.py` | 数据源字段变化 |
| `src/strict_factor_review.py` | 严格因子审查 | clean panel, daily proxy report | strict/rejected/research-only 文件 | `python src/strict_factor_review.py` | 缺少 clean 输出 |
| `src/build_static_portfolios_bl.py` | Stage 2 静态组合和 BL | `asset_returns.csv`, config yaml | 静态权重、BL views、图表 | `python src/build_static_portfolios_bl.py` | YAML 路径或字段错误 |
| `src/run_rolling_backtest_bl.py` | Stage 3 rolling 回测 | `asset_returns.csv`, clean factor panel, YAML | rolling 回测全套输出 | `python src/run_rolling_backtest_bl.py` | 未先生成 `asset_returns.csv` |
| `src/factors/*.py` | 资产因子模块 | 因子面板 | 新增因子列 | 被主脚本调用 | 缺少依赖列 |

## 九、项目局限性

- 免费数据源质量有限；
- `GC=F`、`CL=F` 等期货连续合约可能存在换月和复权问题；
- 中国黄金数据当前不能确认是 SHFE AU 期货；
- 日频数据无法完全捕捉盘中时差信号；
- 新闻和事件因子尚未完全结构化；
- 部分因子统计显著性不足；
- custom factor views 不能被解释为强 alpha；
- 回测不等于实盘；
- 交易成本、滑点、保证金、税费、杠杆被简化；
- 比特币、期货、ETF 的交易时间不完全一致；
- Black-Litterman views 依赖人为设定的置信度。

## 十、项目展示话术

### 30 秒介绍

这是一个多资产组合优化项目，覆盖黄金、国债、比特币、原油和纳斯达克。我先构建宏观和跨市场因子，再做严格因子检验，然后把通过或研究型因子以低置信度 views 的形式接入 Black-Litterman，最后用 rolling out-of-sample backtest 比较 BL、Minimum Variance、Risk Parity 等方法。

### 1 分钟介绍

这个项目不是简单预测单一资产收益，而是搭建一个完整的多资产研究框架。第一阶段处理数据、构建因子并做质量审查；第二阶段构建静态 Black-Litterman 组合；第三阶段做 rolling 样本外回测。一个重要原则是：未通过严格检验的因子不能直接决定权重，只能作为低置信度 scenario view。项目强调避免未来函数、计入交易成本、区分 gross 和 net 表现，并输出风险贡献、权重演化、换手率和回撤分析。

### 3 分钟介绍

本项目的研究动机是：多资产组合能够表达宏观环境变化，例如黄金和国债偏防御，纳斯达克和比特币偏风险资产，原油反映通胀和商品周期。传统均值-方差优化对历史收益均值非常敏感，因此我使用 Black-Litterman，从市场权重反推出均衡收益，再把宏观观点和研究型因子以低置信度 views 加入。

项目分三阶段。Stage 1 构建收益和因子，包括黄金中美时差、VIX 变化、油价动量、BTC 相关性等，并做严格审查。Stage 2 构建静态组合，包括 No-view BL、Conservative BL、Exploratory BL 和 Custom Factor Views BL。Stage 3 做 rolling out-of-sample backtest，使用 252/504 日窗口、月度/季度调仓、交易成本、换手率和净值曲线，验证不同组合方法的稳健性。

我特别强调不使用未来数据。每个调仓日只用过去收益和过去因子；custom factor 必须 lag；因子不会直接改权重，只会通过 BL views 影响 posterior expected returns。最终结果显示，Minimum Variance 和 Risk Parity 在风险调整后表现较稳健，这并不代表 BL 失败，而说明在当前数据和 views 置信度下，防御型组合更稳健。Conservative BL 仍适合作为可解释主模型，Custom BL 适合作为用户情景分析。

### 常见追问

**为什么用 Black-Litterman？**  
因为历史收益均值不稳定，直接均值-方差优化容易产生极端权重。BL 能结合市场均衡和主观观点，并通过置信度控制观点影响。

**为什么 Minimum Variance 表现更好？**  
样本期经历了高波动和危机阶段，低波动组合在风险调整后表现更稳健。这是合理结果，不代表 BL 框架无效。

**为什么 custom factor 没有直接用来交易？**  
因为严格统计检验没有证明它是高置信度 alpha，尤其中国黄金因子还有数据源和时区风险。因此只能作为低置信度 scenario view。

**项目创新点是什么？**  
完整串联了因子工程、严格因子审查、BL views、静态组合和 rolling 回测，并把 research-only 因子放在可控的低置信度框架里，而不是直接当信号。
