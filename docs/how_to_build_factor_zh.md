# 怎么做一个因子

这个项目里的因子基本都遵循同一个模式：

```python
def add_xxx_factors(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 检查输入列是否存在
    # 2. 用已有列计算新因子列
    # 3. 返回同一个 df
    return df
```

## 1. 先写清楚经济假设

不要先写代码，先用一句话说明因子想表达什么。

例子：

- VIX 上升代表风险偏好下降，可能压低 Nasdaq。
- BTC 过去 20 日上涨，可能代表短期动量延续。
- 中国黄金 morning-to-evening 上涨，可能代表中国交易时段释放了偏多黄金信息。

## 2. 明确输入列

每个因子都应该知道自己需要哪些输入列。

例子：

```python
check_required_columns(df, ["nasdaq", "treasury"], "add_equity_factors")
```

如果缺列，就不要硬算。硬算会报错，也容易产生错误结果。

## 3. 只用当前或过去信息

构造因子时最重要的是避免未来函数。

可以用：

```python
df["nasdaq_mom_20d"] = safe_rolling_sum(df["nasdaq"], 20)
```

不要在因子列里使用未来收益，例如不要这样做：

```python
df["bad_factor"] = df["gold_us"].shift(-1)
```

`shift(-1)` 通常只能用于创建 target，例如 `gold_next_ret`，不能用于创建 factor。

## 4. 常见因子写法

### 动量因子

```python
df["asset_mom_20d"] = safe_rolling_sum(df["asset_return"], 20)
```

含义：过去 20 个观测日累计收益。

### 变化因子

```python
df["vix_change"] = df["vix_level"].diff()
```

含义：今天 VIX 水平相对上一次观测的变化。

### 利差因子

```python
df["yield_curve_10y_2y"] = df["DGS10"] - df["DGS2"]
```

含义：长端利率和短端利率的差。

### 滚动相关因子

```python
df["btc_nasdaq_corr_60d"] = safe_rolling_corr(df["btc"], df["nasdaq"], 60)
```

含义：BTC 最近 60 天是否更像 Nasdaq 这类风险资产。

### 交互项因子

```python
df["oil_vix_interaction"] = df["oil"] * df["vix_change"]
```

含义：原油收益和风险情绪同时变化时的状态。

### z-score 因子

```python
rolling_mean = signal.rolling(252, min_periods=126).mean()
rolling_std = signal.rolling(252, min_periods=126).std()
df["signal_z"] = (signal - rolling_mean) / rolling_std
```

含义：今天的信号相对过去一年历史分布有多极端。

## 5. 把因子接入项目

通常有四步：

1. 在 `src/factors/xxx_factors.py` 里新增因子列。
2. 确认 `build_clean_factor_panel.py` 调用了这个模块。
3. 在质量报告配置里加入因子和目标收益的映射。
4. 用相关性、分位数组、样本外测试检查它是否有意义。

## 6. 判断因子好不好

至少看这些：

- `non_null_count`: 有效样本够不够。
- `missing_rate`: 缺失率是否太高。
- `spearman_corr`: 排序相关是否符合预期。
- `q5_minus_q1_spread`: 高分位组是否比低分位组表现好。
- `rolling IC`: 是否稳定，而不是只在某段历史有效。
- 是否有 look-ahead bias。

## 7. 本项目里的一个完整例子

中国黄金新版 custom factor：

```python
china_gold_morning_to_evening_ret = china_gold_evening / china_gold_morning - 1
china_gold_morning_to_evening_z = rolling_zscore(china_gold_morning_to_evening_ret)
```

经济含义：

中国黄金当天从 morning 到 evening 明显上涨，代表中国交易时段可能释放了偏多黄金的信息。

注意：

因为当前没有美国黄金 intraday open/reopen 价格，所以只能用 daily close-to-close proxy 检验，不能严格证明美国高开。
