# 常见问题 FAQ

## 1. 这个项目能直接实盘吗？

不能。它是研究和课程项目框架，不是完整实盘系统。还需要处理滑点、手续费、保证金、税费、期货换月、实时数据和风控执行。

## 2. 为什么 approved factors 是空的？

严格审查标准较高。当前因子虽然有经济逻辑，但统计稳定性、样本外表现或数据源可信度不足，因此没有被标为高置信度 approved。

## 3. research-only 因子还有价值吗？

有。它们可以作为低置信度 scenario views，用于 Black-Litterman 情景分析，但不能直接当成交易信号。

## 4. 为什么 custom factor 不能直接改权重？

因为这会绕过因子检验和组合风险约束。项目原则是：因子只能影响 BL views，再通过 posterior expected returns 和优化器影响权重。

## 5. 中国黄金因子为什么风险高？

当前数据更像 SGE proxy，不是确认的 SHFE AU 黄金期货；美国黄金收益也是日频 close-to-close proxy，无法完全验证盘中领先关系。此外 lead-1 alignment 异常强，提示潜在对齐风险。

## 6. 为什么 Minimum Variance 表现好？

在高波动样本期，低波动组合可能有更高 Sharpe 和较低回撤。这是合理现象，不代表 BL 失败。

## 7. Conservative BL 为什么仍是主模型？

因为它最可解释、最稳健，不依赖未验证因子。项目不是只追求样本内最高收益，而是强调研究透明度和风险控制。

## 8. 如何添加新资产？

需要：

1. 在 `config.py` 增加 ticker；
2. 修改收益率构建；
3. 在 BL 资产列表中加入资产；
4. 更新权重约束；
5. 更新图表和报告解释。

## 9. 如何添加新因子？

请先阅读：

```text
docs/如何添加新因子.md
```

## 10. 图表没有生成怎么办？

部分图表需要对应数据。如果数据为空，脚本会 warning 并继续运行。先检查 CSV 是否存在，再检查是否有足够样本。

## 11. 为什么 rolling backtest 和静态 ex-ante 表现不同？

静态组合是 ex-ante 估计，不是历史持有收益。Rolling backtest 每个调仓日重新估计输入，并用后续样本外收益验证。

## 12. 交易成本怎么算？

当前使用单边 10 bps：

```text
transaction_cost = turnover * 0.001
turnover = sum(abs(new_weight - old_weight))
```

第一次建仓也计入 turnover。

## 13. risk overlay 是否启用？

默认关闭。当前主回测是无止盈止损版本。启用后需要单独审查风控路径。

## 14. 如何判断是否有未来函数？

查看：

- `stage3_backtest_audit_report.md`
- `rolling_backtest_custom_factor_log.csv`
- `rolling_backtest_view_log.csv`

每个调仓日应该只使用过去数据。

