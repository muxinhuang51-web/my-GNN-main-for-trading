# A 股关系图嵌入与动态簇级轮动研究

本项目探索一种面向 A 股市场的量化研究框架：先用关系图神经网络学习股票之间的结构化表示，再将股票嵌入动态聚类为若干“隐含板块”或“功能簇”，最后在簇级别进行收益预测与组合轮动。

项目当前处于论文初稿、实验整理和方法验证阶段。当前结果展示了研究方向的潜力，但尚不能被视为稳定实盘策略结论。

## 1. 研究问题

A 股市场中，个股收益预测通常面临三个困难：

1. 个股日收益信噪比较低，直接做 TopK 个股预测容易受到噪声影响。
2. 股票之间并非独立样本，行业、概念、相关性和信息传导关系会影响收益。
3. 市场结构具有非平稳性，静态行业分类难以刻画短期主题轮动和风格切换。

本项目关注的问题是：

> 能否先用图神经网络学习股票之间的关系结构，再通过动态聚类形成更稳定的簇级投资单元，从而缓解个股预测噪声并提升组合构建效果？

## 2. 方法概览

核心流程如下：

```text
历史收益与关系数据
  -> 构造股票关系图
  -> RGCN 学习股票 embedding
  -> KMeans 动态聚类形成股票簇
  -> MLP 预测簇级收益
  -> 按预测簇收益排序
  -> 逐簇加入股票并等权回测
```

其中：

- 节点表示：使用滚动历史收益特征，包括 20 日动量、5 日均值、10 日均值、20 日波动率、60 日波动率和最近一日收益。
- 图结构：当前主要考虑行业边、概念边和收益相关性边。
- 图模型：使用双层 RGCN 学习股票嵌入。
- 聚类方法：每日对股票 embedding 进行 KMeans 聚类。
- 组合构建：按预测簇收益排序，逐簇加入股票，直到达到目标持仓规模。

## 3. 当前贡献定位

本项目并不声称首次将图神经网络用于股票预测。已有研究已经广泛使用 GNN、异构图、动态图、超图和知识图谱进行股票收益或涨跌预测。

本项目的差异在于：

1. **从个股预测转向簇级轮动**  
   不是直接依赖个股收益 TopK，而是将图嵌入转化为动态簇，再在簇级别预测和选取组合。

2. **将图嵌入解释为动态市场结构**  
   聚类结果可以被视为模型学习出的隐含板块，既不同于静态行业分类，也不同于单纯收益相关性分组。

3. **关注关系信息如何转化为组合构建单元**  
   研究重点不是继续堆叠更复杂的预测器，而是探索图表示、聚类和组合轮动之间的连接。

## 4. 初步实验发现

当前实验以约 5200 只 A 股为股票池，使用滚动收益特征和 RGCN 嵌入进行簇级轮动回测。

### 4.1 个股级 baseline

同一 RGCN 模型直接进行个股收益预测时，短期回测表现较差：

| 方法 | 年化收益 | 年化波动 | Sharpe | 最大回撤 |
|------|----------|----------|--------|----------|
| 个股级 RGCN baseline | -18.3% | 25.3% | -0.72 | -17.0% |

这说明在当前数据和模型设置下，直接进行个股级预测并不稳定。

### 4.2 簇级轮动结果

在扩展回测区间中，较稳定的主结果如下：

| 设置 | 回测天数 | 年化收益 | Sharpe | 最大回撤 | 平均簇级 IC |
|------|----------|----------|--------|----------|-------------|
| cluster_count=20, target=100, seed=42 | 86 | 54.5% | 1.74 | -16.6% | 0.080 |

该结果相对于个股级 baseline 有明显改善，说明“图嵌入 + 动态聚类 + 簇级预测”的组合构建方式具有进一步研究价值。

### 4.3 谨慎解释

当前结果仍需要谨慎看待：

- 86 天仍然是较短样本，年化收益和 Sharpe 可能被样本阶段放大。
- 尚未完整加入交易成本、滑点、换手率和容量约束。
- 不同聚类数、随机种子和回测区间会显著影响结果。
- 46 天短样本中曾出现 Sharpe 3.57 的结果，但扩展到 86 天后明显衰减，因此不应作为主结论。
- 相关性边在当前实验中没有稳定提升收益，反而经常降低表现，这是后续需要重点解释的现象。

因此，当前更稳妥的结论是：

> 簇级轮动框架在本文当前数据和回测设定下优于个股级 RGCN baseline，但是否具有长期稳定性，还需要更长区间、更多基准和更严格交易约束验证。

## 5. 与已有研究的关系

相关领域已有较多工作，包括：

- RSR: Temporal Relational Ranking for Stock Prediction
- HATS: Hierarchical Graph Attention Network for Stock Movement Prediction
- HIST: Graph-based Stock Trend Forecasting via Concept-Oriented Shared Information
- STHAN-SR: Spatiotemporal Hypergraph Attention Network
- THGNN: Temporal and Heterogeneous Graph Neural Network
- MDGNN: Multi-Relational Dynamic Graph Neural Network

这些方法多将关系图用于个股涨跌预测、收益排序或 TopK 选股。本文希望补充的角度是：将图神经网络学习到的股票关系表示进一步组织成动态簇，并在簇级别进行收益预测和轮动组合构建。

更详细的相关工作对比见 [paper-draft.md](paper-draft.md) 第 6 节。

## 6. 项目文件结构

```text
.
├── README.md                       # 项目说明
├── paper-draft.md                  # 论文草稿
├── paper-plan.md                   # 论文推进计划
├── math-derivation.md              # 数学推导与符号说明
├── experiment-summary.md           # 当前实验结果汇总
├── figures/
│   ├── method_flowchart.mmd         # 方法流程图 Mermaid 源文件
│   └── method_flowchart.png         # 方法流程图图片
├── backtest_cluster.py             # 当前簇级轮动主回测脚本
├── longrun_backtest.py             # 扩展回测实验脚本
├── optimize_cluster_backtest.py     # 有边界的参数搜索脚本
├── models/
│   ├── embedding_model.py           # RGCN embedding 导出
│   └── cluster_predictor.py         # 簇级收益预测器
├── main/                           # 原始 notebook 探索流程
├── stractegy/                      # 策略设计和理论笔记
├── data/                           # 本地数据
└── outputs/                        # 回测输出和实验结果
```

建议阅读顺序：

1. [README.md](README.md)：快速了解项目。
2. [paper-draft.md](paper-draft.md)：查看论文当前草稿。
3. [math-derivation.md](math-derivation.md)：查看公式和数学定义。
4. [experiment-summary.md](experiment-summary.md)：查看实验结果和风险提示。
5. [backtest_cluster.py](backtest_cluster.py)：查看主回测实现。

## 7. 运行方式

推荐使用 Conda 环境：

```bash
conda activate paper
```

主要依赖包括：

```text
python
numpy
pandas
torch
torch-geometric
scikit-learn
matplotlib
tushare
```

运行一次默认簇级回测：

```bash
python backtest_cluster.py
```

默认输出目录：

```text
outputs/cluster_backtest/
```

运行参数搜索：

```bash
python optimize_cluster_backtest.py
```

参数搜索脚本设置了最大实验次数和最长运行时间，避免无边界搜索。

## 8. 数据与输出说明

主要输入数据：

- `data/daily_returns.csv`：日收益率矩阵。
- `data/industry_mapping.csv`：股票行业映射。
- `data/stock_pool_all.csv`：股票池。
- `best_model.pt`：已训练 RGCN 权重。

主要输出：

- `metrics.json`：回测汇总指标。
- `daily_returns.csv`：每日组合收益。
- `daily_decisions.jsonl`：每日选簇和选股明细。
- `run_config.json`：本次回测配置。
- `run_summary.json`：参数与结果摘要。
- `cum_return.png`、`daily_return.png`、`drawdown.png`：回测图表。

说明：`outputs/` 中包含多批探索性实验，其中部分来自旧代码版本或不同参数记录方式。正式论文结果应以固定代码版本、固定参数和包含 `run_config.json` 的复现实验为准。

## 9. 当前局限

当前项目仍存在以下限制：

1. 回测样本仍偏短，需要扩展到更长时间区间。
2. 尚未完整纳入交易成本、滑点、换手率和容量约束。
3. KMeans 随机种子对结果有一定影响，需要更多种子验证。
4. 相关性边在当前实现下表现不稳定，需要进一步解释其负贡献原因。
5. 预训练 RGCN 和簇级预测器尚未联合优化。
6. 与外部论文不能直接比较收益指标，需要在统一数据和统一交易设定下复现基准。

## 10. 下一步计划

短期计划：

1. 冻结一批论文正式实验，统一保存到 `outputs/paper_experiments/`。
2. 加入交易成本、滑点、换手率和容量约束。
3. 补充传统量化基准：全市场等权、行业等权、动量 TopN、低波动 TopN。
4. 补充模型基准：无图 MLP/LSTM、RGCN 个股 TopK、无聚类版本。
5. 对动态簇进行解释：行业分布、概念分布、簇内股票、簇轮动路径。
6. 尝试引入 LLM/文本语义边，但作为扩展实验而非当前主结论。

中期目标：

> 形成一篇围绕“关系图嵌入驱动的动态簇级轮动”的论文初稿，并用更严谨的复现实验验证该框架是否具有稳定的组合构建价值。
