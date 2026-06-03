# A 股图神经网络与簇级轮动研究项目

本项目研究一条 A 股量化策略链路：先用异构图神经网络学习股票之间的关系表示，再把股票 embedding 动态聚类成若干“功能簇”，最后在簇层面预测收益并做轮动回测。

项目当前处于论文初稿和实验整理阶段。请注意：`outputs/` 里有多批探索性实验，不能把所有结果当成同等可信的论文证据。

## 1. 一句话看懂

核心方法：

```text
股票收益和关系数据
  -> 构造行业/概念/相关性等异构图
  -> 加载已训练 RGCN 模型
  -> 导出股票 embedding
  -> KMeans 动态聚类
  -> 训练簇收益预测器
  -> 按预测收益选择簇
  -> 等权持有簇内股票并回测
```

核心入口：

- 跑一次簇级回测：`backtest_cluster.py`
- 有边界地搜索参数：`optimize_cluster_backtest.py`
- 论文计划和待补内容：`paper-plan.md`
- 理论依据：`stractegy/theory.md`

## 2. 当前目录结构

```text
.
├── main/                         # 原始研究 notebook 主线
├── stractegy/                    # 策略设计、理论依据和执行文档
├── models/                       # 工程化模型组件
│   ├── embedding_model.py         # 加载 RGCN 并导出股票 embedding
│   └── cluster_predictor.py       # 簇级收益预测器
├── data/                         # 本地数据和图数据
├── outputs/                      # 回测输出和实验结果
├── backtest_cluster.py           # 簇级轮动主回测脚本
├── optimize_cluster_backtest.py  # 有上限的参数搜索脚本
├── paper-plan.md                 # 论文推进计划
├── best_model.pt                 # 已训练 RGCN 权重
├── feature_config.json           # 特征配置
├── cluster_profile.csv           # 早期簇画像输出
└── README.md
```

## 3. 哪些文件是人写的，哪些是工程化整理

大致分层：

- `main/` 和 `stractegy/`：研究主线、原始探索、理论想法。
- `backtest_cluster.py`、`models/`、`optimize_cluster_backtest.py`：为了把 notebook 中的思路工程化运行而整理出的脚本。
- `outputs/`：生成结果，包含旧实验、夜间实验和正式回测输出。
- `paper-plan.md`：论文写作计划，不是代码。

## 4. 数据文件说明

主要数据：

- `data/daily_returns.csv`：日收益率矩阵，index 为交易日，columns 为股票代码。
- `data/industry_mapping.csv`：股票行业映射。
- `data/stock_pool_all.csv`：股票池。
- `data/node_feats_allx10.csv`、`data/node_feats_allx10.npy`：节点特征。
- `data/hetero_edges.pt`：早期异构图边数据。
- `data/train_graph.pt`、`data/val_graph.pt`：训练/验证图样本。

当前 `backtest_cluster.py` 主要依赖：

- `data/daily_returns.csv`
- `data/industry_mapping.csv`
- 可选概念文件：`data/stock_concept.csv`、`data/concept_mapping.csv` 或 `data/stock_concepts.csv`
- `best_model.pt`

如果概念文件不存在，脚本会返回空概念表，回测仍可运行。

## 5. 运行环境

推荐使用本机 Conda 环境：

```bash
conda activate paper
```

主要依赖：

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

如果 `python` 命令找不到，优先使用：

```bash
conda run -n paper python ...
```

## 6. 跑一次簇级回测

直接运行：

```bash
conda activate paper
python backtest_cluster.py
```

默认输出目录：

```text
outputs/cluster_backtest/
```

也可以在 Python 里指定较短区间调试：

```python
import torch
import backtest_cluster as b

result_df, metrics = b.run_backtest(
    start_date="2026-05-22",
    end_date="2026-05-22",
    out_dir="/tmp/cluster_backtest_smoke",
    device=torch.device("cuda"),
)
print(metrics)
```

## 7. 当前默认回测参数

`backtest_cluster.py` 当前默认值：

```text
lookback=60
train_window=20
top_neighbor_count=0
cluster_count=20
seed_value=42
min_cluster_valid_count=5
min_portfolio_valid_stocks=50
target_portfolio_valid_stocks=100
min_market_valid_stocks=1000
predictor_epochs=3
kmeans_n_init=1
```

关键解释：

- `top_neighbor_count=0`：默认关闭相关性边，只用行业边和概念边，保证回测速度和稳定性。相关性边仍然是论文设计的一部分，但需要作为消融实验单独比较。
- `min_market_valid_stocks=1000`：过滤掉有效股票数过少的交易日。当前数据最后几个交易日只有几十只股票有收益，不过滤会严重污染指标。
- `target_portfolio_valid_stocks=100`：按簇预测收益排序，逐簇加入，直到有效股票数达到目标。

## 8. 回测输出文件

每次新版 `backtest_cluster.py` 回测会输出：

```text
outputs/cluster_backtest/
├── metrics.json                  # 汇总指标
├── daily_returns.csv             # 每日组合收益和选股概览
├── daily_returns_partial.csv     # 每 10 天保存一次的中间结果
├── daily_decisions.jsonl         # 每日完整决策明细
├── daily_decisions_partial.jsonl # 中间决策明细
├── run_config.json               # 本次回测参数和数据范围
├── run_summary.json              # 参数 + 最终指标
├── cum_return.png                # 累计净值图
├── daily_return.png              # 日收益图
└── drawdown.png                  # 回撤图
```

重要判断：

- 有 `run_config.json` 的结果更容易追溯。
- 没有 `run_config.json` 的旧结果只能作为探索记录，不建议直接进论文主表。

## 9. 参数搜索脚本

`optimize_cluster_backtest.py` 用来自动尝试多组参数，但有明确停止边界。

默认运行：

```bash
python optimize_cluster_backtest.py
```

默认限制：

```text
max_trials=6
max_minutes=120
trial_timeout_minutes=20
```

包含相关性边实验：

```bash
python optimize_cluster_backtest.py --allow-corr-edges
```

快速 smoke test：

```bash
python optimize_cluster_backtest.py \
  --max-trials 1 \
  --max-minutes 10 \
  --trial-timeout-minutes 5 \
  --start-date 2026-05-22 \
  --end-date 2026-05-22 \
  --out-root /tmp/cluster_opt_smoke
```

输出目录示例：

```text
outputs/cluster_backtest_experiments/nightly_20260602_203126/
```

## 10. 当前实验结果应该怎么看

### 10.1 正式默认回测

路径：

```text
outputs/cluster_backtest/
```

当前指标：

```json
{
  "annualized_return": 1.4001169069532642,
  "sharpe": 2.8376923052578196,
  "max_drawdown": -0.11118583088512288,
  "days": 46,
  "mean_cluster_ic": 0.06939749818547629
}
```

该结果可作为当前工程主线的稳定基线。

### 10.2 夜间参数搜索

路径：

```text
outputs/cluster_backtest_experiments/nightly_20260602_203126/
```

这批实验有完整 `run_config.json` 和 `run_summary.json`，比早期实验更可追溯。

其中一个高表现结果：

```text
outputs/cluster_backtest_experiments/nightly_20260602_203126/c25_t100_tw20_e3_seed42/
```

指标：

```json
{
  "annualized_return": 1.4405517077434187,
  "sharpe": 3.5737923114517938,
  "max_drawdown": -0.07039054696795984,
  "days": 46,
  "mean_cluster_ic": 0.16946085935136304
}
```

但是：同名近似参数在旧批次中表现很差，因此该结果必须重新固定环境复现实验后，才能作为论文主结论。

### 10.3 早期探索实验

路径示例：

```text
outputs/cluster_backtest_experiments/c25_target100_corr0_seed42/
outputs/cluster_backtest_experiments/summary_2.json
```

这些目录多为早期探索输出，很多没有 `run_config.json`，可能来自不同代码版本、不同设备或不同参数记录方式。

处理原则：

- 可以用来回顾探索过程。
- 不建议直接作为论文主表。
- 如果发现和新实验冲突，以“当前代码 + 固定环境 + 有 run_config 的复现实验”为准。

## 11. 论文实验整理建议

为了避免输出目录越来越乱，论文正式实验建议另建：

```text
outputs/paper_experiments/
```

论文主实验只从这个目录取数。建议至少包含：

1. `cluster_c20_t100_corr0`
2. `cluster_c25_t100_corr0`
3. `cluster_c30_t100_corr0`
4. `cluster_c20_t100_corr5`
5. `cluster_c20_t100_corr10`
6. `cluster_c20_t150_corr0`
7. `cluster_c20_t200_corr0`
8. 个股级 baseline

每个实验必须保留：

- `run_config.json`
- `run_summary.json`
- `metrics.json`
- `daily_returns.csv`
- `daily_decisions.jsonl`

## 12. 研究主线 notebook

`main/` 中 notebook 的大致分工：

- `main01.ipynb`：构建股票池和 10 维节点快照特征。
- `main02.ipynb`：构建行业、概念、相关性等图边，并生成收益率数据。
- `main03.ipynb`：基于滚动历史窗口构建训练样本，训练 RGCN 回归模型。
- `main04.ipynb`：个股级预测、选股、回测和图表导出。
- `main05.ipynb`：簇级链路演示，包括 embedding、KMeans 聚类、簇画像和簇收益预测。

`stractegy/` 中重要文档：

- `stractegy/theory.md`：经济学领先-滞后关系和边类型理论。
- `stractegy/strctegy05.md`：图嵌入聚类与簇级轮动执行文档。

## 13. 写论文时的当前结论边界

可以说：

- 当前样本下，簇级轮动框架明显优于不使用聚类的个股级回测结果。
- 聚类数、目标持仓数、相关性边都会显著影响结果。
- 相关性边有理论意义，但当前实现下计算成本高，且在本轮实验中未稳定提升收益。

不要直接说：

- 模型已经稳定战胜市场。
- `cluster_count=25` 已经最终最优。
- 相关性边没有价值。

原因：

- 当前有效回测天数只有 46 天。
- 尚未加入交易成本、滑点和换手率。
- 旧实验和新实验存在冲突，需要正式复现实验冻结。

## 14. Git 和输出文件注意事项

当前 `.gitignore` 忽略了：

- `outputs/`
- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.vscode/`

如果需要保存论文关键结果，建议：

- 不直接依赖被忽略的 `outputs/`。
- 将汇总表、最终图片或论文用结果复制到单独的论文材料目录。
- 或在论文中明确记录实验目录、代码版本和运行参数。

常用检查命令：

```bash
git status --short --branch
find . -type d -name '__pycache__'
```

## 15. 下一步最建议做什么

短期目标是给导师看一版逻辑完整的论文初稿：

1. 先按 `paper-plan.md` 写论文骨架。
2. 暂时把 `outputs/cluster_backtest/` 作为稳定基线。
3. 把夜间搜索结果作为“参数敏感性参考”。
4. 之后单独跑一批 `outputs/paper_experiments/`，冻结论文正式结果。
