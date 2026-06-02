# A 股图神经网络交易研究项目

本项目围绕 A 股股票关系图建模，尝试用图神经网络预测个股或股票簇的未来收益，并通过回测验证选股效果。

当前仓库可以分成两层：

- `main/` 和 `stractegy/` 是研究主线，包含原始 notebook、策略设计和理论文档。
- `backtest_cluster.py` 和 `models/` 是工程化执行层，主要由 AI 从 notebook 和策略文档中抽取、整理而来。

## 目录结构

```text
.
├── main/                    # 研究 notebook 主线
├── stractegy/               # 策略设计、理论和执行文档
├── models/                  # 模型和簇级预测工具
├── data/                    # 本地数据和中间图数据
├── outputs/                 # 回测输出图表和指标
├── backtest_cluster.py      # 簇级轮动回测脚本
├── best_model.pt            # 已训练 RGCN 模型权重
├── feature_config.json      # 特征标准化配置
└── metrics.json             # 训练或评估指标
```

## 研究主线

`main/` 中的 notebook 按阶段组织：

- `main01.ipynb`：构建股票池和 10 维节点快照特征。
- `main02.ipynb`：构建行业、概念、相关性等图边，并生成收益率数据。
- `main03.ipynb`：基于滚动历史窗口构建训练样本，训练 RGCN 回归模型。
- `main04.ipynb`：个股级预测、选股、回测和图表导出。
- `main05.ipynb`：簇级链路演示，包括 embedding、KMeans 聚类、簇画像和簇收益预测。

`stractegy/` 中的文档记录策略想法：

- `stractegy01.md` 到 `stractegy04.md`：从图建模、训练到回测的阶段计划。
- `strctegy05.md`：图嵌入聚类和簇间轮动预测执行文档。
- `theory.md`：可用于构造图边的经济学领先-滞后关系。

## 工程化执行层

`models/embedding_model.py` 提供 RGCN 模型加载和节点 embedding 导出。

`models/cluster_predictor.py` 提供簇级样本构造、簇收益预测器训练和预测函数。

`backtest_cluster.py` 是簇级轮动回测入口。它会：

1. 读取 `data/daily_returns.csv` 和行业/概念映射。
2. 按交易日滚动构造历史特征和图边。
3. 加载 `best_model.pt` 并导出节点 embedding。
4. 对股票 embedding 做 KMeans 聚类。
5. 只用历史窗口训练簇收益预测器。
6. 在预测日按簇预测收益排序，逐簇加入直到达到目标有效持仓数。
7. 计算等权组合收益并保存回测结果。

## 环境

推荐使用 Conda 环境。当前本机使用的环境名是 `paper`：

```bash
conda activate paper
```

核心依赖包括：

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

如果环境丢失，需要重新安装 PyTorch、PyTorch Geometric、pandas、numpy、scikit-learn 等依赖。

## 运行方式

运行簇级回测：

```bash
python backtest_cluster.py
```

运行有边界的参数优化：

```bash
python optimize_cluster_backtest.py
```

优化脚本默认最多尝试 6 组参数、总时间最多 120 分钟、单组参数最多 20 分钟。每组参数会在独立子进程中运行，超时后会被终止，汇总结果写入：

```text
outputs/cluster_backtest_experiments/summary_bounded.json
```

如果只想快速验证流程，可以限制到一个交易日和一个 trial：

```bash
python optimize_cluster_backtest.py \
  --max-trials 1 \
  --max-minutes 10 \
  --trial-timeout-minutes 5 \
  --start-date 2026-05-22 \
  --end-date 2026-05-22
```

也可以在 Python 中指定较短日期范围做调试：

```python
import torch
import backtest_cluster as b

result_df, metrics = b.run_backtest(
    start_date="2026-02-06",
    end_date="2026-02-20",
    device=torch.device("cpu"),
)
print(metrics)
```

默认输出目录：

```text
outputs/cluster_backtest/
```

当前 `backtest_cluster.py` 的默认回测设置偏向可运行性和稳定性：

- 训练窗口 `train_window=20`，减少单日训练成本。
- 默认关闭相关性边 `top_neighbor_count=0`，避免相关矩阵和密集边拖慢回测。
- 聚类数 `cluster_count=20`。
- 按预测簇收益从高到低逐簇加入，直到目标有效持仓数 `target_portfolio_valid_stocks=100`。
- 跳过有效股票数不足 `min_market_valid_stocks=1000` 的交易日，避免数据残缺日期污染指标。

最近一次正式输出位于 `outputs/cluster_backtest/`，回测区间为 2026-03-16 到 2026-05-22，共 46 个有效交易日。当前指标：

```json
{
  "annualized_return": 1.4001169069532642,
  "sharpe": 2.8376923052578196,
  "max_drawdown": -0.11118583088512288,
  "days": 46,
  "mean_cluster_ic": 0.06939749818547629
}
```

注意：46 个交易日样本仍然偏短，指标只能说明当前数据片段下的有效性，后续应补充更长区间、交易成本、换手率和 CPU/GPU 一致性验证。

## 数据文件

主要数据文件包括：

- `data/stock_pool_all.csv`：股票池。
- `data/node_feats_allx10.csv` 和 `data/node_feats_allx10.npy`：节点特征。
- `data/industry_mapping.csv`：行业映射。
- `data/daily_returns.csv`：日收益率矩阵。
- `data/hetero_edges.pt`：异构图边数据。
- `data/train_graph.pt` 和 `data/val_graph.pt`：训练/验证图样本。

这些文件是本地研究数据或中间产物，不建议频繁提交大文件变动。

## 注意事项

- 回测必须严格使用历史窗口构造特征和边，避免未来函数。
- `main01.ipynb` 的最新快照特征适合当日推理，不适合直接做历史回测训练。
- `outputs/`、`.venv/`、`__pycache__/` 等生成文件不应提交到 Git。
- 若 VS Code Git 同步失败，先检查 `git status --short --branch`，确认是否存在本地和远端分叉。

## 当前状态

当前已整理：

- 删除本地虚拟环境目录 `.venv/`。
- 删除 Python 缓存目录 `__pycache__/` 和 `models/__pycache__/`。
- 新增 `.gitignore`，防止环境、缓存和输出产物再次进入 Git。
- 新增本 README，说明项目结构、环境和运行方式。
