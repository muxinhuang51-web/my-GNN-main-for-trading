# 今天下午工作计划（回归版）：基于异构图完成 RGCN 回归模型训练与评估

## 1. 背景与目标
- **成果**：已通过 `build_hetero_graph()` 生成 `HeteroData`，包含 3 类关系（`same_industry`, `same_concept`, `high_corr`），并保存为 `data/hetero_edges.pt`。
- **下午目标**：基于该图，实现 RGCN **回归**模型，直接预测下一交易日收益率（连续值），跑通训练-验证全流程，产出第一个基线结果（验证集 IC / MSE），并为后续扩展预留接口。
- **总耗时预估**：4 小时（14:00 - 18:00）。

---

## 2. 前置条件检查（14:00 - 14:15）
- 确认 `data/hetero_edges.pt` 可正常加载，包含：
  - `data['stock'].x`: (5000, 10) 特征矩阵
  - 三类 `edge_index`（`same_industry`, `same_concept`, `high_corr`）
  - 可选：`high_corr` 的 `edge_weight`
- 确认已有一份日线行情数据，可用于生成标签（下一交易日收益率）。
- 确认 Python 环境：PyTorch, PyG, sklearn, numpy, pandas 已安装，GPU 可用。

---

## 3. 任务分解

### 3.1 数据准备：标签生成与时间分割（14:15 - 15:00）
**目标**：生成每个交易日的连续收益率标签，并按时间顺序划分训练/验证/测试集。

**步骤**：
1. 加载全量日收益率数据，计算下一交易日收益率：
   \[
   y = \frac{P_{t+1} - P_t}{P_t}
   \]
   （直接保留原始连续值，不做二值化）
2. 标签即为 `future_ret`，设为 `data['stock'].y`。
3. 考虑**时序划分**（避免未来信息）：
   - 取最近 250 个交易日作为回看窗口。
   - 生成多张“快照图”（或按日期切片），每张图使用当日 T 的过去信息（特征、边），标签为 T+1 的收益率。
   - 划分：前 200 天作为**训练集**，后 50 天作为**验证集**。
   - 对于单张图：若简化，可只挑两个日期（一个较早的 T_train，一个较晚的 T_val），分别构建两个 `HeteroData` 对象。
4. 将标签和 `train_mask`/`val_mask` 存入对应图的 `data` 对象。
5. 保存 `data/train_graph.pt` 和 `data/val_graph.pt`（或直接扩展原有图对象）。

**产出**：`data/train_graph.pt`, `data/val_graph.pt`（或包含 mask 的完整图）。

**关键提醒**：
- 必须使用**过去**数据构建特征和边。相关性边计算窗口为 T-19 到 T（不含 T+1）。
- 特征标准化：用训练集特征的均值和标准差，对验证集进行 z-score 变换。
- 若某股票在 T+1 日停牌导致收益率缺失，将其标签设为 NaN，并在 loss 计算时用 mask 排除。

### 3.2 模型搭建（15:00 - 15:30）
**目标**：定义两层的 RGCN 回归器，输出一个连续预测值。

**代码框架**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class StockRGCN_Regression(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=64, num_relations=3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.regressor = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index_dict, edge_type_dict):
        x = self.conv1(x, edge_index_dict, edge_type_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict, edge_type_dict)
        return self.regressor(x).squeeze(-1)  # [N] 形状
```

**关系映射**（需与图定义一致）：
```python
edge_type_dict = {
    ('stock', 'same_industry', 'stock'): 0,
    ('stock', 'same_concept', 'stock'): 1,
    ('stock', 'high_corr', 'stock'): 2
}
```

**产出**：notebook 中可运行的模型类。

### 3.3 训练循环与监控（15:30 - 17:00）
**目标**：训练模型，监控验证集 MSE 和 IC（信息系数），保存最佳模型。

**步骤**：
1. 实现训练函数，包含：
   - 损失函数：`MSELoss`（回归任务无需处理类别不平衡）
   - 优化器：Adam，学习率 0.001
   - 早停机制：验证集 IC（预测值与真实值的相关系数）连续 10 个 epoch 未提升则停止
2. 每个 epoch 记录：训练 loss、验证 loss（MSE）、验证 IC。
3. 可视化训练曲线（Loss 下降、IC 上升）。
4. 保存最佳模型权重 `best_model.pt`。

**评估指标**：
- **MSE / RMSE**：预测误差的绝对大小
- **IC（信息系数）**：`corr(pred, actual)`，核心排序指标，越大越好
- **ICIR**（多日期可算）：IC 均值 / IC 标准差，衡量稳定性
- **分层回测**（可选）：将验证集股票按预测值排序分 5 组，观察各组实际平均收益是否单调递增

**潜在问题与对策**：
| 问题 | 对策 |
|------|------|
| IC 接近 0 或不收敛 | 检查标签是否正确、特征是否标准化、边是否合理 |
| 预测值全部趋同（全预测同一个值） | 增大隐藏层维度、降低正则化、检查边是否过多导致过平滑 |
| 损失突然飙升 | 梯度裁剪 `torch.nn.utils.clip_grad_norm_` |
| 显存不足 | 确认图稀疏度，K=20 边数约 10 万，显存应 < 1 GB |
| 标签缺失（停牌） | 用 mask 排除 NaN 标签的节点 |

**产出**：`best_model.pt`, `training_curves.png`, `metrics.json`（含 MSE, IC）。

### 3.4 结果分析与记录（17:00 - 17:30）
**目标**：对基线模型结果进行整理，为后续迭代提供依据。

**内容**：
1. 记录超参数：hidden=64, lr=0.001, K=20, 三类边。
2. 对比消融实验（可选，若时间充裕）：
   - 只用 `same_industry` 边
   - 只用 `high_corr` 边
   - 使用全部三类边（当前基线）
3. 输出预测值与真实值的散点图，检查预测分布。
4. 分层回测：按预测值分 5 组计算各组实际平均收益率，验证单调性。
5. 将结果汇总到 `README.md` 或实验日志。

**产出**：`experiment_log.md`，包含关键数字和初步发现。

---

## 4. 后续扩展方向（预留接口）
- 增加新边类型（`leader_follower`, `limit_overflow`, `common_ownership`）
- 引入 LLM 文本表征（拼接新特征 + 构建 `text_sim` 边）
- 多日回测框架（逐日生成图，计算 ICIR 累积曲线）
- 尝试 HAN 自动学习关系权重

---

## 5. 注意事项
- **严禁未来信息**：所有特征、边的计算窗口必须严格在标签日期之前。
- **特征标准化**：使用训练集统计量，变换验证/测试集。
- **可重复性**：设置随机种子（`torch.manual_seed(42)`）。
- **数据对齐**：确认所有股票按统一代码顺序排列，避免特征与标签错位。
- **缺失值处理**：停牌或无交易股票，标签设为 NaN 并用 mask 排除；特征缺失值用 0 或行业中位数填充。
```

---

这样就把原来的分类方案完全转换成回归方案，输出目标、损失函数、评估指标、代码都对应修改了。你可以直接拿这个文档下午执行。