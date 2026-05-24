# 今晚初版实施计划（基于已有节点特征）

## 1. 目标
- 基于已有的 5000 个股票节点和 10 维特征，构建包含 3 种基础边类型的异构图。
- 使用 RGCN 进行节点二分类（预测下一交易日涨跌）。
- 跑通全流程，输出验证集 AUC 作为基线。
- 为后续扩展（增加新关系、LLM 文本表征）奠定代码框架。

## 2. 假设与前序准备
- 已有特征矩阵 `node_feats_allx10.npy`，形状 `(5000, 10)`，按某种股票顺序对齐（建议统一使用 `stock_code` 排序）。
- 日线行情数据待获取（至少包含 `close` 或 `return`），用于计算相关性边和标签。
- 申万一级行业分类待获取（或概念板块数据）。
- 软件环境：PyTorch + PyG 已安装，GPU 可用。

## 3. 边类型选择（今晚使用）
| 边类型 | 简称 | 方向 | 数据源 | 更新频率 |
|--------|------|------|--------|----------|
| `(stock, same_industry, stock)` | `same_industry` | 无向 | 申万一级行业（最新分类） | 季度/按需 |
| `(stock, same_concept, stock)` | `same_concept` | 无向 | 同花顺/Wind 概念板块 | 月度/按需 |
| `(stock, high_corr, stock)` | `high_corr` | 无向 | 过去 20 日日收益率相关系数 Top-K | 日频 |

**备注**：
- `same_industry` 和 `same_concept` 提供静态关联，反映长期信息传播路径。
- `high_corr` 提供动态关联，反映短期资金和情绪传染。
- 所有边均做稀疏化：每个节点保留 K=20 条最重要的边。

## 4. 实施步骤

### 4.1 数据预处理（约 1 小时）

#### a. 对齐节点顺序
- 统一 `stock_code` 列表，确保特征矩阵、行业分类、行情数据中的股票顺序一致。
- 保存映射字典 `stock2idx`。

#### b. 构建静态边
- 读取申万一级行业分类，生成同行业配对。
- 读取概念板块数据（如同花顺概念），生成同概念配对。
- 对每条边去重，转换为 `edge_index`（形状 `[2, E]`）。
- 可选：若同时属于同行业和同概念，可保留两条边或合并权重，今晚简单起见分开保留。

#### c. 构建动态边（`high_corr`）
- 取最近 20 个交易日的日收益率矩阵（N_stock × 20），计算皮尔逊相关系数矩阵。
- 对每只股票，保留相关系数最高的 20 个邻居（去除自身）。
- 得到 `edge_index` 和可选的 `edge_weight`（相关系数值，softmax 归一化）。

#### d. 生成标签
- 计算下一交易日收益率 `r(t+1)`，定义标签：`r(t+1) > 0` 为 1，否则为 0。
- 标签与节点对齐。

#### e. 数据集划分
- 按照日期划分训练/验证/测试集（例如：最近 N 天之前的为训练，最近 N 天的一部分为验证）。
- 今晚可简单使用“最近一个交易日”作为验证集，再往前 20 个交易日作为训练集（由于节点数 5000，样本量足够）。
- 构建 `train_mask` 和 `val_mask` 布尔向量。

### 4.2 构建异构图（约 0.5 小时）
```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['stock'].x = node_feats                     # [N, 10]
data['stock'].y = labels                         # [N]
data['stock'].train_mask = train_mask
data['stock'].val_mask = val_mask

data['stock', 'same_industry', 'stock'].edge_index = ind_edge_index
data['stock', 'same_concept', 'stock'].edge_index = concept_edge_index
data['stock', 'high_corr', 'stock'].edge_index = corr_edge_index
data['stock', 'high_corr', 'stock'].edge_weight = corr_weights  # 可选
```
- 将所有关系映射为整数 ID：`edge_type_dict = {('stock','same_industry','stock'):0, ...}`，用于 RGCN。

### 4.3 模型定义（复用已有代码，约 15 分钟）
```python
from torch_geometric.nn import RGCNConv

class StockPredictor(nn.Module):
    def __init__(self, in_dim=10, hidden=64, num_relations=3, num_classes=2):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index_dict, edge_type_dict):
        x = self.conv1(x, edge_index_dict, edge_type_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict, edge_type_dict)
        return self.classifier(x)
```

### 4.4 训练与验证（约 1 小时）
- 使用 `Adam`（lr=0.001），`CrossEntropyLoss`（可加 `pos_weight` 处理类不平衡）。
- 每 epoch 计算训练损失和验证 AUC/准确率。
- 训练约 100 epochs，观察验证指标，保存最佳模型。
- 由于是单图全量训练，可直接使用整张图。

### 4.5 结果记录与交付
- 输出验证集 AUC、Accuracy、Top-K 组合模拟收益（可选）。
- 保存 `model.pt`、`data.pt`、`metrics.json`。
- 记录所用超参数、边类型及版本。

## 5. 时间线总控（约 3 小时）
| 任务 | 预计耗时 |
|------|----------|
| 数据预处理（对齐、静态边、动态边、标签） | 1h |
| 图构建与模型定义 | 0.5h |
| 训练与调参 | 1h |
| 评估与记录 | 0.5h |

## 6. 潜在问题与应对
- **类别不平衡**：如果下跌样本过少，使用 `pos_weight` 或 Focal Loss。
- **显存不足**：确认边索引类型为 Long，且关系数少，5000 节点全连接约几百万边，Top-20 约 10 万边，显存占用极小。
- **验证集过于乐观**：确保严格按时间划分，避免未来信息泄漏。
- **特征尺度差异**：对所有 10 个特征做时序标准化（z-score），使用过去 252 天的统计量。

## 7. 后续衔接
- 今晚跑通后，明天可立即引入新边（如 `leader_follower`、`limit_overflow`）或节点文本表征，只需在 `HeteroData` 中添加新关系或在 `x` 后拼接新特征，无需改动模型框架。
```

这份计划足够具体，今晚可执行。```markdown
# 今晚初版实施计划（基于已有节点特征）

## 1. 目标
- 基于已有的 5000 个股票节点和 10 维特征，构建包含 3 种基础边类型的异构图。
- 使用 RGCN 进行节点二分类（预测下一交易日涨跌）。
- 跑通全流程，输出验证集 AUC 作为基线。
- 为后续扩展（增加新关系、LLM 文本表征）奠定代码框架。

## 2. 假设与前序准备
- 已有特征矩阵 `node_feats.npy`，形状 `(5000, 10)`，按某种股票顺序对齐（建议统一使用 `stock_code` 排序）。
- 已有日线行情数据（至少包含 `close` 或 `return`），用于计算相关性边和标签。
- 已有申万一级行业分类（或概念板块数据）。
- 软件环境：PyTorch + PyG 已安装，GPU 可用。

## 3. 边类型选择（今晚使用）
| 边类型 | 简称 | 方向 | 数据源 | 更新频率 |
|--------|------|------|--------|----------|
| `(stock, same_industry, stock)` | `same_industry` | 无向 | 申万一级行业（最新分类） | 季度/按需 |
| `(stock, same_concept, stock)` | `same_concept` | 无向 | 同花顺/Wind 概念板块 | 月度/按需 |
| `(stock, high_corr, stock)` | `high_corr` | 无向 | 过去 20 日日收益率相关系数 Top-K | 日频 |

**备注**：
- `same_industry` 和 `same_concept` 提供静态关联，反映长期信息传播路径。
- `high_corr` 提供动态关联，反映短期资金和情绪传染。
- 所有边均做稀疏化：每个节点保留 K=20 条最重要的边。

## 4. 实施步骤

### 4.1 数据预处理（约 1 小时）

#### a. 对齐节点顺序
- 统一 `stock_code` 列表，确保特征矩阵、行业分类、行情数据中的股票顺序一致。
- 保存映射字典 `stock2idx`。

#### b. 构建静态边
- 读取申万一级行业分类，生成同行业配对。
- 读取概念板块数据（如同花顺概念），生成同概念配对。
- 对每条边去重，转换为 `edge_index`（形状 `[2, E]`）。
- 可选：若同时属于同行业和同概念，可保留两条边或合并权重，今晚简单起见分开保留。

#### c. 构建动态边（`high_corr`）
- 取最近 20 个交易日的日收益率矩阵（N_stock × 20），计算皮尔逊相关系数矩阵。
- 对每只股票，保留相关系数最高的 20 个邻居（去除自身）。
- 得到 `edge_index` 和可选的 `edge_weight`（相关系数值，softmax 归一化）。

#### d. 生成标签
- 计算下一交易日收益率 `r(t+1)`，定义标签：`r(t+1) > 0` 为 1，否则为 0。
- 标签与节点对齐。

#### e. 数据集划分
- 按照日期划分训练/验证/测试集（例如：最近 N 天之前的为训练，最近 N 天的一部分为验证）。
- 今晚可简单使用“最近一个交易日”作为验证集，再往前 20 个交易日作为训练集（由于节点数 5000，样本量足够）。
- 构建 `train_mask` 和 `val_mask` 布尔向量。

### 4.2 构建异构图（约 0.5 小时）
```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['stock'].x = node_feats                     # [N, 10]
data['stock'].y = labels                         # [N]
data['stock'].train_mask = train_mask
data['stock'].val_mask = val_mask

data['stock', 'same_industry', 'stock'].edge_index = ind_edge_index
data['stock', 'same_concept', 'stock'].edge_index = concept_edge_index
data['stock', 'high_corr', 'stock'].edge_index = corr_edge_index
data['stock', 'high_corr', 'stock'].edge_weight = corr_weights  # 可选
```
- 将所有关系映射为整数 ID：`edge_type_dict = {('stock','same_industry','stock'):0, ...}`，用于 RGCN。

### 4.3 模型定义（复用已有代码，约 15 分钟）
```python
from torch_geometric.nn import RGCNConv

class StockPredictor(nn.Module):
    def __init__(self, in_dim=10, hidden=64, num_relations=3, num_classes=2):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index_dict, edge_type_dict):
        x = self.conv1(x, edge_index_dict, edge_type_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict, edge_type_dict)
        return self.classifier(x)
```

### 4.4 训练与验证（约 1 小时）
- 使用 `Adam`（lr=0.001），`CrossEntropyLoss`（可加 `pos_weight` 处理类不平衡）。
- 每 epoch 计算训练损失和验证 AUC/准确率。
- 训练约 100 epochs，观察验证指标，保存最佳模型。
- 由于是单图全量训练，可直接使用整张图。

### 4.5 结果记录与交付
- 输出验证集 AUC、Accuracy、Top-K 组合模拟收益（可选）。
- 保存 `model.pt`、`data.pt`、`metrics.json`。
- 记录所用超参数、边类型及版本。

## 5. 时间线总控（约 3 小时）
| 任务 | 预计耗时 |
|------|----------|
| 数据预处理（对齐、静态边、动态边、标签） | 1h |
| 图构建与模型定义 | 0.5h |
| 训练与调参 | 1h |
| 评估与记录 | 0.5h |

## 6. 潜在问题与应对
- **类别不平衡**：如果下跌样本过少，使用 `pos_weight` 或 Focal Loss。
- **显存不足**：确认边索引类型为 Long，且关系数少，5000 节点全连接约几百万边，Top-20 约 10 万边，显存占用极小。
- **验证集过于乐观**：确保严格按时间划分，避免未来信息泄漏。
- **特征尺度差异**：对所有 10 个特征做时序标准化（z-score），使用过去 252 天的统计量。

## 7. 后续衔接
- 今晚跑通后，明天可立即引入新边（如 `leader_follower`、`limit_overflow`）或节点文本表征，只需在 `HeteroData` 中添加新关系或在 `x` 后拼接新特征，无需改动模型框架。
```