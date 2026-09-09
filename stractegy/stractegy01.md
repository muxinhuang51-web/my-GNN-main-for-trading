# A股 5000 股票图神经网络建模方案与实施计划

## 1. 目标与约束
- **目标**：构建一个可表达多种关系、支持动态更新、高可扩展性的股票图神经网络，今晚产出第一版可运行模型。
- **规模**：~5000 只 A 股节点，初始特征 ≤ 10 维。
- **扩展方向**：后续接入 LLM 生成的金融文本表征，增加节点特征或新边类型。

## 2. 图结构选型 —— 带多种边类型的异构图
### 2.1 节点类型
- **`stock`**：约 5000 个节点，每股对应一个节点。

### 2.2 边类型（关系）
每种关系定义为独立的边类型，互不干扰，可随时增删：
- `(stock, same_industry, stock)`：同申万一级行业（无向边，长期稳定）。
- `(stock, high_corr, stock)`：基于 20 日收益率相关性的动态边（有向/无向均可，每日更新）。
- 后续可扩展：`same_supply_chain`、`text_sim`、`same_concept` 等。

### 2.3 为何选异构图
- 多关系天然隔离，每种关系可学习独立的变换权重。
- 动态更新只需替换特定关系的边索引和权重，不影响全局结构。
- 扩展新关系仅需新增一个边类型，无需重构整个图。

### 2.4 实现框架
**PyTorch Geometric (PyG)** 的 `HeteroData` + `RGCNConv`（关系图卷积网络），版本 ≥ 2.3。  
DGL 亦可，但 PyG 更轻量且与 RGCN 集成更直接。

---

## 3. 节点初始特征（≤10 维，今晚可直接使用）
选取容易获取的基本面与量价因子，缺失值用 0 或行业中位数填充：

| 序号 | 特征名            | 含义               |
|------|-------------------|--------------------|
| 1    | ln_cap            | 对数总市值         |
| 2    | beta              | 60日市场 Beta      |
| 3    | momentum_20d      | 过去20日收益率     |
| 4    | volatility_60d    | 60日波动率         |
| 5    | turnover_20d      | 20日平均换手率     |
| 6    | pe_ttm            | 市盈率 TTM         |
| 7    | pb                | 市净率             |
| 8    | roe               | 净资产收益率       |
| 9    | debt_ratio        | 资产负债率         |
| 10   | eps_growth        | EPS 同比增长率     |

数据源：Wind / Tushare / JoinQuant 等接口。

---

## 4. 图的动态构建与更新
### 4.1 行业关系边（长期稳定）
- 根据申万一级行业分类建立双向边，除非股票调出行业，否则保持不变。

### 4.2 相关性关系边（动态稀疏化）
- 每日或每周：计算过去 20 个交易日收益率向量的皮尔逊相关系数。
- 稀疏化策略（防止全连接导致 OOM）：
  - **Top-K 方案**：每只股票保留相关系数绝对值最大的 K 个邻居（K=10~30）。
  - 或设置阈值（如 |corr| > 0.6），保留边并可选附上 `edge_weight`。
- 更新时直接替换 `edge_index` 和 `edge_weight`，无需重建全图。

---

## 5. 模型架构 —— 双层 RGCN + 分类器
```python
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
import torch

class StockRGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2, num_relations=2):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, num_relations)
        self.conv2 = RGCNConv(hidden, hidden, num_relations)
        self.classifier = torch.nn.Linear(hidden, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        x = self.conv1(x_dict['stock'], edge_index_dict, edge_type_dict)
        x = F.relu(x)
        x = self.conv2(x, edge_index_dict, edge_type_dict)
        return self.classifier(x)
```
- `edge_type_dict` 将关系名字符串映射为整数 ID（例：`same_industry→0`，`high_corr→1`）。
- 第一版任务：预测下一交易日收益率方向（涨/跌）的二分类。

---

## 6. 今晚实施步骤（约 4 小时）

### 6.1 数据准备（约 1.5 小时）
1. 获取 5000 只股票最新日的 10 维特征矩阵 `[5000, 10]`。
2. 获取近 20 个交易日的日收益率矩阵 `[5000, 20]`，计算相关系数矩阵。
3. 生成 Top-K 相关性边 `edge_index`（形状 `[2, E]`）。
4. 获取申万一级行业分类，构建同行业边 `edge_index`。

### 6.2 构建异构图（约 0.5 小时）
```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['stock'].x = node_feats                    # [5000, 10]
data['stock', 'same_industry', 'stock'].edge_index = ind_edge_index   # [2, E1]
data['stock', 'high_corr', 'stock'].edge_index = corr_edge_index      # [2, E2]
# 可选：data['stock', 'high_corr', 'stock'].edge_weight = corr_weights
```

### 6.3 标签与数据集划分（0.5 小时）
- 计算下一交易日收益率，标记涨（1）跌（0）。
- 按时间顺序划分训练/验证/测试集（严禁使用未来信息）。
- 生成训练所需的 `mask`。

### 6.4 模型训练与评估（约 1 小时）
- 实例化 `StockRGCN`，`num_relations=2`。
- 使用交叉熵损失，Adam 优化器。
- 训练 50~100 epoch，监控验证集 AUC 或准确率。
- 保存模型权重与图结构。

### 6.5 结果与交付
- 输出第一版基线模型的验证集/测试集性能。
- 保存 `model.pt` 和 `data.pt`，供后续扩展使用。

---

## 7. 后续 LLM 文本表征的扩展方案
### 7.1 增加节点特征
- 使用 LLM（如 FinBERT、ChatGPT 提取新闻/研报摘要）生成每只股票的文本嵌入向量 `[D]`。
- 拼接到原始 10 维特征后，形成新特征矩阵，重新训练或微调 RGCN。

### 7.2 增加新边类型
- 根据文本相似度（如业务描述相似度、新闻共现）构建 `text_sim` 边。
- 作为第三种边类型加入 `HeteroData`，RGCN 的 `num_relations` 改为 3，模型自动适配。

### 7.3 动态更新流程不变
- 文本嵌入和文本相似度边同样可每日更新，与量价图动态更新逻辑完全一致。

---

## 8. 注意事项
- 相关性图务必稀疏化，避免显存溢出。
- 训练/验证/测试必须按时间切分，避免使用未来信息。
- PyG 的 `RGCNConv` 按关系独立参数，关系数量增加时计算量线性增长，初期 2~5 种关系完全可行。
```