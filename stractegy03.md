# 下一步：基于已建异构图进行 RGCN 训练与评估

你已经完成了图的构建（`HeteroData` 含三类边），现在需要立即衔接**模型训练与基线评估**，这是今晚初版的核心产出。

---

## 1. 前置条件确认
- 已有 `data/hetero_edges.pt`，内含 `HeteroData` 对象，包含：
  - `stock.x`: 节点特征 `[N, 10]` （需确保已按股票代码对齐，N=5000）
  - `stock.same_industry.edge_index`
  - `stock.same_concept.edge_index`
  - `stock.high_corr.edge_index` （及可选的 `edge_weight`）
- 已有日频行情数据，可用于计算下一交易日收益率标签。
- 软件环境：PyTorch + PyG 可用。

---

## 2. 任务定义
**节点二分类**：预测每个股票下个交易日的涨跌（1=上涨，0=下跌/平盘）。  
这是最直接且可立刻评估的基线任务。

---

## 3. 实施步骤（约 2 小时）

### 3.1 生成标签并合并到图
- 对齐股票顺序，计算下个交易日收益率 `future_ret`。
- 标签 `y = (future_ret > 0).astype(int)`。
- 将 `y` 存入 `data['stock'].y`。
- 进行 **时间序列划分**，避免未来信息泄漏。例如：
  - 使用最近 T 个交易日生成图（当天图用过去信息），标签是下一日。
  - 训练集：较早的日期；验证集：最近的 N 天。
  - 简单方法：取所有交易日的最后 20% 作为验证，或固定某一天（如最近一个交易日）验证。
  - 在 `HeteroData` 中设置 `train_mask`, `val_mask` 布尔张量。

```python
# 假设 data 已加载，y 已计算
num_nodes = data['stock'].x.size(0)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

# 示例：最近一次截面作为验证，其余为训练（需确保图是同一时间截面）
# 如果你的图是某个特定交易日构建的，则所有节点共享同一日期，需要多个日期的图。
# 更规范做法：生成多日图，按日期切分。但今晚可简化：使用历史多日图构建多个data，或固定一个截面。