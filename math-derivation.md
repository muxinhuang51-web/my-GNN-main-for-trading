# 数学推导与公式汇总

> 本文档为论文中涉及的数学公式提供完整推导和定义，对应论文的"数据与方法"和"理论基础"章节。

---

## 1. 符号约定

| 符号 | 含义 |
|------|------|
| $N$ | 股票总数 |
| $T$ | 交易日数 |
| $D$ | 节点特征维度（当前 $D = 6$） |
| $H$ | GNN 隐层维度（当前 $H = 64$） |
| $K$ | 聚类簇数 |
| $R$ | 关系边类型数（当前 $R = 3$） |
| $\mathcal{G}_t$ | 交易日 $t$ 的异构图 |
| $\mathbf{X}_t$ | 交易日 $t$ 的节点特征矩阵，$\mathbf{X}_t \in \mathbb{R}^{N \times D}$ |
| $\mathbf{E}_t$ | 交易日 $t$ 的嵌入矩阵，$\mathbf{E}_t \in \mathbb{R}^{N \times H}$ |
| $\mathbf{r}_t$ | 交易日 $t$ 的日收益率向量，$\mathbf{r}_t \in \mathbb{R}^{N}$ |
| $L$ | 回望窗口长度（默认 $L = 60$） |
| $W$ | 训练窗口长度（默认 $W = 20$） |

---

## 2. 节点特征构造

对交易日 $t$，取历史窗口 $[t-L, t)$ 的收益率矩阵 $\mathbf{R}_{t}^{\text{window}} \in \mathbb{R}^{L \times N}$，构造每只股票 $i$ 的 6 维特征向量：

> 注：本文回测部分使用的是 6 维滚动历史收益特征，而不是早期 `main01.ipynb` 中生成的 10 维静态快照特征。这样做的目的是保证每个回测日的节点特征都只由预测日前的历史收益构造，避免把未来时点的静态财务或行情快照混入历史回测。
>
> 下文中的 $r_\tau^{(i)}$ 表示股票 $i$ 在交易日 $\tau$ 的简单日收益率。所有特征的时间下标都采用“左闭右开”的历史窗口，即预测 $t$ 日收益时，只允许使用 $t-1$ 日及以前的信息。

### 2.1 特征定义

$$\text{mom20}_t^{(i)} = \prod_{\tau=t-20}^{t-1} \left(1 + r_\tau^{(i)}\right) - 1$$

该特征表示过去 20 个交易日的复合收益率，而不是简单平均收益。它更接近投资者在过去约一个月内真实持有该股票得到的累计收益，用于刻画中短期动量强弱。

$$\text{mean5}_t^{(i)} = \frac{1}{5} \sum_{\tau=t-5}^{t-1} r_\tau^{(i)}$$

该特征表示过去 5 个交易日的平均收益，主要反映最近一周左右的短期涨跌状态。相对于 20 日动量，它对最新行情变化更加敏感。

$$\text{mean10}_t^{(i)} = \frac{1}{10} \sum_{\tau=t-10}^{t-1} r_\tau^{(i)}$$

该特征表示过去 10 个交易日的平均收益，位于 5 日短期信号和 20 日动量之间，可用于平滑过短窗口带来的噪声。

$$\text{vol20}_t^{(i)} = \sqrt{\frac{1}{20} \sum_{\tau=t-20}^{t-1} \left(r_\tau^{(i)} - \bar{r}_{20}^{(i)}\right)^2}, \quad \bar{r}_{20}^{(i)} = \frac{1}{20}\sum_{\tau=t-20}^{t-1} r_\tau^{(i)}$$

该特征表示过去 20 个交易日收益率的波动水平。它不是收益方向信号，而是风险和不确定性的度量；在图神经网络中，它可以帮助模型区分“高收益但高波动”和“稳定上涨”的股票。

$$\text{vol60}_t^{(i)} = \sqrt{\frac{1}{60} \sum_{\tau=t-60}^{t-1} \left(r_\tau^{(i)} - \bar{r}_{60}^{(i)}\right)^2}$$

该特征表示过去 60 个交易日的波动水平，窗口更长，因此比 $\text{vol20}$ 更接近股票的中期风险暴露。若一只股票近期波动突然放大，$\text{vol20}$ 与 $\text{vol60}$ 的差异也会反映这种风险状态变化。

$$\text{last\_ret}_t^{(i)} = r_{t-1}^{(i)}$$

该特征表示上一交易日的收益，是所有特征中时间上最新的价格信息。它既可能包含短期动量，也可能包含短期反转，因此在后续模型中由 RGCN 和簇级预测器自动学习其作用方向。

以上 6 个字段与代码中的 `build_features_from_window` 函数一一对应：

| 代码字段 | 论文符号 | 使用窗口 | 主要信息 |
|----------|----------|----------|----------|
| `mom20` | $\text{mom20}_t^{(i)}$ | $[t-20,t)$ | 20 日复合动量 |
| `mean5` | $\text{mean5}_t^{(i)}$ | $[t-5,t)$ | 近一周平均收益 |
| `mean10` | $\text{mean10}_t^{(i)}$ | $[t-10,t)$ | 近两周平均收益 |
| `vol20` | $\text{vol20}_t^{(i)}$ | $[t-20,t)$ | 近一个月波动率 |
| `vol60` | $\text{vol60}_t^{(i)}$ | $[t-60,t)$ | 三个月左右波动率 |
| `last_ret` | $\text{last\_ret}_t^{(i)}$ | $t-1$ | 最新单日收益 |

这 6 个特征对应的含义如下：

| 特征 | 数学定义 | 直观含义 | 作用 |
|------|----------|----------|------|
| $\text{mom20}$ | 过去 20 日复合收益 | 中短期动量 | 捕捉趋势延续或反转前的价格强弱 |
| $\text{mean5}$ | 过去 5 日日均收益 | 短期收益水平 | 捕捉最近一周左右的短期强弱 |
| $\text{mean10}$ | 过去 10 日日均收益 | 中短期收益水平 | 平滑 5 日收益，降低单日噪声 |
| $\text{vol20}$ | 过去 20 日收益标准差 | 中短期波动率 | 衡量近期风险和不确定性 |
| $\text{vol60}$ | 过去 60 日收益标准差 | 较长期波动率 | 衡量更稳定的风险暴露 |
| $\text{last\_ret}$ | 上一交易日收益 | 最新价格冲击 | 捕捉最近一天的信息反应和短期反转风险 |

因此，对股票 $i$，原始节点特征向量可写为：

$$
\mathbf{x}_t^{(i)}
=
\left[
\text{mom20}_t^{(i)},
\text{mean5}_t^{(i)},
\text{mean10}_t^{(i)},
\text{vol20}_t^{(i)},
\text{vol60}_t^{(i)},
\text{last\_ret}_t^{(i)}
\right]
\in \mathbb{R}^{6}
$$

这些特征均由 $t$ 日之前的收益数据计算得到。换言之，在预测 $t$ 日组合收益时，特征中最新可用收益为 $r_{t-1}^{(i)}$，不包含 $r_t^{(i)}$。

### 2.2 截面标准化（Z-score）

对每个特征维度 $d$，在所有 $N$ 只股票上做截面标准化：

$$\tilde{\mathbf{x}}_{t}^{(i,d)} = \frac{\mathbf{x}_{t}^{(i,d)} - \mu_t^{(d)}}{\sigma_t^{(d)}}$$

其中：

$$\mu_t^{(d)} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_{t}^{(i,d)}, \quad \sigma_t^{(d)} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\mathbf{x}_{t}^{(i,d)} - \mu_t^{(d)}\right)^2}$$

标准化后的节点特征矩阵为：

$$
\mathbf{X}_t =
\begin{bmatrix}
\tilde{\mathbf{x}}_t^{(1)} \\
\tilde{\mathbf{x}}_t^{(2)} \\
\vdots \\
\tilde{\mathbf{x}}_t^{(N)}
\end{bmatrix}
\in \mathbb{R}^{N \times 6}
$$

缺失值处理原则：

- 单只股票某个历史窗口内收益不足时，对应特征可能为缺失值。
- 截面标准化时，均值和标准差只由当前横截面中的有效值估计。
- 标准化后仍缺失的值填充为 $0$，可理解为将缺失特征置于截面均值附近。
- 这种处理不会引入预测日收益，因为所有输入仍来自 $[t-L,t)$。

---

## 3. 异构图构建

### 3.1 图定义

交易日 $t$ 的异构图 $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t^{(1)}, \mathcal{E}_t^{(2)}, \mathcal{E}_t^{(3)})$，其中：

- $\mathcal{V} = \{v_1, v_2, \ldots, v_N\}$ 为股票节点集合
- 每条边 $(u, v, r) \in \mathcal{E}_t^{(r)}$ 表示在关系 $r$ 下存在从股票 $u$ 到股票 $v$ 的有向边

三种关系类型：

| $r$ | 边类型 | 符号 |
|-----|--------|------|
| 0 | 行业边 | $\mathcal{E}^{\text{ind}}$ |
| 1 | 概念边 | $\mathcal{E}^{\text{con}}$ |
| 2 | 相关性边 | $\mathcal{E}^{\text{corr}}$ |

### 3.2 行业边构造

给定行业映射函数 $\text{industry}: \mathcal{V} \to \mathbb{N}$，同行业股票之间构造有向边：

$$\mathcal{E}^{\text{ind}} = \left\{ (i, j) \;\middle|\; \text{industry}(v_i) = \text{industry}(v_j),\; i \neq j \right\}$$

对大行业限制每个节点的最大邻居数 $\text{max\_neighbors} = 20$，按股票代码排序取最近邻。

### 3.3 概念边构造

给定概念映射函数 $\text{concept}: \mathcal{V} \to \mathbb{N}$（多对多），同概念股票之间构造有向边：

$$\mathcal{E}^{\text{con}} = \left\{ (i, j) \;\middle|\; \text{concept}(v_i) = \text{concept}(v_j),\; i \neq j \right\}$$

同样限制每节点最大邻居数 $20$。

### 3.4 相关性边构造

基于窗口内收益率向量的皮尔逊相关系数矩阵 $\mathbf{C} \in \mathbb{R}^{N \times N}$：

$$C_{ij} = \frac{\sum_{\tau=t-L}^{t-1} \left(r_\tau^{(i)} - \bar{r}^{(i)}\right)\left(r_\tau^{(j)} - \bar{r}^{(j)}\right)}{\sqrt{\sum_{\tau=t-L}^{t-1} \left(r_\tau^{(i)} - \bar{r}^{(i)}\right)^2} \cdot \sqrt{\sum_{\tau=t-L}^{t-1} \left(r_\tau^{(j)} - \bar{r}^{(j)}\right)^2}}$$

要求两股票均有至少 $\text{min\_overlap} = 20$ 个重叠有效交易日。

对每只股票 $i$，选取 $|C_{ij}|$ 最大的 $\text{top\_neighbor\_count}$ 个邻居 $j$（$j \neq i$），构造有向边：

$$\mathcal{E}^{\text{corr}} = \bigcup_{i=1}^{N} \left\{ (i, j) \;\middle|\; j \in \text{TopK}_{k \neq i}\left(|C_{ik}|,\; \text{top\_neighbor\_count}\right) \right\}$$

### 3.5 边索引表示

所有边合并为统一的 `edge_index` 和 `edge_type` 张量：

$$\text{edge\_index} = \left[ \mathbf{E}_{\text{ind}}^{\top} \;\|\; \mathbf{E}_{\text{con}}^{\top} \;\|\; \mathbf{E}_{\text{corr}}^{\top} \right]^{\top} \in \mathbb{R}^{2 \times |\mathcal{E}|}$$

$$\text{edge\_type} = [\underbrace{0, \ldots, 0}_{|\mathcal{E}^{\text{ind}}|}, \underbrace{1, \ldots, 1}_{|\mathcal{E}^{\text{con}}|}, \underbrace{2, \ldots, 2}_{|\mathcal{E}^{\text{corr}}|}]^{\top}$$

其中 $\|\cdot\|$ 表示列拼接，$|\mathcal{E}| = |\mathcal{E}^{\text{ind}}| + |\mathcal{E}^{\text{con}}| + |\mathcal{E}^{\text{corr}}|$。

---

## 4. RGCN 嵌入模型

### 4.1 关系图卷积层

RGCN 第 $\ell$ 层对节点 $i$ 的更新公式为：

$$\mathbf{h}_i^{(\ell+1)} = \sigma\left( \mathbf{W}_0^{(\ell)} \mathbf{h}_i^{(\ell)} + \sum_{r=1}^{R} \sum_{j \in \mathcal{N}_i^{(r)}} \frac{1}{|\mathcal{N}_i^{(r)}|} \mathbf{W}_r^{(\ell)} \mathbf{h}_j^{(\ell)} \right)$$

其中：

- $\mathbf{h}_i^{(\ell)} \in \mathbb{R}^{d_\ell}$ 为第 $\ell$ 层节点 $i$ 的表示
- $\mathcal{N}_i^{(r)} = \{j \mid (j, i, r) \in \mathcal{E}^{(r)}\}$ 为关系 $r$ 下节点 $i$ 的邻居集合
- $\mathbf{W}_0^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ 为自环变换矩阵
- $\mathbf{W}_r^{(\ell)} \in \mathbb{R}^{d_{\ell+1} \times d_\ell}$ 为关系 $r$ 的变换矩阵
- $\sigma(\cdot)$ 为非线性激活函数（本文使用 $\text{ReLU}(x) = \max(0, x)$）

### 4.2 模型架构

两层 RGCN + 线性回归头：

$$\mathbf{H}^{(1)} = \text{ReLU}\left( \text{RGCNConv}_1(\mathbf{X}_t, \text{edge\_index}, \text{edge\_type}) \right)$$

$$\mathbf{E}_t = \text{RGCNConv}_2(\mathbf{H}^{(1)}, \text{edge\_index}, \text{edge\_type})$$

$$\hat{r}_t^{(i)} = \mathbf{w}_{\text{reg}}^{\top} \mathbf{e}_t^{(i)} + b_{\text{reg}}$$

其中 $\mathbf{e}_t^{(i)} \in \mathbb{R}^{H}$ 为股票 $i$ 在交易日 $t$ 的嵌入向量，$\mathbf{w}_{\text{reg}} \in \mathbb{R}^{H}$、$b_{\text{reg}} \in \mathbb{R}$ 为线性回归参数。

### 4.3 嵌入导出

回测中只使用嵌入层，不经过回归头：

$$\mathbf{E}_t = \text{RGCN}_2\left( \text{ReLU}\left( \text{RGCN}_1(\mathbf{X}_t, \mathcal{G}_t) \right) \right)$$

---

## 5. 动态 KMeans 聚类

### 5.1 目标函数

对每个交易日 $t$，对嵌入矩阵 $\mathbf{E}_t$ 做 KMeans 聚类：

$$\min_{\mathbf{C}, \mathbf{S}} \sum_{k=1}^{K} \sum_{i \in S_k} \|\mathbf{e}_t^{(i)} - \mathbf{c}_k\|_2^2$$

其中：

- $S_k$ 为第 $k$ 个簇的节点集合，$\bigcup_k S_k = \{1, \ldots, N\}$，$S_k \cap S_{k'} = \emptyset\;(k \neq k')$
- $\mathbf{c}_k \in \mathbb{R}^{H}$ 为第 $k$ 个簇的中心向量
- 聚类标签 $\ell_t^{(i)} \in \{0, 1, \ldots, K-1\}$ 满足 $\ell_t^{(i)} = k \iff i \in S_k$

### 5.2 簇内样本分组

将股票索引按簇标签分组：

$$\mathcal{I}_k = \{i \mid \ell_t^{(i)} = k\}, \quad k = 0, 1, \ldots, K-1$$

---

## 6. 簇级特征与样本构造

### 6.1 均值池化

对每个簇 $k$，计算簇级特征向量（对簇内股票嵌入取均值）：

$$\mathbf{f}_t^{(k)} = \frac{1}{|\mathcal{I}_k|} \sum_{i \in \mathcal{I}_k} \mathbf{e}_t^{(i)} \in \mathbb{R}^{H}$$

### 6.2 簇级标签

簇 $k$ 在交易日 $t$ 的真实收益为簇内股票下一日收益的均值：

$$y_t^{(k)} = \frac{1}{|\mathcal{I}_k^{\text{valid}}|} \sum_{i \in \mathcal{I}_k^{\text{valid}}} r_t^{(i)}$$

其中 $\mathcal{I}_k^{\text{valid}} = \{i \in \mathcal{I}_k \mid r_t^{(i)} \text{ 非空且有限}\}$。

过滤条件：$|\mathcal{I}_k^{\text{valid}}| \geq \text{min\_cluster\_valid\_count}$（默认 5）。

### 6.3 训练样本构造

对回测日 $t$，取历史训练窗口 $\mathcal{T}_{\text{train}} = [t-W, t)$，对每个训练日 $\tau \in \mathcal{T}_{\text{train}}$：

1. 导出嵌入 $\mathbf{E}_\tau$
2. 聚类得标签 $\boldsymbol{\ell}_\tau$
3. 构造簇级特征 $\mathbf{F}_\tau = [\mathbf{f}_\tau^{(0)}, \ldots, \mathbf{f}_\tau^{(K-1)}]^{\top}$
4. 计算簇级标签 $\mathbf{y}_\tau = [y_\tau^{(0)}, \ldots, y_\tau^{(K-1)}]^{\top}$

堆叠所有训练日样本：

$$\mathbf{F}_{\text{train}} = \begin{bmatrix} \mathbf{F}_{t-W} \\ \vdots \\ \mathbf{F}_{t-1} \end{bmatrix} \in \mathbb{R}^{M \times H}, \quad \mathbf{y}_{\text{train}} = \begin{bmatrix} \mathbf{y}_{t-W} \\ \vdots \\ \mathbf{y}_{t-1} \end{bmatrix} \in \mathbb{R}^{M}$$

其中 $M$ 为有效簇级样本总数。

---

## 7. 簇收益预测器（MLP）

### 7.1 模型结构

两层 MLP + Dropout：

$$\mathbf{z}^{(k)} = \text{ReLU}\left( \mathbf{W}_1 \mathbf{f}^{(k)} + \mathbf{b}_1 \right)$$

$$\tilde{\mathbf{z}}^{(k)} = \text{Dropout}(\mathbf{z}^{(k)}, p)$$

$$\hat{y}^{(k)} = \mathbf{w}_2^{\top} \tilde{\mathbf{z}}^{(k)} + b_2$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{H_{\text{hid}} \times H}$，$\mathbf{W}_2 \in \mathbb{R}^{1 \times H_{\text{hid}}}$，$p = 0.1$，$H_{\text{hid}} = 32$。

### 7.2 训练目标

均方误差损失：

$$\mathcal{L} = \frac{1}{M} \sum_{m=1}^{M} \left( \hat{y}_m - y_m \right)^2$$

### 7.3 优化器

Adam 优化器：

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中 $\eta = 10^{-3}$ 为学习率，$\beta_1 = 0.9$，$\beta_2 = 0.999$，权重衰减 $\lambda = 10^{-4}$。

---

## 8. 组合构建

### 8.1 簇排序

按预测收益降序排列簇：

$$\text{rank}(k) = \text{argsort}(-\hat{y}^{(k)})$$

### 8.2 逐簇加入选股

从预测收益最高的簇开始，依次加入簇内股票，直到组合中预测日前已知的可交易股票数达到目标。

为避免未来信息泄漏，严格的论文版回测应使用预测日前已知的可交易集合 $\mathcal{A}_{t-1}$，而不是使用 $t$ 日真实收益是否缺失来决定选股。可交易集合可以由上一交易日是否有有效收益、停牌信息或其他预测日前可得的交易状态构造。

$$\mathcal{S}_t = \bigcup_{j=1}^{J^*} \mathcal{I}_{\text{rank}^{-1}(j)}$$

其中 $J^*$ 为满足以下条件的最小索引：

$$J^* = \min \left\{ J \;\middle|\; \sum_{j=1}^{J} \left|\mathcal{I}_{\text{rank}^{-1}(j)} \cap \mathcal{A}_{t-1}\right| \geq \text{target\_portfolio\_valid\_stocks} \right\}$$

默认 $\text{target\_portfolio\_valid\_stocks} = 100$。

### 8.3 组合日收益率

等权持有入选股票：

$$R_t^{\text{port}} = \frac{1}{|\mathcal{S}_t^{\text{valid}}|} \sum_{i \in \mathcal{S}_t^{\text{valid}}} r_t^{(i)}$$

其中 $\mathcal{S}_t^{\text{valid}} = \{i \in \mathcal{S}_t \mid r_t^{(i)} \text{ 有效}\}$。若 $|\mathcal{S}_t^{\text{valid}}| < \text{min\_portfolio\_valid\_stocks}$（默认 50），则当日收益记为 $\text{NaN}$。

注意：$\mathcal{A}_{t-1}$ 用于预测前的组合构建，$\mathcal{S}_t^{\text{valid}}$ 仅用于事后计算组合真实收益。二者在数学上需要区分，以避免在选股阶段使用预测日数据可用性。

---

## 9. 回测评估指标

### 9.1 累计收益

设回测区间内有效交易日为 $t_1, t_2, \ldots, t_T$，对应组合日收益 $R_1, R_2, \ldots, R_T$：

$$\text{CumRet}_t = \prod_{\tau=1}^{t} (1 + R_\tau)$$

### 9.2 年化收益率

$$\text{AnnRet} = \left( \prod_{t=1}^{T} (1 + R_t) \right)^{\frac{252}{T}} - 1$$

### 9.3 年化夏普比率

$$\text{Sharpe} = \frac{\bar{R}}{\sigma_R} \cdot \sqrt{252}$$

其中：

$$\bar{R} = \frac{1}{T} \sum_{t=1}^{T} R_t, \quad \sigma_R = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} (R_t - \bar{R})^2}$$

### 9.4 最大回撤

$$\text{Wealth}_t = \prod_{\tau=1}^{t} (1 + R_\tau)$$

$$\text{Peak}_t = \max_{1 \leq \tau \leq t} \text{Wealth}_\tau$$

$$\text{DD}_t = \frac{\text{Wealth}_t - \text{Peak}_t}{\text{Peak}_t}$$

$$\text{MaxDD} = \min_{1 \leq t \leq T} \text{DD}_t$$

### 9.5 簇级信息系数（IC）

对每个回测日 $t$，计算预测簇收益与真实簇收益之间的皮尔逊相关系数（截面 IC）：

$$\text{IC}_t = \frac{\sum_{k=1}^{K_t} (\hat{y}_t^{(k)} - \bar{\hat{y}}_t)(y_t^{(k)} - \bar{y}_t)}{\sqrt{\sum_{k=1}^{K_t} (\hat{y}_t^{(k)} - \bar{\hat{y}}_t)^2} \cdot \sqrt{\sum_{k=1}^{K_t} (y_t^{(k)} - \bar{y}_t)^2}}$$

其中 $K_t$ 为交易日 $t$ 的有效簇数，$\bar{\hat{y}}_t$ 和 $\bar{y}_t$ 分别为预测值和真实值的截面均值。

平均 IC：

$$\overline{\text{IC}} = \frac{1}{T_{\text{IC}}} \sum_{t: \text{IC}_t \text{ 有效}} \text{IC}_t$$

仅当有效簇数 $\geq 2$ 时计算 IC。

---

## 10. 超参搜索评分函数

在参数优化中，综合评分函数为：

$$\text{Score} = \text{Sharpe} + 0.25 \cdot \tanh(\text{AnnRet}) + 2.0 \cdot \text{MaxDD} + 0.5 \cdot \overline{\text{IC}}$$

设计意图：

- $\text{Sharpe}$ 为主信号，直接度量风险调整后收益
- $\tanh(\text{AnnRet})$ 对年化收益做饱和变换，避免短样本高收益主导评分
- $\text{MaxDD}$ 为负值（最大回撤越深，负得越厉害），天然作为惩罚项
- $\overline{\text{IC}}$ 度量预测一致性

---

## 11. 研究假设的数学表述

### H1: 关系图嵌入能提取股票间联动信息

$$\text{Cov}(r_t^{(i)}, r_t^{(j)}) \neq 0 \;\Longrightarrow\; \|\mathbf{e}_t^{(i)} - \mathbf{e}_t^{(j)}\|_2 \text{ 较小}$$

即存在联动关系的股票在嵌入空间中距离更近。

### H2: 簇级预测比个股级预测更稳健

$$\text{Var}\left( \frac{1}{|S_k|} \sum_{i \in S_k} r_t^{(i)} \right) \leq \frac{1}{|S_k|^2} \sum_{i \in S_k} \text{Var}(r_t^{(i)})$$

由簇内均值池化的方差压缩性质，簇级收益的方差低于个股收益方差的均值。

### H3: 不同边类型对预测表现有不同贡献

设模型在边类型子集 $\mathcal{R}' \subseteq \{0, 1, 2\}$ 下的表现函数为 $\text{Score}(\mathcal{R}')$，则：

$$|\text{Score}(\{0,1,2\}) - \text{Score}(\{0,1,2\} \setminus \{r\})| > 0$$

即去掉任一关系边会导致表现下降。

### H4: 动态簇选择能形成有效轮动收益

$$\mathbb{E}\left[ R_t^{\text{port}} \right] > 0 \quad \text{且} \quad \text{Sharpe} > \text{Baseline}_{\text{Sharpe}}$$

---

## 12. 补充：相关性边构造中的皮尔逊相关系数细节

给定窗口内收益率矩阵 $\mathbf{R}^{\text{window}} \in \mathbb{R}^{L \times N}$，对股票 $i$ 和 $j$：

（1）有效日索引：$\mathcal{T}_{ij} = \{\tau \in [t-L, t) \mid r_\tau^{(i)} \neq \text{NA} \land r_\tau^{(j)} \neq \text{NA}\}$

（2）需要 $|\mathcal{T}_{ij}| \geq \text{min\_overlap}$（默认 20）才计算相关系数

（3）均值估计：

$$\bar{r}^{(i)} = \frac{1}{|\mathcal{T}_{ij}|} \sum_{\tau \in \mathcal{T}_{ij}} r_\tau^{(i)}$$

（4）相关系数：

$$C_{ij} = \frac{\sum_{\tau \in \mathcal{T}_{ij}} (r_\tau^{(i)} - \bar{r}^{(i)})(r_\tau^{(j)} - \bar{r}^{(j)})}{\sqrt{\sum_{\tau \in \mathcal{T}_{ij}} (r_\tau^{(i)} - \bar{r}^{(i)})^2} \sqrt{\sum_{\tau \in \mathcal{T}_{ij}} (r_\tau^{(j)} - \bar{r}^{(j)})^2}}$$

---

## 13. KMeans 初始化

使用 k-means++ 初始化策略（`n_init = kmeans_n_init`）：

（1）随机选择第一个中心 $\mathbf{c}_1$

（2）对每个节点 $i$，计算到最近中心的距离 $D(i) = \min_{j} \|\mathbf{e}^{(i)} - \mathbf{c}_j\|_2^2$

（3）以概率 $\frac{D(i)}{\sum_i D(i)}$ 选择下一个中心

（4）重复步骤 (2)-(3) 直至选出 $K$ 个中心

（5）执行标准 Lloyd 迭代至收敛

（6）重复以上过程 `kmeans_n_init` 次（默认 1），选择 SSE 最小的结果

$$\text{SSE} = \sum_{k=1}^{K} \sum_{i \in S_k} \|\mathbf{e}^{(i)} - \mathbf{c}_k\|_2^2$$

---

## 14. 重要约束：无未来信息泄漏

本文严格遵守时序因果约束：

（1）节点特征 $\mathbf{X}_t$ 仅使用 $[t-L, t)$ 的数据

（2）嵌入 $\mathbf{E}_t$ 仅使用 $\mathbf{X}_t$ 和当前图结构 $\mathcal{G}_t$ 计算

（3）聚类仅使用 $\mathbf{E}_t$ 做 KMeans

（4）簇收益预测器只用 $[t-W, t)$ 的历史簇级样本训练

（5）$t$ 日真实收益 $r_t^{(i)}$ 仅用于评估和回测收益计算，不参与任何预测输入

用数学语言表述：

$$\hat{y}_t^{(k)} = f_\theta\left( \left\{ (\mathbf{E}_\tau, \ell_\tau, y_\tau) \right\}_{\tau=t-W}^{t-1}, \;\mathbf{E}_t, \ell_t \right)$$

其中 $f_\theta$ 为簇收益预测器，训练不涉及 $t$ 日及之后的信息。

---

## 15. 随机种子与可复现性

所有随机操作统一使用固定种子 $\text{seed}$：

- `torch.manual_seed(seed)`
- `np.random.seed(seed)`
- `KMeans(random_state=seed)`

这保证 KMeans 簇分配、MLP 参数初始化和 mini-batch 采样顺序在相同 seed 下完全可复现。
