# 实验设计 v2（修复版协议）

> 本文档是 v2 实验的规范。代码入口见文末。旧版（main/*.ipynb + backtest_cluster.py）
> 因训练/回测重叠泄漏、零成本、幸存者偏差等问题已废弃，仅作对照素材。

## 1. 数据

- 来源：tushare（daily + adj_factor + stock_basic(L/D/P) + 申万一级 index_member + index_weight + suspend_d + stk_limit）
- 区间：2015-01 → 至今；后复权收益；停牌日 NaN（跳空计入复牌日）
- 股票池：PIT（上市满 180 日、含退市股、按日期判定）；**北交所全剔除**
- 三个规模：csi300 / csi500（月度历史成分）/ all（沪深全市场 ~5450）
- 校验：10 项自动校验全绿才可进实验（data_layer/validate.py）

## 2. 时间防火墙

| 分区 | 区间 | 用途 |
|---|---|---|
| train | 2016-01 ~ 2021-12 | 编码器训练（选型用训练期内部末 10%） |
| valid | 2022-01 ~ 2023-12 | 一切探索、调参、扫描、诊断 |
| test  | 2024-01 ~ | 冻结。结果定稿前代码级禁止访问（LeakageError） |

## 3. 方法链路（被评测对象）

6 维价格特征（截面 z-score）→ 异构图（PIT 申万行业边 + 滚动相关 top-k_e 边）
→ 2 层 RGCN（每个 k_e × 编码器种子独立训练，训练图=推理图）→ 64 维嵌入
→ 每日 KMeans(k) → 簇均值池化 → 过去 20 日簇样本训练 MLP → 预测次日簇收益
→ 按预测降序逐簇选股至 ~100 只 → 等权，日频调仓

## 4. 交易语义（回测引擎）

- 决策只用 ≤ t-1 信息；买入过滤：t-1 停牌或涨停触板不可买
- **强制持仓**：t-1 停牌/跌停锁死的持仓不可卖，复牌跳空由持有人承担
- 成本：买 5bp / 卖 15bp（含印花税），按实际换手计；停牌持仓冻结计 0
- 基准：股票池 ∩ t-1 可买 的等权收益（与策略同一套过滤，可投资）
- 指标：毛/净超额 Sharpe（统一公式）、NW t 值、block bootstrap CI、换手、簇级 IC

## 5. 实验矩阵

- 双轴：聚类数 k ∈ {5,10,15,20,30,50}（k_e=0）；相关边 k_e ∈ {0,5,10,20,40}（k=20）
- 每格 = 3 编码器种子 × 3 下游种子 = 9 组独立运行（全市场粗网格 2×2）
- 基线：指数等权（基准本身）、截面动量/反转、随机 top-100、行业轮动、随机分簇、
  无 GNN 原始特征 KMeans、个股级 RGCN——全部同引擎同指标
- 统计：格内 mean±std + 跨运行同日均值序列的 NW t（保守处理种子间同日相关）

## 6. 当前结论（验证期，详见 experiment-summary-v2.md）

1. csi300 / csi500：双轴全部 0±噪声，IC≈0——6 维价格特征在指数池无截面信息
2. 全市场：**簇级 IC 随 k 单调升至 0.117，但选簇组合毛超额 -30bp+/日**——
   "可预测但不可变现"，候选机制：涨停延续（+183bp/日，不可买）主导簇均值，
   可执行残差呈日频反转。机制分解见 exp/diag_unmonetizable.py 与搜捕工作流报告
3. 净口径：60%+ 日换手 × 20bp 成本 ≈ -13bp/日机械拖累，任何日频全换仓策略不可行

## 7. 代码地图

| 模块 | 职责 |
|---|---|
| data_layer/{fetch,panel,validate}.py | 拉取 / 面板 / 校验 |
| engine/config.py | 协议常量（防火墙、成本） |
| engine/data.py | PIT 接口、可交易性、可投资基准、泄漏断言 |
| engine/features.py | 特征与图构造 + 缓存 |
| engine/encoder.py | RGCN 与无泄漏训练 |
| engine/cluster_strategy.py | 嵌入缓存 + 簇轮动策略 |
| engine/baselines.py | 全部基线策略 |
| engine/backtest.py | 回测循环 + 统计工具 |
| exp/run_t1.py | 网格与阶段 |
| exp/pipeline.py | 全链编排 |
| exp/report.py | 汇总报告生成 |
| exp/diag_unmonetizable.py | 不可变现 IC 诊断 |
| tests/test_engine.py | 12 项语义回归测试 |

复现：`python -m exp.pipeline`（幂等，断点续跑）。
