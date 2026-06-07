"""
跨季节回测与消融实验脚本。

设计原则：
  - 与 backtest_cluster.py 共用同一核心引擎（b.run_backtest），本脚本只负责编排实验。
  - 不绕过数据限制：数据长度决定回测天数，不通过降 lookback 参数来虚增回测日数。
  - 三种模式互斥，由命令行参数控制。

模式一：全量回测（默认）
    python longrun_backtest.py
    → 使用 lookback=60（与 best_model.pt 训练时一致），跑全量可用区间内所有交易日。
    → 输出 outputs/longrun/summary.csv（8 组参数组合的指标对比）。

模式二：lookback 消融 sweep
    python longrun_backtest.py --lookback-sweep 20,40,60
    → 对比不同 lookback 下的表现差异。注意：lookback != 60 时的输入特征分布
      与 best_model.pt 的训练分布不同，sweep 结果仅作为分布偏移分析，不替代
      lookback=60 的主结果。

模式三：滚动窗交叉验证
    python longrun_backtest.py --rolling 2
    → 按 2 个月切片，每窗独立跑回测。用于检查策略在不同时段的表现一致性。
    → 每窗的 lookback 需求由 date_positions 内部保证，窗内有效日过少则返回 NaN。
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import backtest_cluster as b


# ═══════════════════════════════════════════════════════════════
# 参数网格
# ═══════════════════════════════════════════════════════════════
#
# 每组 combo 只需列出与 baseline 不同的参数；未列出的字段使用
# build_backtest_params 中的默认值。
#
# 覆盖的消融维度：
#   1. 聚类数：15 / 20（baseline）/ 25 / 30
#   2. 随机种子：7 / 42（baseline）/ 123
#   3. 相关性边数量：0（baseline）/ 5 / 10
#
PARAM_COMBOS: List[Dict[str, Any]] = [
    # ── baseline ──
    {"tag": "baseline_c20",       "cluster_count": 20, "seed_value": 42},

    # ── 聚类数消融：检验簇粒度对策略的影响 ──
    {"tag": "c15",                "cluster_count": 15},
    {"tag": "c25",                "cluster_count": 25},
    {"tag": "c30",                "cluster_count": 30},

    # ── 种子稳健性：KMeans 初始中心 + torch 初始化随机性 ──
    {"tag": "seed7",              "cluster_count": 20, "seed_value": 7},
    {"tag": "seed123",            "cluster_count": 20, "seed_value": 123},

    # ── 相关性边消融：最核心的图结构贡献分析 ──
    # top_neighbor_count=0 表示不构造相关性边；>0 则每股票连 K 个最高 |corr| 邻居
    {"tag": "corr5",              "cluster_count": 20, "top_neighbor_count": 5},
    {"tag": "corr10",             "cluster_count": 20, "top_neighbor_count": 10},
]


# ═══════════════════════════════════════════════════════════════
# 命令行参数解析
# ═══════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """
    返回经过 argparse 解析的 Namespace。
    所有参数都有明确默认值，未传参数时可被其他函数安全引用。
    """
    p = argparse.ArgumentParser(description="Long-run backtest + ablation.")
    p.add_argument("--data-dir", default="data",
                   help="数据目录，需包含 daily_returns.csv, industry_mapping.csv")
    p.add_argument("--model-path", default="best_model.pt",
                   help="预训练 RGCN 权重文件路径")
    p.add_argument("--out-root", default="outputs/longrun",
                   help="实验输出根目录，每组参数会自动创建子目录")
    p.add_argument("--lookback", type=int, default=60,
                   help="GNN 输入的历史窗口长度。需与 best_model.pt 训练时一致，"
                        "否则输入特征分布偏移会导致不可比的结果")
    p.add_argument("--lookback-sweep", type=str, default="",
                   help="逗号分隔的 lookback 值，用于消融实验，例如 20,40,60。"
                        "传入后启用 sweep 模式，忽略其他模式。")
    p.add_argument("--train-window", type=int, default=20,
                   help="簇收益预测器的训练窗口长度（天）")
    p.add_argument("--rolling", type=int, default=0,
                   help="大于 0 时启用滚动窗模式，数值为每窗的月份数。"
                        "例如 --rolling 2 表示每 2 个月独立回测。")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="计算设备：auto 自动检测 CUDA，cpu/cuda 强制指定")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def resolve_device(choice: str) -> torch.device:
    """
    将命令行设备选项转为 torch.device 对象。
    auto 模式下优先使用 CUDA，不可用时回退到 CPU。
    """
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    # auto：自动检测
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backtest_params(
    args: argparse.Namespace,
    overrides: Dict[str, Any],
    out_dir: str,
    device: torch.device,
    lookback: int,
) -> Dict[str, Any]:
    """
    构造 b.run_backtest 的参数字典。

    设计要点：
      - 不引入中间调用层——直接复用 args + overrides 拼出完整参数。
      - 不传 start_date 和 end_date，让 run_backtest 内部的 date_positions
        自动使用全量可用日期区间。
      - 固定参数（如 min_market_valid_stocks=1000）硬编码在此处，
        避免通过命令行暴露无意义配置维度。
    """
    return {
        # 路径
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "out_dir": out_dir,

        # 窗口参数
        "lookback": lookback,                     # GNN 特征构造窗口
        "train_window": args.train_window,         # 簇预测器训练窗口

        # 图结构参数
        "top_neighbor_count": overrides.get("top_neighbor_count", 0),

        # 聚类参数
        "cluster_count": overrides.get("cluster_count", 20),
        "seed_value": overrides.get("seed_value", 42),

        # 组合与过滤
        "min_cluster_valid_count": 5,              # 簇内最少有效股票数
        "min_portfolio_valid_stocks": 50,           # 组合日最少有效股票数，不足则记 NaN
        "target_portfolio_valid_stocks": 100,       # 目标持仓有效股票数
        "min_market_valid_stocks": 1000,            # 交易日最少全市场有效股票数

        # 训练参数
        "predictor_epochs": 3,
        "kmeans_n_init": 1,                        # KMeans 重复初始化次数，1=fast
        "device": device,

        # 日期范围：不传 = 让 run_backtest 内部用 date_positions 自动确定
    }


# ═══════════════════════════════════════════════════════════════
# 滚动窗模式
# ═══════════════════════════════════════════════════════════════
#
# 用途：检验策略在不同时段的表现是否一致。
# 原理：把可回测日期按时间切片，每窗独立训练簇预测器 + 聚类 + 选股，
#       避免因整个回测区间恰好处于某段牛市而产生有偏结论。
#
def rolling_date_windows(
    returns: pd.DataFrame,
    months: int,
    lookback: int,
    train_window: int,
) -> List[Dict[str, Any]]:
    """
    按月滑动生成 (start_date, end_date, label) 三元组。

    窗边界基于 date_positions 输出的有效回测日（不是原始 CSV 的全部日期），
    保证每个窗至少覆盖了满足 min_market_valid_stocks=1000 的交易日。

    参数：
        returns:  日收益率 DataFrame
        months:   每窗覆盖的日历月数
        lookback, train_window: 传给 date_positions 计算有效日期
    返回：
        [{start: "2026-02-01", end: "2026-03-31", label: "20260201_20260331"}, ...]

    边界处理：
      - 最后一窗的 end 可能不满 months 个月（数据末尾），取 dates[-1] 即可。
      - start 一定 >= date_positions 的最早有效日，避免前序 lookback 不足。
    """
    # 获取所有满足条件的回测日索引及对应日期
    all_positions = b.date_positions(
        returns, lookback, train_window,
        min_market_valid_stocks=1000,
    )
    if not all_positions:
        return []

    dates = returns.index[all_positions]          # 有效回测日的日期序列
    offset = pd.DateOffset(months=months)          # 窗宽

    cur = dates[0]                                 # 窗起始日为第一个有效日
    windows = []
    while cur <= dates[-1]:
        end = min(cur + offset - pd.DateOffset(days=1), dates[-1])
        label = f"{cur.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        windows.append({
            "start": cur.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "label": label,
        })
        cur = end + pd.DateOffset(days=1)          # 下一窗从 end+1 开始
        # 边界保护：避免 cur 是月末导致无限循环
        if cur > dates[-1]:
            break

    return windows


# ═══════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════
def main() -> int:
    """
    根据命令行参数选择运行模式。

    返回值约定：
        0 = 成功
        1 = 无可回测日期（需放宽参数或检查数据）

    三种模式的调度逻辑（互斥，按优先级）：
        1. --lookback-sweep 非空 → sweep 模式
        2. --rolling > 0        → 滚动窗模式
        3. 都不设               → 全量回测模式（默认）
    """
    args = parse_args()
    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 模式一：lookback sweep
    # ──────────────────────────────────────────────────────────
    # 对多个 lookback 值分别跑回测，分析输入窗口长度对策略的影响。
    # 注意：lookback != 60 时与 best_model.pt 的训练分布不匹配，
    #       此模式用于消融分析而非主结果。
    if args.lookback_sweep:
        lookbacks = [int(x.strip()) for x in args.lookback_sweep.split(",") if x.strip()]
        records = []
        for lb in lookbacks:
            # sweep 模式只跑前 3 组 baseline（时间有限，不做全量网格扫描）
            for combo in PARAM_COMBOS[:3]:
                tag = f"lb{lb}_{combo['tag']}"
                out = str(out_root / tag)
                print(f"[sweep] lookback={lb} {tag}")

                t0 = time.perf_counter()
                try:
                    params = build_backtest_params(args, combo, out, device, lb)
                    _, m = b.run_backtest(**params)
                    # 将 sweep 元信息附到 metrics dict 上
                    m["tag"] = tag
                    m["lookback"] = lb
                    m["elapsed"] = round(time.perf_counter() - t0, 1)
                    records.append(m)
                except Exception as e:
                    # 个别组合因窗口不足等原因失败不应阻断整体 sweep
                    records.append({"tag": tag, "lookback": lb, "error": str(e)[:200]})

        pd.DataFrame(records).to_csv(out_root / "lookback_sweep.csv", index=False)
        print(f"[sweep] 完成 → {out_root / 'lookback_sweep.csv'}")
        return 0

    # ──────────────────────────────────────────────────────────
    # 模式二：滚动窗交叉验证
    # ──────────────────────────────────────────────────────────
    # 按月份切片，每窗独立跑回测。各窗的 Sharpe / MaxDD / IC 填入
    # 同一行（一个 combo 一行，每窗一列），方便横向对比时段稳定性。
    if args.rolling > 0:
        full_returns = b.load_returns_csv(os.path.join(args.data_dir, "daily_returns.csv"))
        windows = rolling_date_windows(
            full_returns, args.rolling, args.lookback, args.train_window,
        )
        print(f"[rolling] {len(windows)} windows of {args.rolling} month(s) each")

        records = []
        for combo in PARAM_COMBOS[:3]:    # 滚动模式只跑 3 组
            row = {"combo": combo["tag"]}
            for w in windows:
                tag = f"{combo['tag']}_{w['label']}"
                out = str(out_root / tag)
                try:
                    params = build_backtest_params(args, combo, out, device, args.lookback)
                    # 关键：传入 start/end 限制该窗的时间范围
                    params["start_date"] = w["start"]
                    params["end_date"] = w["end"]
                    _, m = b.run_backtest(**params)
                    row[f"{w['label']}_sharpe"] = m.get("sharpe")
                    row[f"{w['label']}_days"] = m.get("days")
                except Exception as e:
                    row[f"{w['label']}_err"] = str(e)[:100]
            records.append(row)

        pd.DataFrame(records).to_csv(out_root / "rolling_summary.csv", index=False)
        print(f"[rolling] 完成 → {out_root / 'rolling_summary.csv'}")
        return 0

    # ──────────────────────────────────────────────────────────
    # 模式三：全量回测（默认）
    # ──────────────────────────────────────────────────────────
    # 用全量可回测日期区间跑所有 8 组参数组合。
    # 这是论文的主实验来源——每个 combo 得到一个独立的 out_dir，
    # 内含 metrics.json、daily_returns.csv、图表等完整输出。
    records = []
    for combo in PARAM_COMBOS:
        tag = combo["tag"]
        out = str(out_root / tag)               # 每组参数独立子目录
        print(f"\n[全量] {tag}  lookback={args.lookback}")

        t0 = time.perf_counter()
        params = build_backtest_params(args, combo, out, device, args.lookback)
        # result_df: 逐日回测记录；metrics: 汇总指标字典
        _, m = b.run_backtest(**params)

        m["tag"] = tag
        m["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        records.append(m)

        print(
            f"  Sharpe={m.get('sharpe'):.3f}  "
            f"AnnRet={m.get('annualized_return')}  "
            f"MaxDD={m.get('max_drawdown')}  "
            f"IC={m.get('mean_cluster_ic'):.4f}  "
            f"days={m.get('days')}"
        )

    # 汇总所有组合的结果
    df = pd.DataFrame(records)
    # 按 Sharpe 降序排列，一目了然最优组合
    sort_col = "sharpe" if "sharpe" in df.columns else "tag"
    df = df.sort_values(sort_col, ascending=False)
    df.to_csv(out_root / "summary.csv", index=False)

    print(f"\n[全量] 完成 → {out_root / 'summary.csv'}")
    # 简洁终端输出，方便快速扫读排名
    cols = ["tag", "sharpe", "annualized_return", "max_drawdown", "mean_cluster_ic", "days"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    return 0


# 脚本入口：通过 raise SystemExit 而非 sys.exit，
# 确保 notebook 环境中也能正常导入和使用 main()。
if __name__ == "__main__":
    raise SystemExit(main())
