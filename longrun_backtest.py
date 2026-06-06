"""
跨季节回测与消融实验脚本。

与 backtest_cluster.py 共用同一核心引擎，不做冗余包装。
本脚本不绕过数据限制——数据长度决定回测天数，不靠降参数来"延长"。

用法：
    # 全量回测（当前数据约 51 天，lookback=60）
    python longrun_backtest.py

    # 按 lookback 对比消融
    python longrun_backtest.py --lookback-sweep 20,40,60

    # 滚动窗交叉验证
    python longrun_backtest.py --rolling 2   # 每 2 个月一窗
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

# ── 参数组合（仅覆盖有意义的变化维度） ──
PARAM_COMBOS = [
    # baseline
    {"tag": "baseline_c20",       "cluster_count": 20, "seed_value": 42},
    # 聚类数消融
    {"tag": "c15",                "cluster_count": 15},
    {"tag": "c25",                "cluster_count": 25},
    {"tag": "c30",                "cluster_count": 30},
    # 种子稳健性
    {"tag": "seed7",              "cluster_count": 20, "seed_value": 7},
    {"tag": "seed123",            "cluster_count": 20, "seed_value": 123},
    # 相关性边消融
    {"tag": "corr5",              "cluster_count": 20, "top_neighbor_count": 5},
    {"tag": "corr10",             "cluster_count": 20, "top_neighbor_count": 10},
]


def parse_args():
    p = argparse.ArgumentParser(description="Long-run backtest + ablation.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-path", default="best_model.pt")
    p.add_argument("--out-root", default="outputs/longrun")
    p.add_argument("--lookback", type=int, default=60, help="需要与 best_model.pt 训练时一致")
    p.add_argument("--lookback-sweep", type=str, default="",
                   help="逗号分隔的 lookback 值，用于消融实验，如 20,40,60")
    p.add_argument("--train-window", type=int, default=20)
    p.add_argument("--rolling", type=int, default=0, help=">0 时按 N 个月滚动窗验证")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backtest_params(args: argparse.Namespace, overrides: Dict[str, Any],
                          out_dir: str, device: torch.device, lookback: int) -> Dict[str, Any]:
    """构造 run_backtest 参数字典，不引入中间层。"""
    return {
        "data_dir": args.data_dir,
        "model_path": args.model_path,
        "lookback": lookback,
        "train_window": args.train_window,
        "top_neighbor_count": overrides.get("top_neighbor_count", 0),
        "cluster_count": overrides.get("cluster_count", 20),
        "seed_value": overrides.get("seed_value", 42),
        "out_dir": out_dir,
        "min_cluster_valid_count": 5,
        "min_portfolio_valid_stocks": 50,
        "target_portfolio_valid_stocks": 100,
        "min_market_valid_stocks": 1000,
        "predictor_epochs": 3,
        "kmeans_n_init": 1,
        "device": device,
        # 不传 start_date/end_date → 用全量可用区间
    }


# ── 滚动窗 ──────────────────────────────────────────────────────
def rolling_date_windows(returns: pd.DataFrame, months: int, lookback: int,
                         train_window: int) -> List[Dict[str, Any]]:
    """返回每个滚动窗的 (start_date, end_date, label)，保证每窗至少有 min_days 天。"""
    all_positions = b.date_positions(returns, lookback, train_window, 1000)
    if not all_positions:
        return []
    dates = returns.index[all_positions]
    offset = pd.DateOffset(months=months)
    cur = dates[0]
    windows = []
    while cur <= dates[-1]:
        end = min(cur + offset - pd.DateOffset(days=1), dates[-1])
        label = f"{cur.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        windows.append({"start": cur.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"), "label": label})
        cur = end + pd.DateOffset(days=1)
        if cur > dates[-1]:
            break
    return windows


# ── main ────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── lookback sweep 模式 ──────────────────────────────────
    if args.lookback_sweep:
        lookbacks = [int(x.strip()) for x in args.lookback_sweep.split(",") if x.strip()]
        records = []
        for lb in lookbacks:
            for combo in PARAM_COMBOS[:3]:  # sweep 只跑 3 组 baseline
                tag = f"lb{lb}_{combo['tag']}"
                out = str(out_root / tag)
                print(f"[sweep] lookback={lb} {tag}")
                t0 = time.perf_counter()
                try:
                    params = build_backtest_params(args, combo, out, device, lb)
                    _, m = b.run_backtest(**params)
                    m["tag"] = tag
                    m["lookback"] = lb
                    m["elapsed"] = round(time.perf_counter() - t0, 1)
                    records.append(m)
                except Exception as e:
                    records.append({"tag": tag, "lookback": lb, "error": str(e)[:200]})
        pd.DataFrame(records).to_csv(out_root / "lookback_sweep.csv", index=False)
        print(f"[sweep] 完成 → {out_root / 'lookback_sweep.csv'}")
        return 0

    # ── 滚动窗模式 ──────────────────────────────────────────
    if args.rolling > 0:
        full_returns = b.load_returns_csv(os.path.join(args.data_dir, "daily_returns.csv"))
        windows = rolling_date_windows(full_returns, args.rolling, args.lookback, args.train_window)
        print(f"[rolling] {len(windows)} windows of {args.rolling} month(s) each")
        records = []
        for combo in PARAM_COMBOS[:3]:
            row = {"combo": combo["tag"]}
            for w in windows:
                tag = f"{combo['tag']}_{w['label']}"
                out = str(out_root / tag)
                try:
                    params = build_backtest_params(args, combo, out, device, args.lookback)
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

    # ── 全量回测模式（默认）─────────────────────────────────
    records = []
    for combo in PARAM_COMBOS:
        tag = combo["tag"]
        out = str(out_root / tag)
        print(f"\n[全量] {tag}  lookback={args.lookback}")
        t0 = time.perf_counter()
        params = build_backtest_params(args, combo, out, device, args.lookback)
        _, m = b.run_backtest(**params)

        m["tag"] = tag
        m["elapsed_sec"] = round(time.perf_counter() - t0, 1)
        records.append(m)
        print(f"  Sharpe={m.get('sharpe'):.3f}  AnnRet={m.get('annualized_return')}  "
              f"MaxDD={m.get('max_drawdown')}  IC={m.get('mean_cluster_ic'):.4f}  "
              f"days={m.get('days')}")

    df = pd.DataFrame(records)
    # 数值列按 Sharpe 排序
    sort_col = "sharpe" if "sharpe" in df.columns else "tag"
    df = df.sort_values(sort_col, ascending=False)
    df.to_csv(out_root / "summary.csv", index=False)
    print(f"\n[全量] 完成 → {out_root / 'summary.csv'}")
    print(df[["tag", "sharpe", "annualized_return", "max_drawdown", "mean_cluster_ic", "days"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
