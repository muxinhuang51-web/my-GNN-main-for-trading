"""v3 原型对打：可交易标签 vs 全体标签 x 持有期 x 种子，全市场验证期。

消融矩阵（k=20, ke=0 固定）：
  label_mode in {tradable, all} x hold_days in {1, 5, 10} x 3 编码器种子 x 2 KMeans 种子
用法：python -m exp.run_v3_proto
"""

import json
from pathlib import Path

import pandas as pd
import torch

from engine import config
from engine.backtest import run_backtest
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.executable_strategy import ExecutableClusterRotation

OUT = config.RUNS_DIR / "v3_proto"


def main() -> int:
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v3_proto")
    for es in (42, 7, 123):
        cache = EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev)
        for label_mode in ("tradable", "all"):
            for hold in (1, 5, 10):
                for seed in (42, 7):
                    name = f"v3_{label_mode}_h{hold}_es{es}_s{seed}"
                    out = OUT / name
                    if (out / "run_summary.json").exists():
                        print(f"[skip] {name}")
                        continue
                    strat = ExecutableClusterRotation(
                        market, cache, cluster_count=20, seed=seed,
                        label_mode=label_mode, hold_days=hold, device=dev,
                    )
                    print(f"[run] {name}")
                    m = run_backtest(market, strat.select, config.VALID_START, config.VALID_END,
                                     out, {"label": label_mode, "hold": hold, "enc_seed": es, "seed": seed},
                                     ic_fn=strat.cluster_ic)
                    print(f"[done] {name}: gross_ex 见 collect  net={m['sharpe_excess_net']:.2f} to={m['mean_daily_turnover']:.3f}")

    # 汇总
    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text())
        d = pd.read_csv(f.parent / "daily.csv")
        import numpy as np
        from engine.backtest import newey_west_tstat, sharpe_annualized
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({**p["params"], "gross_shp": sharpe_annualized(ge),
                     "net_shp": p["metrics"]["sharpe_excess_net"],
                     "nw_t_gross": newey_west_tstat(ge),
                     "to": p["metrics"]["mean_daily_turnover"], "ic": p["metrics"].get("mean_ic")})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "summary.csv", index=False)
    g = frame.groupby(["label", "hold"]).agg(
        gross_shp=("gross_shp", "mean"), gross_std=("gross_shp", "std"),
        net_shp=("net_shp", "mean"), to=("to", "mean"), ic=("ic", "mean"), n=("seed", "count"))
    print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
