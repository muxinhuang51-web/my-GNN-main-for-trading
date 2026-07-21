"""T1 实验运行器：在 VALID 期（2022-2023）做全部探索。TEST 期结果冻结前不触碰。

用法：
  python -m exp.run_t1 --scope csi300 --stage encoders    # 训练 5 个 k_e 编码器
  python -m exp.run_t1 --scope csi300 --stage sweep       # k 扫描 + k_e 扫描 x 5 种子
  python -m exp.run_t1 --scope csi300 --stage baselines   # 全部基线
  python -m exp.run_t1 --scope csi300 --stage collect     # 汇总 summary.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from engine import config
from engine.backtest import run_backtest
from engine.baselines import (
    IndustryRotation,
    MomentumTopK,
    RandomClusterRotation,
    RandomTopK,
    RawFeatureKMeans,
    StockLevelRGCN,
)
from engine.cluster_strategy import ClusterRotation, EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path, train_encoder

KE_GRID = [0, 5, 10, 20, 40]
K_GRID = [5, 10, 15, 20, 30, 50]
SEEDS = [42, 7, 123]           # 下游（KMeans/MLP）种子
ENCODER_SEEDS = [42, 7, 123]   # 编码器训练种子：每个 k_e 3 个，消除单次训练噪声混淆
TRAIN_START = pd.Timestamp("2016-01-01")


def out_root(scope: str) -> Path:
    root = config.RUNS_DIR / f"t1_{scope}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def stage_encoders(scope: str) -> None:
    market = MarketData(scope)
    for k_e in KE_GRID:
        for enc_seed in ENCODER_SEEDS:
            train_encoder(
                market, k_e, enc_seed,
                TRAIN_START, config.TRAIN_END, deploy_start=config.VALID_START, device=device(),
            )


def _run(market, strategy, name, scope, params, ic_fn=None):
    out = out_root(scope) / name
    if (out / "run_summary.json").exists():
        print(f"[skip] {name} 已完成")
        return
    assert_no_test_access(config.VALID_END, f"T1:{name}")
    print(f"[run] {name}")
    metrics = run_backtest(
        market, strategy.select, config.VALID_START, config.VALID_END, out, params, ic_fn=ic_fn
    )
    print(f"[done] {name}: sharpe_excess={metrics['sharpe_excess_net']:.2f} "
          f"to={metrics['mean_daily_turnover']:.2f} ic={metrics.get('mean_ic')}")


def stage_sweep(scope: str) -> None:
    market = MarketData(scope)
    dev = device()
    # 每格 = 3 编码器种子 x 3 下游种子；嵌入缓存按 (k_e, enc_seed) 复用
    caches = {(k_e, es): EmbeddingCache(market, encoder_path(scope, k_e, es, config.TRAIN_END), k_e, dev)
              for k_e in KE_GRID for es in ENCODER_SEEDS}
    for es in ENCODER_SEEDS:
        for seed in SEEDS:
            for k in K_GRID:
                strat = ClusterRotation(market, caches[(0, es)], cluster_count=k, seed=seed, device=dev)
                _run(market, strat, f"cluster_k{k}_ke0_es{es}_s{seed}", scope,
                     {"k": k, "k_e": 0, "enc_seed": es, "seed": seed}, ic_fn=strat.cluster_ic)
            for k_e in KE_GRID[1:]:
                strat = ClusterRotation(market, caches[(k_e, es)], cluster_count=20, seed=seed, device=dev)
                _run(market, strat, f"cluster_k20_ke{k_e}_es{es}_s{seed}", scope,
                     {"k": 20, "k_e": k_e, "enc_seed": es, "seed": seed}, ic_fn=strat.cluster_ic)
        # 个股级 RGCN（每个 k_e x 编码器种子）
        for k_e in KE_GRID:
            strat = StockLevelRGCN(market, caches[(k_e, es)])
            _run(market, strat, f"stock_rgcn_ke{k_e}_es{es}", scope,
                 {"baseline": "stock_rgcn", "k_e": k_e, "enc_seed": es})


def stage_baselines(scope: str) -> None:
    market = MarketData(scope)
    dev = device()
    _run(market, MomentumTopK(market), "momentum_top100", scope, {"baseline": "momentum"})
    _run(market, MomentumTopK(market, ascending=True), "reversal_top100", scope, {"baseline": "reversal"})
    for seed in range(6):
        _run(market, RandomTopK(market, seed=seed), f"random_top100_seed{seed}", scope,
             {"baseline": "random", "seed": seed})
    for es in ENCODER_SEEDS:
        cache0 = EmbeddingCache(market, encoder_path(scope, 0, es, config.TRAIN_END), 0, dev)
        for seed in SEEDS:
            strat = IndustryRotation(market, cache0, cluster_count=31, seed=seed, device=dev)
            _run(market, strat, f"industry_rotation_es{es}_s{seed}", scope,
                 {"baseline": "industry_rotation", "enc_seed": es, "seed": seed}, ic_fn=strat.cluster_ic)
            strat = RandomClusterRotation(market, cache0, cluster_count=20, seed=seed, device=dev)
            _run(market, strat, f"random_cluster_k20_es{es}_s{seed}", scope,
                 {"baseline": "random_cluster", "k": 20, "enc_seed": es, "seed": seed}, ic_fn=strat.cluster_ic)
            strat = RawFeatureKMeans(market, cache0, cluster_count=20, seed=seed, device=dev)
            _run(market, strat, f"rawfeat_kmeans_k20_es{es}_s{seed}", scope,
                 {"baseline": "rawfeat_kmeans", "k": 20, "enc_seed": es, "seed": seed}, ic_fn=strat.cluster_ic)


def stage_collect(scope: str) -> None:
    import numpy as np

    from engine.backtest import newey_west_tstat, sharpe_annualized

    rows = []
    for summary in sorted(out_root(scope).glob("*/run_summary.json")):
        payload = json.loads(summary.read_text())
        row = {"name": summary.parent.name, **payload["params"], **payload["metrics"]}
        daily_path = summary.parent / "daily.csv"
        if daily_path.exists():
            daily = pd.read_csv(daily_path)
            gross_excess = (daily["gross"] - daily["bench"]).to_numpy()
            row["sharpe_excess_gross"] = sharpe_annualized(gross_excess)
            row["mean_gross_excess_bp"] = float(np.nanmean(gross_excess) * 1e4)
            row["nw_tstat_gross"] = newey_west_tstat(gross_excess)
            row["mean_cost_bp"] = float(np.nanmean(daily["gross"] - daily["net"]) * 1e4)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_root(scope) / "summary.csv", index=False)
    cols = [c for c in ("name", "sharpe_excess_gross", "nw_tstat_gross", "sharpe_excess_net",
                        "mean_daily_turnover", "mean_ic", "days") if c in frame.columns]
    print(frame[cols].sort_values("sharpe_excess_gross", ascending=False).to_string(index=False))


def main() -> int:
    global KE_GRID, K_GRID, SEEDS, ENCODER_SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["csi300", "csi500", "all"], required=True)
    parser.add_argument("--stage", choices=["encoders", "sweep", "baselines", "collect"], required=True)
    parser.add_argument("--coarse", action="store_true", help="全市场 T2 粗网格：少 k_e、少种子")
    args = parser.parse_args()
    if args.coarse:
        KE_GRID = [0, 5, 20]
        K_GRID = [10, 20, 30, 50]
        SEEDS = [42, 7]
        ENCODER_SEEDS = [42, 7]
    config.ensure_dirs()
    {"encoders": stage_encoders, "sweep": stage_sweep,
     "baselines": stage_baselines, "collect": stage_collect}[args.stage](args.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
