"""W1+W2：收割机制与信息上限的双重检验（全市场验证期，h=5 持有）。

W1 中段收割：九模型共识打分 → 持有指定分位带（回避两端伪信号），等权
W2 信息上限：已验证最强可交易信号的朴素组合（低波 top100 / rev5 top100）——
   若朴素信号净收益也≈0，说明价格类信息的可执行上限就在零附近
"""
import json
import numpy as np, pandas as pd, torch
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from exp.run_v32_consensus import Consensus

OUT = config.RUNS_DIR / "v33"
HOLD = 5


class BandHarvest(Consensus):
    """持有共识打分的 [lo, hi] 分位带（而非 top-k）。"""

    def __init__(self, market, caches, km_seeds, lo, hi):
        super().__init__(market, caches, km_seeds)
        self.lo, self.hi = lo, hi

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        parts = []
        for ci in range(len(self.caches)):
            for ks in self.km_seeds:
                sc = self._stock_scores(ci, ks, t)
                if sc is not None:
                    parts.append(sc)
        if not parts:
            return self._held
        consensus = pd.concat(parts, axis=1).mean(axis=1)
        tr = self.m.tradable_at_decision(t)
        consensus = consensus[[s for s in consensus.index if bool(tr.get(s, False))]].dropna()
        ranks = consensus.rank(pct=True)
        band = ranks[(ranks >= self.lo) & (ranks <= self.hi)]
        self._held = list(band.index)
        self._cd = HOLD - 1
        return self._held


class SignalTopK:
    """朴素单信号组合：按信号排序取 top_k（h 日持有）。signal: lowvol | rev5"""

    def __init__(self, market, signal, top_k=100):
        self.m = market
        self.signal = signal
        self.top_k = top_k
        self._held, self._cd = [], 0

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        stocks = self.m.universe(t)
        w = self.m.window_returns(t, config.LOOKBACK)[stocks]
        if self.signal == "lowvol":
            score = -w.tail(20).std(ddof=0)   # 波动越低分越高
        else:  # rev5
            score = -w.tail(5).mean()          # 近 5 日跌得越多分越高
        tr = self.m.tradable_at_decision(t)
        score = score[[s for s in score.index if bool(tr.get(s, False))]].dropna()
        self._held = list(score.sort_values(ascending=False).index[: self.top_k])
        self._cd = HOLD - 1
        return self._held


def run_one(market, strat, name, params):
    out = OUT / name
    if (out / "run_summary.json").exists():
        print(f"[skip] {name}")
        return
    print(f"[run] {name}", flush=True)
    run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out, params)


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v33")
    caches = [EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev) for es in (42, 7, 123)]
    for lo, hi in ((0.50, 0.85), (0.60, 0.90), (0.30, 0.70)):
        run_one(market, BandHarvest(market, caches, (42, 7, 123), lo, hi),
                f"band_{int(lo*100)}_{int(hi*100)}", {"variant": "band", "lo": lo, "hi": hi})
    for sig in ("lowvol", "rev5"):
        run_one(market, SignalTopK(market, sig), f"naive_{sig}_top100", {"variant": sig})

    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, "gross_shp": sharpe_annualized(ge),
                     "nw_t": newey_west_tstat(ge), "net_shp": p["metrics"]["sharpe_excess_net"],
                     "to": p["metrics"]["mean_daily_turnover"], "n_stk": int(d["n_stocks"].mean())})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
