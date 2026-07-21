"""v3.2 共识集成：3 编码器 x 3 KMeans 分区 = 9 模型的股票级共识打分。
股票分 = 各模型中该股所在簇的预测收益（小簇置 NaN），nanmean 后选 top-100 可交易。
可交易标签、h=5 持有继承 v3 语义。
"""
import json
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path

OUT = config.RUNS_DIR / "v32"
K, TRAIN_W, HOLD, MIN_SIZE, TOP = 20, 60, 5, 10, 100


class Consensus:
    def __init__(self, market, caches, km_seeds):
        self.m = market
        self.caches = caches
        self.km_seeds = km_seeds
        self._part = {}   # (cache_id, km_seed, t) -> (stocks, labels, feats, ids)
        self._held, self._cd = [], 0
        self._last_scores = None

    def _clusters(self, ci, ks, t):
        key = (ci, ks, t)
        if key not in self._part:
            stocks, emb = self.caches[ci].get(t)
            labels = KMeans(n_clusters=K, random_state=ks, n_init=1).fit_predict(emb)
            ids = sorted(set(labels))
            feats = np.stack([emb[labels == c].mean(axis=0) for c in ids])
            if len(self._part) > 9 * (TRAIN_W + 10):
                oldest = min(self._part, key=lambda k: k[2])
                if oldest != key:
                    self._part.pop(oldest)
            self._part[key] = (stocks, labels, feats, ids)
        return self._part[key]

    def _stock_scores(self, ci, ks, t):
        xs, ys = [], []
        for s in range(t - TRAIN_W, t):
            st, lb, ft, ids = self._clusters(ci, ks, s)
            realized = self.m.realized_returns(s).reindex(st).to_numpy(dtype=float)
            tr = self.m.tradable_at_decision(s).reindex(st).fillna(False).to_numpy(dtype=bool)
            for pos_i, c in enumerate(ids):
                member = (lb == c) & tr
                rets = realized[member]; rets = rets[np.isfinite(rets)]
                if len(rets) >= 5:
                    xs.append(ft[pos_i]); ys.append(rets.mean())
        if len(xs) < 100:
            return None
        model = Ridge(alpha=1.0).fit(np.stack(xs), np.array(ys))
        st, lb, ft, ids = self._clusters(ci, ks, t)
        cluster_score = model.predict(ft)
        tr = self.m.tradable_at_decision(t).reindex(st).fillna(False).to_numpy(dtype=bool)
        out = pd.Series(np.nan, index=st)
        for pos_i, c in enumerate(ids):
            member = lb == c
            if (member & tr).sum() >= MIN_SIZE:
                out[np.asarray(st)[member]] = cluster_score[pos_i]
        return out

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
        self._held = list(consensus.sort_values(ascending=False).index[:TOP])
        self._cd = HOLD - 1
        return self._held


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v32")
    caches = [EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev) for es in (42, 7, 123)]
    name = "v32_consensus9_h5"
    out = OUT / name
    if not (out / "run_summary.json").exists():
        strat = Consensus(market, caches, km_seeds=(42, 7, 123))
        print(f"[run] {name}", flush=True)
        run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out, {"variant": name})
    p = json.loads((out / "run_summary.json").read_text())
    d = pd.read_csv(out / "daily.csv")
    ge = (d["gross"] - d["bench"]).to_numpy()
    print(f"consensus9: gross {sharpe_annualized(ge):+.2f} (NW t={newey_west_tstat(ge):+.2f}) "
          f"net {p['metrics']['sharpe_excess_net']:+.2f} to={p['metrics']['mean_daily_turnover']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
