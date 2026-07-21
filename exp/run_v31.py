"""v3.1：两个针对性修复的消融。
A. 编码器集成：簇打分 = 三个编码器种子 ridge 预测的均值（消除单编码器抽签）
B. 因子增强：预测器输入 = [簇均值嵌入] + [簇级 vol20/mom20/rev5 等已验证存活的因子]
用法：python -m exp.run_v31
"""
import json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.executable_strategy import ExecutableClusterRotation

OUT = config.RUNS_DIR / "v31"


class EnsembleExecutable(ExecutableClusterRotation):
    """A+B：多编码器打分集成 + 簇级因子增强（可交易标签、锁死准入、h 日持有继承自 v3）。"""

    def __init__(self, market, caches, use_factors, **kw):
        super().__init__(market, caches[0], **kw)
        self.caches = caches
        self.use_factors = use_factors
        self._factor_cache = {}

    def _cluster_factors(self, t, stocks, labels, ids):
        key = t
        if key in self._factor_cache:
            return self._factor_cache[key]
        w = self.market.window_returns(t, config.LOOKBACK)[stocks]
        vol20 = w.tail(20).std(ddof=0).to_numpy(dtype=np.float32)
        mom20 = ((1 + w.tail(20)).prod(min_count=10) - 1).to_numpy(dtype=np.float32)
        rev5 = (-w.tail(5).mean()).to_numpy(dtype=np.float32)

        def z(a):
            m, s = np.nanmean(a), np.nanstd(a)
            return np.where(np.isfinite(a), (a - m) / (s + 1e-9), 0.0)

        vol20, mom20, rev5 = z(vol20), z(mom20), z(rev5)
        F = np.stack([
            [vol20[labels == c].mean(), mom20[labels == c].mean(), rev5[labels == c].mean()]
            for c in ids
        ]).astype(np.float32)
        self._factor_cache[key] = F
        if len(self._factor_cache) > 140:
            self._factor_cache.pop(min(self._factor_cache))
        return F

    def _xy_for_cache(self, cache, t):
        """单编码器视角下的训练集与当日特征。"""
        self.emb = cache
        self._cluster_cache = getattr(cache, "_cc", {})
        x, y = self._train_samples(t)
        stocks, labels, feats, ids = self._clusters(t)
        cache._cc = self._cluster_cache
        return x, y, stocks, labels, feats, ids

    def select(self, t):
        if self._hold_countdown > 0 and self._held:
            self._hold_countdown -= 1
            return self._held
        score_sum, meta = None, None
        for cache in self.caches:
            x, y, stocks, labels, feats, ids = self._xy_for_cache(cache, t)
            if x is None:
                continue
            if self.use_factors:
                # 训练集与预测特征都拼上簇级因子
                xs_f, ys_f = [], []
                for s in range(t - self.train_window, t):
                    st, lb, ft, cid = self._clusters(s)
                    F = self._cluster_factors(s, st, lb, cid)
                    realized = self.market.realized_returns(s).reindex(st).to_numpy(dtype=float)
                    tr = self.market.tradable_at_decision(s).reindex(st).fillna(False).to_numpy(dtype=bool)
                    for pos_i, c in enumerate(cid):
                        member = (lb == c) & tr
                        rets = realized[member]; rets = rets[np.isfinite(rets)]
                        if len(rets) >= self.min_cluster_valid:
                            xs_f.append(np.concatenate([ft[pos_i], F[pos_i]])); ys_f.append(rets.mean())
                x, y = np.stack(xs_f).astype(np.float32), np.array(ys_f, dtype=np.float32)
                F_t = self._cluster_factors(t, stocks, labels, ids)
                feats = np.concatenate([feats, F_t], axis=1)
            model = Ridge(alpha=1.0).fit(x, y)
            s_hat = model.predict(feats)
            score_sum = s_hat if score_sum is None else score_sum + s_hat
            meta = (stocks, labels, ids)
        if score_sum is None:
            return self._held
        stocks, labels, ids = meta
        scores = score_sum / len(self.caches)
        self._last_scores = (scores, ids, stocks, labels)
        tradable = self.market.tradable_at_decision(t)
        selection, count = [], 0
        for pos in np.argsort(-scores):
            members = [s for s, l in zip(stocks, labels) if l == ids[pos]]
            n_tr = sum(1 for s in members if bool(tradable.get(s, False)))
            if n_tr < self.min_tradable_size:
                continue
            selection.extend(members); count += n_tr
            if count >= self.target_stocks:
                break
        self._held = selection
        self._hold_countdown = self.hold_days - 1
        return selection


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v31")
    caches = [EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev) for es in (42, 7, 123)]
    for use_factors in (False, True):
        for seed in (42, 7, 123):
            name = f"v31_{'emb+fac' if use_factors else 'emb'}_h5_s{seed}"
            out = OUT / name
            if (out / "run_summary.json").exists():
                print(f"[skip] {name}"); continue
            strat = EnsembleExecutable(market, caches, use_factors,
                                       cluster_count=20, seed=seed, hold_days=5, device=dev)
            print(f"[run] {name}", flush=True)
            run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out,
                         {"variant": name, "seed": seed}, ic_fn=strat.cluster_ic)
    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, "gross_shp": sharpe_annualized(ge),
                     "nw_t": newey_west_tstat(ge), "net_shp": p["metrics"]["sharpe_excess_net"],
                     "to": p["metrics"]["mean_daily_turnover"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
