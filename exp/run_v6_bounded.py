"""v6 有界状态转移模型对决：先验放动力学（M1/M2/M3）vs 放问题定义（v5 FWL）。
全部可行性加权标签、h=5、{top, band} 双收割、同引擎。
"""
import json
import numpy as np, pandas as pd, torch
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.features import build_features

OUT = config.RUNS_DIR / "v6_bounded"
HOLD, TOP, WIN = 5, 100, 60


class BoundedBase:
    def __init__(self, market, cache, harvest):
        self.m, self.cache, self.harvest = market, cache, harvest
        self._held, self._cd = [], 0
        self._X = {}

    def _emb(self, s):
        if s not in self._X:
            self._X[s] = self.cache.get(s)
            if len(self._X) > WIN + 8:
                self._X.pop(min(self._X))
        return self._X[s]

    def _select_from(self, score: pd.Series, t):
        tr = self.m.tradable_at_decision(t)
        score = score[[x for x in score.index if bool(tr.get(x, False))]].dropna()
        if self.harvest == "top":
            self._held = list(score.sort_values(ascending=False).index[:TOP])
        else:
            pct = score.rank(pct=True)
            self._held = list(pct[(pct >= 0.60) & (pct <= 0.90)].index)
        self._cd = HOLD - 1
        return self._held


class M1Markov(BoundedBase):
    """五状态转移：{-2 锁跌停, -1 跌, 0 平, +1 涨, +2 锁涨停}，嵌入作协变量。"""

    def _state(self, s, stocks):
        r = self.m.realized_returns(s).reindex(stocks)
        up = self.m.at_up_limit.iloc[s].reindex(stocks).fillna(0) > 0.5
        dn = self.m.at_down_limit.iloc[s].reindex(stocks).fillna(0) > 0.5
        st = pd.Series(np.nan, index=stocks)
        st[r.notna()] = 0
        st[(r > 0.01)] = 1
        st[(r < -0.01)] = -1
        st[up] = 2
        st[dn] = -2
        return st

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        Xs, ys, mus = [], [], {k: [] for k in (-2, -1, 0, 1, 2)}
        for s in range(t - WIN, t):
            stocks, emb = self._emb(s)
            a = self.m.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
            st = self._state(s, stocks).to_numpy(float)
            r = self.m.realized_returns(s).reindex(stocks).to_numpy(float)
            ok = a & np.isfinite(st) & np.isfinite(r)   # 可行性加权：只在可买样本上学
            Xs.append(emb[ok]); ys.append(st[ok])
            for k in mus:
                mus[k].extend(r[ok & (st == k)].tolist())
        X, y = np.vstack(Xs), np.concatenate(ys)
        clf = LogisticRegression(max_iter=300).fit(X, y)
        mu = np.array([np.mean(mus[k]) if mus[k] else 0.0 for k in clf.classes_])
        stocks, emb = self._emb(t)
        P = clf.predict_proba(emb)
        return self._select_from(pd.Series(P @ mu, index=stocks), t)


class M2Filtered(BoundedBase):
    """删失滤波特征：锁板日收益以截断正态条件期望复原后重算特征，FWL Ridge。"""

    def _filtered_features(self, s):
        stocks = self.m.universe(s)
        win = self.m.window_returns(s, config.LOOKBACK)[stocks]
        up = self.m.at_up_limit.iloc[s - config.LOOKBACK:s].reindex(columns=stocks).fillna(0).to_numpy() > 0.5
        dn = self.m.at_down_limit.iloc[s - config.LOOKBACK:s].reindex(columns=stocks).fillna(0).to_numpy() > 0.5
        W = win.to_numpy(float).copy()
        sig = np.nanstd(np.where(up | dn, np.nan, W), axis=0)
        sig = np.where(np.isfinite(sig) & (sig > 1e-4), sig, 0.02)
        with np.errstate(all="ignore"):
            aU = np.abs(W) / sig      # 标准化边界
            boost = sig * norm.pdf(aU) / np.clip(1 - norm.cdf(aU), 1e-6, None)
        W = np.where(up, W + boost, W)
        W = np.where(dn, W - boost, W)
        fw = pd.DataFrame(W, index=win.index, columns=stocks)
        return stocks, build_features(fw, stocks)

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        Xs, ys = [], []
        for s in range(t - WIN, t):
            stocks, X = self._filtered_features(s)
            r = self.m.realized_returns(s).reindex(stocks).to_numpy(float)
            a = self.m.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
            ok = np.isfinite(r) & a
            Xs.append(X[ok]); ys.append(r[ok])
        model = Ridge(alpha=10.0).fit(np.vstack(Xs), np.concatenate(ys))
        stocks, X_t = self._filtered_features(t)
        return self._select_from(pd.Series(model.predict(X_t), index=stocks), t)


class M3Median(BoundedBase):
    """有界稳健目标：LightGBM 中位数回归（pinball tau=0.5），嵌入特征。"""

    def select(self, t):
        from sklearn.ensemble import HistGradientBoostingRegressor
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        Xs, ys = [], []
        for s in range(t - WIN, t):
            stocks, emb = self._emb(s)
            r = self.m.realized_returns(s).reindex(stocks).to_numpy(float)
            a = self.m.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
            ok = np.isfinite(r) & a
            Xs.append(emb[ok]); ys.append(r[ok])
        model = HistGradientBoostingRegressor(loss="quantile", quantile=0.5,
                                              max_iter=100, learning_rate=0.05)
        model.fit(np.vstack(Xs), np.concatenate(ys))
        stocks, emb = self._emb(t)
        return self._select_from(pd.Series(model.predict(emb), index=stocks), t)


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v6")
    cache = EmbeddingCache(market, encoder_path("all", 0, 42, config.TRAIN_END), 0, dev)
    arms = [("m1_markov", M1Markov), ("m2_filtered", M2Filtered), ("m3_median", M3Median)]
    for name, cls in arms:
        for harvest in ("top", "band"):
            run_name = f"v6_{name}_{harvest}"
            out = OUT / run_name
            if (out / "run_summary.json").exists():
                print(f"[skip] {run_name}", flush=True); continue
            strat = cls(market, cache, harvest)
            print(f"[run] {run_name}", flush=True)
            run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out,
                         {"model": name, "harvest": harvest})
    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, "gross_shp": sharpe_annualized(ge),
                     "nw_t": newey_west_tstat(ge), "net_shp": p["metrics"]["sharpe_excess_net"],
                     "to": p["metrics"]["mean_daily_turnover"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\n参照 v5（同种子 es42）: feasible_band gross +4.78 net +3.57 | feasible_top +0.88 net -0.76")


if __name__ == "__main__":
    raise SystemExit(main())
