"""v5 优雅性检验：股票级 Feasibility-Weighted Learning。

假设：尾部偏差 b 是簇聚合人造物；股票级可行性加权学习不需要带收割。
臂：{nominal, feasible} 标签 x {top100, band60-90} 收割 x {嵌入, 原始特征}，h=5。
"""
import json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.features import build_features

OUT = config.RUNS_DIR / "v5_fwl"
HOLD, TOP, WIN = 5, 100, 60


class StockFWL:
    def __init__(self, market, cache, label_mode, harvest, rep):
        self.m, self.cache = market, cache
        self.label_mode, self.harvest, self.rep = label_mode, harvest, rep
        self._held, self._cd = [], 0
        self._X = {}  # s -> (stocks, X)

    def _xs(self, s):
        if s not in self._X:
            if self.rep == "emb":
                stocks, emb = self.cache.get(s)
                self._X[s] = (stocks, emb)
            else:
                stocks = self.m.universe(s)
                self._X[s] = (stocks, build_features(self.m.window_returns(s, config.LOOKBACK), stocks))
            if len(self._X) > WIN + 8:
                self._X.pop(min(self._X))
        return self._X[s]

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        Xs, ys = [], []
        for s in range(t - WIN, t):
            stocks, X = self._xs(s)
            r = self.m.realized_returns(s).reindex(stocks).to_numpy(float)
            ok = np.isfinite(r)
            if self.label_mode == "feasible":
                a = self.m.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
                ok &= a  # 一行修正：损失只在可买样本上取
            Xs.append(X[ok]); ys.append(r[ok])
        X_train = np.vstack(Xs); y_train = np.concatenate(ys)
        model = Ridge(alpha=10.0).fit(X_train, y_train)
        stocks, X_t = self._xs(t)
        score = pd.Series(model.predict(X_t), index=stocks)
        tr = self.m.tradable_at_decision(t)
        score = score[[s_ for s_ in score.index if bool(tr.get(s_, False))]].dropna()
        if self.harvest == "top":
            self._held = list(score.sort_values(ascending=False).index[:TOP])
        else:
            pct = score.rank(pct=True)
            self._held = list(pct[(pct >= 0.60) & (pct <= 0.90)].index)
        self._cd = HOLD - 1
        return self._held


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v5")
    cache = EmbeddingCache(market, encoder_path("all", 0, 42, config.TRAIN_END), 0, dev)
    for rep in ("emb", "raw"):
        for mode in ("nominal", "feasible"):
            for harvest in ("top", "band"):
                name = f"v5_{rep}_{mode}_{harvest}"
                out = OUT / name
                if (out / "run_summary.json").exists():
                    print(f"[skip] {name}", flush=True); continue
                strat = StockFWL(market, cache, mode, harvest, rep)
                print(f"[run] {name}", flush=True)
                run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out,
                             {"rep": rep, "mode": mode, "harvest": harvest})
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


def seed_check():
    """emb_feasible_band 的编码器种子稳健性。"""
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v5seed")
    for es in (7, 123):
        cache = EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev)
        name = f"v5_emb_feasible_band_es{es}"
        out = OUT / name
        if (out / "run_summary.json").exists():
            continue
        strat = StockFWL(market, cache, "feasible", "band", "emb")
        print(f"[run] {name}", flush=True)
        run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out,
                     {"rep": "emb", "mode": "feasible", "harvest": "band", "es": es})
    for f in sorted(OUT.glob("v5_emb_feasible_band*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        print(f"{f.parent.name}: gross {sharpe_annualized(ge):+.2f} (t={newey_west_tstat(ge):+.2f}) net {p['metrics']['sharpe_excess_net']:+.2f}")


if __name__ == "__main__" and __import__("sys").argv[-1] == "seedcheck":
    seed_check()
