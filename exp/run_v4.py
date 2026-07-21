"""v4 通宵矩阵：判决理论预测 T1/T2/T3（paper-theory-v4.md §5）。全部 valid 期。

E1 标签三臂：mask/delete/tobit x 9 单模型（3es x 3km，band 收割）——判 T1
E2 后处理四臂：band / grinold / tweedie / ct_tweedie（mask 标签共识分，隔离 C3）——判 T2
E3 影子价格：ct_tweedie 分 - lambda*sigma_z*P(锁板)，lambda 网格——判 T3
E4 全组合：tobit 标签 + ct_tweedie + lambda* 影子价格，band 与 top100 双口径
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.v4_components import TobitLabelRotation, LockProbability, tweedie_correct, grinold_baseline

OUT = config.RUNS_DIR / "v4"
ES, KM = (42, 7, 123), (42, 7, 123)
HOLD, TOP = 5, 100


class ConsensusV4:
    """9 单模型逐股分 -> (z, s2) -> 指定后处理 -> 选择。"""

    def __init__(self, market, caches, label_mode, post, lam=0.0, lockprob=None):
        self.m = market
        self.models = [TobitLabelRotation(market, c, 20, km, label_mode=label_mode,
                                          hold_days=HOLD, device=torch.device("cpu"))
                       for c in caches for km in KM]
        # 每个 (cache, km) 组合一个模型：上行会让同 cache 重复 KM——修正为按对构造
        self.models = []
        for c in caches:
            for km in KM:
                self.models.append(TobitLabelRotation(market, c, 20, km, label_mode=label_mode,
                                                      hold_days=HOLD, device=torch.device("cpu")))
        self.post = post
        self.lam = lam
        self.lockprob = lockprob
        self._held, self._cd = [], 0

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        cols = []
        for m in self.models:
            sc = m.score_stocks(t)
            if sc is not None:
                cols.append(sc)
        if not cols:
            return self._held
        M = pd.concat(cols, axis=1)
        z, s2 = M.mean(axis=1), M.var(axis=1)
        tradable = self.m.tradable_at_decision(t)
        z = z[[s for s in z.index if bool(tradable.get(s, False))]].dropna()
        s2 = s2.reindex(z.index)

        if self.post == "band":
            pct = z.rank(pct=True)
            self._held = list(pct[(pct >= 0.60) & (pct <= 0.90)].index)
        else:
            if self.post == "grinold":
                score = grinold_baseline(z)
            elif self.post in ("tweedie", "ct_tweedie", "shadow"):
                locked = None
                if self.post in ("ct_tweedie", "shadow"):
                    prev = t - 1
                    locked = (self.m.at_up_limit.iloc[prev].reindex(z.index).fillna(0) > 0.5) | \
                             (self.m.at_down_limit.iloc[prev].reindex(z.index).fillna(0) > 0.5)
                score = tweedie_correct(z, s2, locked_mask=locked)
                if self.post == "shadow":
                    p = self.lockprob.predict(t, list(score.index))
                    score = score - self.lam * float(z.std()) * pd.Series(p, index=score.index)
            else:
                score = z
            self._held = list(score.sort_values(ascending=False).index[:TOP])
        self._cd = HOLD - 1
        return self._held


def _run(market, strat, name, params):
    out = OUT / name
    if (out / "run_summary.json").exists():
        print(f"[skip] {name}", flush=True)
        return
    print(f"[run] {name}", flush=True)
    run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out, params)


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v4")
    caches = {es: EmbeddingCache(market, encoder_path("all", 0, es, config.TRAIN_END), 0, dev) for es in ES}

    # E1：标签三臂 x 9 单模型（band 收割）
    for mode in ("mask", "delete", "tobit"):
        for es in ES:
            for km in KM:
                strat = TobitLabelRotation(market, caches[es], 20, km, label_mode=mode,
                                           hold_days=HOLD, device=dev)
                _run(market, strat, f"e1_{mode}_es{es}_s{km}", {"exp": "E1", "mode": mode, "es": es, "km": km})

    # E2：后处理四臂（mask 标签，共识）
    for post in ("band", "grinold", "tweedie", "ct_tweedie"):
        strat = ConsensusV4(market, list(caches.values()), "mask", post)
        _run(market, strat, f"e2_{post}", {"exp": "E2", "post": post})

    # E3：影子价格 lambda 网格（ct_tweedie 基础上）
    lp = LockProbability(market)
    for lam in (0.5, 1.0, 2.0, 4.0):
        strat = ConsensusV4(market, list(caches.values()), "mask", "shadow", lam=lam, lockprob=lp)
        _run(market, strat, f"e3_shadow_lam{lam}", {"exp": "E3", "lam": lam})

    # E4：全组合（tobit 标签 + ct_tweedie / + shadow lam=1 / band）
    strat = ConsensusV4(market, list(caches.values()), "tobit", "ct_tweedie")
    _run(market, strat, "e4_full_topk", {"exp": "E4", "variant": "tobit+ct_tweedie+top100"})
    strat = ConsensusV4(market, list(caches.values()), "tobit", "shadow", lam=1.0, lockprob=lp)
    _run(market, strat, "e4_full_shadow", {"exp": "E4", "variant": "tobit+ct_tweedie+shadow1"})
    strat = ConsensusV4(market, list(caches.values()), "tobit", "band")
    _run(market, strat, "e4_full_band", {"exp": "E4", "variant": "tobit+band"})

    # 汇总
    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, **p["params"],
                     "gross_shp": sharpe_annualized(ge), "nw_t": newey_west_tstat(ge),
                     "net_shp": p["metrics"]["sharpe_excess_net"],
                     "to": p["metrics"]["mean_daily_turnover"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    print(df.round(2).to_string(index=False))

    # E1 配对（T1 判决）
    e1 = df[df["exp"] == "E1"]
    piv = e1.pivot_table(index=["es", "km"], columns="mode", values="gross_shp")
    for a, b in (("tobit", "mask"), ("tobit", "delete"), ("mask", "delete")):
        if a in piv and b in piv:
            diff = (piv[a] - piv[b]).dropna()
            print(f"T1 配对 {a}-{b}: mean={diff.mean():+.2f} 全正={(diff>0).all()} "
                  f"t={diff.mean()/diff.std()*np.sqrt(len(diff)):+.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
