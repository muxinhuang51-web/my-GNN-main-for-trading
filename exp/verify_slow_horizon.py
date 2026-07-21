"""独立复核：可交易簇级 IC 是否随预测期限 h 上升（怀疑论者标记的 A2 主张）。

密不透风的时序约定：
- 训练样本 (s, y)：特征 = s-1 及之前窗口的嵌入簇均值；
  标签 y = 簇内"s 日决策时可买"成员在 [s, s+h-1] 的累计收益均值（停牌日冻结计 0）
- 决策日 t：只用 s <= t-h 的样本（标签最晚覆盖到 t-1，严格先于预测窗口）
- 预测目标：[t, t+h-1] 的可交易簇累计收益；前视窗口不得越过 2024-01-01（test 防火墙）
- 评估：逐日 Spearman IC，重叠窗口用 Newey-West lags = h+5
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from engine import config
from engine.backtest import newey_west_tstat
from engine.cluster_strategy import EmbeddingCache
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path

HORIZONS = [1, 2, 3, 5, 10]
TRAIN_DAYS = 60
K = 20

market = MarketData("all")
cache = EmbeddingCache(market, encoder_path("all", 0, 42, config.TRAIN_END), 0, torch.device("cpu"))

positions = market.positions_in_range(config.VALID_START, config.VALID_END, config.LOOKBACK + 25)
last_valid_pos = positions[-1]

cluster_cache = {}


def clusters(t):
    if t not in cluster_cache:
        stocks, emb = cache.get(t)
        labels = KMeans(n_clusters=K, random_state=42, n_init=1).fit_predict(emb)
        ids = sorted(set(labels))
        feats = np.stack([emb[labels == c].mean(axis=0) for c in ids])
        if len(cluster_cache) > 160:
            oldest = min(cluster_cache)
            if oldest != t:
                cluster_cache.pop(oldest)
        cluster_cache[t] = (stocks, labels, feats, ids)
    return cluster_cache[t]


def tradable_cum_return(t, h, stocks, member_mask):
    """s..s+h-1 的可交易成员累计收益均值（决策可买用 t-1 信息；停牌冻结 0）。"""
    tr = market.tradable_at_decision(t).reindex(stocks).fillna(False).to_numpy(dtype=bool)
    sel = member_mask & tr
    if sel.sum() < 5:
        return np.nan
    seg = market.returns.iloc[t : t + h].reindex(columns=stocks).to_numpy(dtype=float)
    seg = np.where(np.isfinite(seg), seg, 0.0)
    cum = np.prod(1 + seg[:, sel], axis=0) - 1
    return float(cum.mean())


rng = np.random.default_rng(0)
print(f"验证期决策日 {len(positions)} 个")
for h in HORIZONS:
    for variant in ("main", "purged", "permuted"):
        # purged: 训练样本再退一天（s <= t-h-1），检验边界敏感性
        # permuted: 当日真实簇收益随机打乱，任何机械泄漏都会现形（应为 0）
        gap = h + 1 if variant == "purged" else h
        ics = []
        usable = [t for t in positions if t + h - 1 <= last_valid_pos]
        assert_no_test_access(market.dates[usable[-1] + h - 1], f"verify_h{h}")
        for t in usable[TRAIN_DAYS + gap :: 1]:
            xs, ys = [], []
            for s in range(t - TRAIN_DAYS - gap + 1, t - gap + 1):
                st, lb, ft, ids = clusters(s)
                for pos_i, c in enumerate(ids):
                    y = tradable_cum_return(s, h, st, lb == c)
                    if np.isfinite(y):
                        xs.append(ft[pos_i]); ys.append(y)
            if len(xs) < 100:
                continue
            model = Ridge(alpha=1.0).fit(np.stack(xs), np.array(ys))
            st, lb, ft, ids = clusters(t)
            preds = model.predict(ft)
            reals = np.array([tradable_cum_return(t, h, st, lb == c) for c in ids])
            mask = np.isfinite(preds) & np.isfinite(reals)
            if mask.sum() >= 5:
                r = reals[mask]
                if variant == "permuted":
                    r = rng.permutation(r)
                ics.append(spearmanr(preds[mask], r).statistic)
        arr = np.array(ics, dtype=float)
        arr = arr[np.isfinite(arr)]
        t_nw = newey_west_tstat(arr, lags=h + 5)
        print(f"h={h:2d} [{variant:8s}]: 可交易簇 IC = {arr.mean():+.4f}  (NW t={t_nw:+.2f}, n={len(arr)})", flush=True)
print("done")
