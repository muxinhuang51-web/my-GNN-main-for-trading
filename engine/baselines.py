"""基线策略集：与主策略共用同一回测引擎与指标，保证可比。"""

from typing import List

import numpy as np
import pandas as pd
import torch

from . import config
from .cluster_strategy import ClusterRotation, EmbeddingCache
from .data import MarketData
from .features import build_features


class MomentumTopK:
    """截面动量：mom20 最高的 top_k 只（可交易过滤由引擎统一执行）。"""

    def __init__(self, market: MarketData, top_k: int = 100, ascending: bool = False):
        self.market = market
        self.top_k = top_k
        self.ascending = ascending  # True = 反转（选跌最多的）

    def select(self, t: int) -> List[str]:
        stocks = self.market.universe(t)
        window = self.market.window_returns(t, config.LOOKBACK)
        mom = (1 + window[stocks].tail(20)).prod(min_count=10) - 1
        mom = mom.dropna().sort_values(ascending=self.ascending)
        return list(mom.index[: self.top_k])


class RandomTopK:
    """随机选股：收益归因的零假设基线。"""

    def __init__(self, market: MarketData, top_k: int = 100, seed: int = 0):
        self.market = market
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

    def select(self, t: int) -> List[str]:
        stocks = self.market.universe(t)
        if len(stocks) <= self.top_k:
            return stocks
        return list(self.rng.choice(stocks, size=self.top_k, replace=False))


class StockLevelRGCN:
    """个股级基线：同一编码器的回归头直接打分，top_k 等权（与簇级同公式同引擎）。"""

    def __init__(self, market: MarketData, emb_cache: EmbeddingCache, top_k: int = 100):
        self.market = market
        self.emb_cache = emb_cache
        self.top_k = top_k

    def select(self, t: int) -> List[str]:
        stocks, x, ei, et = self.emb_cache.builder.day_inputs(t, self.emb_cache.k_e)
        with torch.no_grad():
            scores = self.emb_cache.model(
                x.to(self.emb_cache.device), ei.to(self.emb_cache.device), et.to(self.emb_cache.device)
            ).cpu().numpy()
        order = np.argsort(-scores)
        return [stocks[i] for i in order[: self.top_k]]


class IndustryRotation(ClusterRotation):
    """行业轮动基线：用申万一级行业代替 KMeans 分簇，其余与簇级策略完全一致。

    回答评审必问：学习到的动态簇是否优于现成行业分组？
    """

    def _clusters(self, t: int):
        if t in self._cluster_cache:
            return self._cluster_cache[t]
        stocks, emb = self.emb.get(t)
        ind_map = self.market.industry_map(t)
        names = sorted({ind_map.get(s) for s in stocks if isinstance(ind_map.get(s), str)})
        name2id = {n: i for i, n in enumerate(names)}
        labels = np.array([name2id.get(ind_map.get(s), -1) for s in stocks])
        ids = sorted({l for l in labels if l >= 0})
        feats = np.stack([emb[labels == c].mean(axis=0) for c in ids])
        self._cluster_cache[t] = (stocks, labels, feats, ids)
        if len(self._cluster_cache) > self.train_window + 8:
            self._cluster_cache.pop(min(self._cluster_cache))
        return self._cluster_cache[t]


class RandomClusterRotation(ClusterRotation):
    """随机分簇基线：同 k、同 MLP、同选簇规则，仅簇标签随机。

    回答评审必问：簇级增益是来自嵌入聚类结构，还是仅仅来自"标签平均降噪+分组下注"？
    """

    def _clusters(self, t: int):
        if t in self._cluster_cache:
            return self._cluster_cache[t]
        stocks, emb = self.emb.get(t)
        rng = np.random.default_rng(self.seed * 100003 + t)
        labels = rng.integers(0, self.k, size=len(stocks))
        ids = sorted(set(labels.tolist()))
        feats = np.stack([emb[labels == c].mean(axis=0) for c in ids])
        self._cluster_cache[t] = (stocks, labels, feats, ids)
        if len(self._cluster_cache) > self.train_window + 8:
            self._cluster_cache.pop(min(self._cluster_cache))
        return self._cluster_cache[t]


class RawFeatureKMeans(ClusterRotation):
    """无 GNN 基线：直接对 6 维原始特征做 KMeans（回答"嵌入是否必要"）。"""

    def _clusters(self, t: int):
        if t in self._cluster_cache:
            return self._cluster_cache[t]
        from sklearn.cluster import KMeans

        stocks, emb = self.emb.get(t)  # 仍取 emb 以对齐 universe；聚类用原始特征
        window = self.market.window_returns(t, config.LOOKBACK)
        x = build_features(window, stocks)
        labels = KMeans(n_clusters=self.k, random_state=self.seed, n_init=1).fit_predict(x)
        ids = sorted(set(labels.tolist()))
        feats = np.stack([x[labels == c].mean(axis=0) for c in ids])
        self._cluster_cache[t] = (stocks, labels, feats, ids)
        if len(self._cluster_cache) > self.train_window + 8:
            self._cluster_cache.pop(min(self._cluster_cache))
        return self._cluster_cache[t]
