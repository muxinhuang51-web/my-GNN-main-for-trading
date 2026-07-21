"""节点特征与图构造（决策日 t 只使用 [t-L, t) 窗口）。

与旧版 backtest_cluster.py 的差异：
- 历史不足的股票在 universe 层面剔除（旧版填 0 = 钉在截面均值，会被选入组合）
- 行业边用 PIT 申万一级（旧版用 tushare 111 类静态快照）
- 相关邻居每日只算一次 top-CORR_CACHE_TOPK 并落盘，k_e<=40 切片复用
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import config
from .data import MarketData


def build_features(window: pd.DataFrame, stocks: List[str]) -> np.ndarray:
    """6 维特征 + 截面 z-score。window 为 [t-L, t) 收益，列已限定 universe。"""
    w = window[stocks]
    feats = pd.DataFrame(
        {
            "mom20": (1 + w.tail(20)).prod(min_count=10) - 1,
            "mean5": w.tail(5).mean(),
            "mean10": w.tail(10).mean(),
            "vol20": w.tail(20).std(ddof=0),
            "vol60": w.std(ddof=0),
            "last_ret": w.tail(1).iloc[0],
        }
    )
    z = (feats - feats.mean()) / feats.std().replace(0, np.nan)
    return z.fillna(0.0).to_numpy(dtype=np.float32)  # 此处仅存少量残缺，universe 已过滤主缺失


def industry_edges(industry_map: pd.Series, stocks: List[str]) -> np.ndarray:
    """同行业有向边，每股最多 INDUSTRY_MAX_NEIGHBORS 个同组邻居。"""
    stock2idx = {s: i for i, s in enumerate(stocks)}
    pairs = []
    groups: Dict[str, List[str]] = {}
    for code in stocks:
        ind = industry_map.get(code)
        if isinstance(ind, str):
            groups.setdefault(ind, []).append(code)
    for members in groups.values():
        for src in members:
            peers = [m for m in members if m != src][: config.INDUSTRY_MAX_NEIGHBORS]
            pairs.extend((stock2idx[src], stock2idx[p]) for p in peers)
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(sorted(set(pairs)), dtype=np.int64).T


def corr_neighbor_table(window: pd.DataFrame, stocks: List[str], min_overlap: int = 20) -> np.ndarray:
    """返回 (N, CORR_CACHE_TOPK) 的邻居索引表（按 |corr| 降序），不足处填 -1。"""
    n = len(stocks)
    table = np.full((n, config.CORR_CACHE_TOPK), -1, dtype=np.int32)
    w = window[stocks].to_numpy(dtype=np.float32)
    valid = np.isfinite(w)
    w0 = np.where(valid, w, 0.0)
    counts = valid.astype(np.float32).T @ valid.astype(np.float32)
    sums = w0.T @ valid.astype(np.float32)
    sq = (w0 * w0).T @ valid.astype(np.float32)
    prod = w0.T @ w0
    with np.errstate(invalid="ignore", divide="ignore"):
        # 配对缺失处理：均值/方差都在两股重叠有效日上计算
        mean_i = sums / counts
        mean_j = (valid.astype(np.float32).T @ w0) / counts
        cov = prod / counts - mean_i * mean_j
        var_i = sq / counts - mean_i**2
        var_j = (valid.astype(np.float32).T @ (w0 * w0)) / counts - mean_j**2
        corr = cov / np.sqrt(var_i * var_j)
    corr[counts < min_overlap] = np.nan
    np.fill_diagonal(corr, np.nan)
    absC = np.abs(corr)
    absC[~np.isfinite(absC)] = -1.0
    k = min(config.CORR_CACHE_TOPK, n - 1)
    top = np.argpartition(-absC, k, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    order = np.argsort(-absC[rows, top], axis=1)
    top_sorted = top[rows, order]
    invalid = absC[rows, top_sorted] < 0
    top_sorted[invalid] = -1
    table[:, :k] = top_sorted
    return table


def corr_edges_from_table(table: np.ndarray, k_e: int) -> np.ndarray:
    """从缓存邻居表切前 k_e 个邻居生成有向边。"""
    if k_e <= 0:
        return np.empty((2, 0), dtype=np.int64)
    k_e = min(k_e, table.shape[1])
    src = np.repeat(np.arange(table.shape[0]), k_e)
    dst = table[:, :k_e].ravel()
    mask = dst >= 0
    return np.stack([src[mask], dst[mask].astype(np.int64)])


class GraphBuilder:
    """带磁盘缓存的逐日图构造。缓存键: (scope, date) -> 特征、行业边、相关邻居表。"""

    def __init__(self, market: MarketData):
        self.market = market
        self.cache_root = config.CACHE_DIR / f"graph_{market.scope}"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def day_inputs(self, t: int, k_e: int) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (universe, x, edge_index, edge_type)。全部只依赖 <= t-1 信息。"""
        import os

        date_key = self.market.dates[t].strftime("%Y%m%d")
        path = self.cache_root / f"{date_key}.npz"
        blob = None
        if path.exists():
            try:
                blob = np.load(path, allow_pickle=True)
                stocks = list(blob["stocks"])
                x = blob["x"]
                ind_edges = blob["ind_edges"]
                corr_table = blob["corr_table"]
            except Exception:
                blob = None  # 损坏/半写文件：重算并覆盖
        if blob is None:
            stocks = self.market.universe(t)
            window = self.market.window_returns(t, config.LOOKBACK)
            x = build_features(window, stocks)
            ind_edges = industry_edges(self.market.industry_map(t), stocks)
            corr_table = corr_neighbor_table(window, stocks)
            tmp = self.cache_root / f".{date_key}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, stocks=np.array(stocks), x=x, ind_edges=ind_edges, corr_table=corr_table)
            os.replace(tmp, path)  # 原子替换，读者永远看不到半写文件
        c_edges = corr_edges_from_table(corr_table, k_e)
        edge_index = np.concatenate([ind_edges, c_edges], axis=1)
        edge_type = np.concatenate(
            [np.zeros(ind_edges.shape[1], dtype=np.int64), np.full(c_edges.shape[1], 1, dtype=np.int64)]
        )
        return (
            stocks,
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(edge_index),
            torch.from_numpy(edge_type),
        )
