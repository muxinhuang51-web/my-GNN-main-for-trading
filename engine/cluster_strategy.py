"""簇级轮动策略（v2）：嵌入 → KMeans → 簇收益 MLP → 逐簇选股。

与旧版差异：
- 嵌入按 (scope, k_e, encoder, date) 磁盘缓存，k/种子扫描零成本复用
- 选簇時的"有效股票数"用 t-1 可交易性估计（旧版偷看 t 日 NaN）
- 簇 IC 作为评估回调返回，与选股路径解耦
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from . import config
from .data import MarketData
from .encoder import StockRGCN, load_encoder
from .features import GraphBuilder


class EmbeddingCache:
    def __init__(self, market: MarketData, encoder_file: Path, k_e: int, device: torch.device):
        self.market = market
        self.builder = GraphBuilder(market)
        self.k_e = k_e
        self.device = device
        self.model = load_encoder(encoder_file, device)
        tag = hashlib.md5(encoder_file.name.encode()).hexdigest()[:10]
        self.dir = config.CACHE_DIR / f"emb_{market.scope}_ke{k_e}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, t: int) -> Tuple[List[str], np.ndarray]:
        import os

        key = self.market.dates[t].strftime("%Y%m%d")
        path = self.dir / f"{key}.npz"
        if path.exists():
            try:
                blob = np.load(path, allow_pickle=True)
                return list(blob["stocks"]), blob["emb"]
            except Exception:
                pass  # 损坏/半写文件：重算并覆盖
        stocks, x, ei, et = self.builder.day_inputs(t, self.k_e)
        with torch.no_grad():
            emb = self.model(x.to(self.device), ei.to(self.device), et.to(self.device), return_embedding=True)
        emb = emb.cpu().numpy().astype(np.float32)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, stocks=np.array(stocks), emb=emb)
        os.replace(tmp, path)
        return stocks, emb


class ClusterRotation:
    """每日：过去 train_window 天的 (簇特征, 簇实现收益) 训练 MLP → 对当日簇打分 → 逐簇选股。"""

    def __init__(
        self,
        market: MarketData,
        emb_cache: EmbeddingCache,
        cluster_count: int,
        seed: int,
        train_window: int = 20,
        target_stocks: int = 100,
        min_cluster_valid: int = 5,
        epochs: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.market = market
        self.emb = emb_cache
        self.k = cluster_count
        self.seed = seed
        self.train_window = train_window
        self.target_stocks = target_stocks
        self.min_cluster_valid = min_cluster_valid
        self.epochs = epochs
        self.device = device or torch.device("cpu")
        self._cluster_cache: Dict[int, Tuple[List[str], np.ndarray, np.ndarray, List[int]]] = {}
        self._last_scores: Optional[Tuple[np.ndarray, List[int], List[str], np.ndarray]] = None

    def _clusters(self, t: int):
        """(stocks, labels, cluster_features, cluster_ids)——只依赖 <= t-1 信息。"""
        if t in self._cluster_cache:
            return self._cluster_cache[t]
        stocks, emb = self.emb.get(t)
        labels = KMeans(n_clusters=self.k, random_state=self.seed, n_init=1).fit_predict(emb)
        ids = sorted(set(labels))
        feats = np.stack([emb[labels == c].mean(axis=0) for c in ids])
        self._cluster_cache[t] = (stocks, labels, feats, ids)
        if len(self._cluster_cache) > self.train_window + 8:  # 控制内存
            oldest = min(self._cluster_cache)
            if oldest != t:  # 永不驱逐刚插入的键（多缓存切换时 t 可能是最小键）
                self._cluster_cache.pop(oldest)
        return self._cluster_cache[t]

    def _train_samples(self, t: int):
        xs, ys = [], []
        for s in range(t - self.train_window, t):
            stocks, labels, feats, ids = self._clusters(s)
            realized = self.market.realized_returns(s).reindex(stocks).to_numpy(dtype=float)
            for pos, c in enumerate(ids):
                member_rets = realized[labels == c]
                member_rets = member_rets[np.isfinite(member_rets)]
                if len(member_rets) >= self.min_cluster_valid:
                    xs.append(feats[pos])
                    ys.append(member_rets.mean())
        if not xs:
            return None, None
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)

    def select(self, t: int) -> List[str]:
        from models.cluster_predictor import ClusterReturnPredictor, train_cluster_predictor

        torch.manual_seed(self.seed + t)  # 每日独立可复现
        x_train, y_train = self._train_samples(t)
        stocks, labels, feats, ids = self._clusters(t)
        if x_train is None:
            self._last_scores = None
            return []
        predictor = ClusterReturnPredictor(input_dim=x_train.shape[1], hidden_dim=32, dropout_rate=0.1)
        predictor = train_cluster_predictor(
            predictor, x_train, y_train, self.device, epochs=self.epochs, batch_size=32, learning_rate=1e-3
        )
        predictor.eval()
        with torch.no_grad():
            scores = predictor(torch.from_numpy(feats).to(self.device)).cpu().numpy()
        self._last_scores = (scores, ids, stocks, labels)

        # 逐簇加入直到预计可买股票数达标（计数用 t-1 可交易性估计）。
        # 返回完整成员列表：可交易性过滤统一由回测引擎执行，
        # 保证与所有基线同一套过滤/持仓延续规则（对抗审查确认项 #2 的一致性要求）。
        tradable = self.market.tradable_at_decision(t)
        selection: List[str] = []
        count = 0
        for pos in np.argsort(-scores):
            members = [s for s, l in zip(stocks, labels) if l == ids[pos]]
            tradable_count = sum(1 for s in members if bool(tradable.get(s, False)))
            if tradable_count == 0:
                continue
            selection.extend(members)
            count += tradable_count
            if count >= self.target_stocks:
                break
        return selection

    def cluster_ic(self, t: int) -> float:
        """评估回调：当日簇预测与簇实现收益的截面相关（仅评估，不进决策）。"""
        if self._last_scores is None:
            return float("nan")
        scores, ids, stocks, labels = self._last_scores
        realized = self.market.realized_returns(t).reindex(stocks).to_numpy(dtype=float)
        real_by_cluster = []
        for c in ids:
            member = realized[labels == c]
            member = member[np.isfinite(member)]
            real_by_cluster.append(member.mean() if len(member) >= self.min_cluster_valid else np.nan)
        real = np.array(real_by_cluster)
        mask = np.isfinite(scores) & np.isfinite(real)
        if mask.sum() < 3:
            return float("nan")
        return float(np.corrcoef(scores[mask], real[mask])[0, 1])
