"""执行感知簇轮动（v3 原型）。

相对 ClusterRotation 的三处修改，逐一可消融：
1. 训练标签 = 簇内"当日决策可买"成员的收益均值（而非全体成员）——
   直接修复"预测器学到不可执行 alpha"的目标错位
2. 选簇时跳过可交易成员数 < min_tradable_size 的簇（极端小簇两端都被证实为伪信号）
3. 可选 h 日持有：每 h 个交易日才重新选簇，换手约降为 1/h
预测器用 Ridge（确定性，与独立复核脚本一致），训练窗口 60 日。
"""

from typing import List, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .cluster_strategy import ClusterRotation, EmbeddingCache
from .data import MarketData


class ExecutableClusterRotation(ClusterRotation):
    def __init__(
        self,
        market: MarketData,
        emb_cache: EmbeddingCache,
        cluster_count: int,
        seed: int,
        label_mode: str = "tradable",     # tradable | all（消融）
        min_tradable_size: int = 10,
        hold_days: int = 1,
        train_window: int = 60,
        target_stocks: int = 100,
        device: Optional[torch.device] = None,
    ):
        super().__init__(market, emb_cache, cluster_count, seed,
                         train_window=train_window, target_stocks=target_stocks, device=device)
        self.label_mode = label_mode
        self.min_tradable_size = min_tradable_size
        self.hold_days = hold_days
        self._held: List[str] = []
        self._hold_countdown = 0

    def _train_samples(self, t: int):
        xs, ys = [], []
        for s in range(t - self.train_window, t):
            stocks, labels, feats, ids = self._clusters(s)
            realized = self.market.realized_returns(s).reindex(stocks).to_numpy(dtype=float)
            if self.label_mode == "tradable":
                tr = self.market.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(dtype=bool)
            else:
                tr = np.ones(len(stocks), dtype=bool)
            for pos_i, c in enumerate(ids):
                member = (labels == c) & tr
                rets = realized[member]
                rets = rets[np.isfinite(rets)]
                if len(rets) >= self.min_cluster_valid:
                    xs.append(feats[pos_i]); ys.append(rets.mean())
        if not xs:
            return None, None
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)

    def select(self, t: int) -> List[str]:
        # h 日持有：非调仓日直接返回既有持仓（引擎照常做可交易过滤与强制持仓）
        if self._hold_countdown > 0 and self._held:
            self._hold_countdown -= 1
            return self._held

        x_train, y_train = self._train_samples(t)
        stocks, labels, feats, ids = self._clusters(t)
        if x_train is None:
            self._last_scores = None
            return self._held
        model = Ridge(alpha=1.0).fit(x_train, y_train)
        scores = model.predict(feats)
        self._last_scores = (scores, ids, stocks, labels)

        tradable = self.market.tradable_at_decision(t)
        selection: List[str] = []
        count = 0
        for pos in np.argsort(-scores):
            members = [s for s, l in zip(stocks, labels) if l == ids[pos]]
            tradable_count = sum(1 for s in members if bool(tradable.get(s, False)))
            if tradable_count < self.min_tradable_size:
                continue  # 极端小簇（两端伪信号）直接跳过
            selection.extend(members)
            count += tradable_count
            if count >= self.target_stocks:
                break
        self._held = selection
        self._hold_countdown = self.hold_days - 1
        return selection
