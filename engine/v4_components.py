"""v4 三个原理化组件（理论见 paper-theory-v4.md）。

C1 TobitLabelRotation：标签三臂 mask / delete / tobit（EM 修正冲板删失）
C3 tweedie_correct：删失感知 Tweedie 后验分（Robbins 公式 + 分歧方差 + 锁定膨胀）
C2 ShadowPriceSelector：分数 - lambda * P(锁板|x) 的 top-k 选择
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde, norm
from sklearn.linear_model import LogisticRegression, Ridge

from . import config
from .cluster_strategy import ClusterRotation
from .data import MarketData


# ---------------- C1: 标签三臂 ----------------

class TobitLabelRotation(ClusterRotation):
    """label_mode: mask（可买成员全部观测值）/ delete（再剔除当日冲板收盘者）/
    tobit（冲板者以截断正态条件期望 EM 插补）。h 日持有与选簇沿用 v3 语义。"""

    def __init__(self, market, emb_cache, cluster_count, seed, label_mode="mask",
                 hold_days=5, min_tradable_size=10, train_window=60, **kw):
        super().__init__(market, emb_cache, cluster_count, seed,
                         train_window=train_window, **kw)
        self.label_mode = label_mode
        self.hold_days = hold_days
        self.min_tradable_size = min_tradable_size
        self._held: List[str] = []
        self._cd = 0

    def _day_label_inputs(self, s):
        stocks, labels, feats, ids = self._clusters(s)
        realized = self.market.realized_returns(s).reindex(stocks).to_numpy(dtype=float)
        tr = self.market.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
        up = self.market.at_up_limit.iloc[s].reindex(stocks).fillna(0).to_numpy(float) > 0.5
        dn = self.market.at_down_limit.iloc[s].reindex(stocks).fillna(0).to_numpy(float) > 0.5
        return stocks, labels, feats, ids, realized, tr, up, dn

    def _train_samples(self, t):
        rows = []  # (s, cluster_pos, feat, member_rets, locked_dir) 展平后按模式聚合
        for s in range(t - self.train_window, t):
            stocks, labels, feats, ids, realized, tr, up, dn = self._day_label_inputs(s)
            for pos_i, c in enumerate(ids):
                member = (labels == c) & tr
                rets = realized[member]
                lock = np.where(up[member], 1, np.where(dn[member], -1, 0))
                ok = np.isfinite(rets)
                if ok.sum() >= self.min_cluster_valid:
                    rows.append((feats[pos_i], rets[ok], lock[ok]))
        if not rows:
            return None, None
        X = np.stack([r[0] for r in rows]).astype(np.float32)

        if self.label_mode == "mask":
            y = np.array([r[1].mean() for r in rows], dtype=np.float32)
            return X, y
        if self.label_mode == "delete":
            y = np.array([r[1][r[2] == 0].mean() if (r[2] == 0).sum() >= 3 else np.nan
                          for r in rows], dtype=np.float32)
            keep = np.isfinite(y)
            return X[keep], y[keep]
        # tobit：EM 插补冲板收盘成员的潜在收益（阈值 = 其观测收益本身）
        y = np.array([r[1].mean() for r in rows], dtype=np.float32)  # 初始化 = mask
        model = Ridge(alpha=1.0)
        sigma = None
        for _ in range(3):
            model.fit(X, y)
            pred = model.predict(X)
            resid = []
            y_new = np.empty_like(y)
            for j, (f_j, rets, lock) in enumerate(rows):
                mu = pred[j]
                free = rets[lock == 0]
                resid.extend((free - mu).tolist())
            sigma = max(float(np.std(resid)), 1e-4)
            for j, (f_j, rets, lock) in enumerate(rows):
                mu = pred[j]
                imput = rets.copy()
                a_up = (rets[lock == 1] - mu) / sigma
                imput_up = mu + sigma * norm.pdf(a_up) / np.clip(1 - norm.cdf(a_up), 1e-6, None)
                imput[lock == 1] = imput_up
                a_dn = (rets[lock == -1] - mu) / sigma
                imput_dn = mu - sigma * norm.pdf(a_dn) / np.clip(norm.cdf(a_dn), 1e-6, None)
                imput[lock == -1] = imput_dn
                y_new[j] = imput.mean()
            y = y_new.astype(np.float32)
        return X, y

    def score_stocks(self, t) -> Optional[pd.Series]:
        """逐股打分（= 所属簇的预测收益；小簇成员置 NaN）。共识层复用。"""
        x_train, y_train = self._train_samples(t)
        if x_train is None:
            return None
        stocks, labels, feats, ids = self._clusters(t)
        model = Ridge(alpha=1.0).fit(x_train, y_train)
        scores = model.predict(feats)
        self._last_scores = (scores, ids, stocks, labels)
        tradable = self.market.tradable_at_decision(t)
        id2score = {c: scores[j] for j, c in enumerate(ids)}
        tr_count = {c: 0 for c in ids}
        for s_, l_ in zip(stocks, labels):
            if l_ in tr_count and bool(tradable.get(s_, False)):
                tr_count[l_] += 1
        out = pd.Series(
            {s_: (id2score[l_] if tr_count.get(l_, 0) >= self.min_tradable_size else np.nan)
             for s_, l_ in zip(stocks, labels) if l_ in id2score})
        return out

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        sc = self.score_stocks(t)
        if sc is None:
            return self._held
        tradable = self.market.tradable_at_decision(t)
        sc = sc.dropna()
        ok = [s for s in sc.index if bool(tradable.get(s, False))]
        pct = sc[ok].rank(pct=True)
        self._held = list(pct[(pct >= 0.60) & (pct <= 0.90)].index)  # band 收割口径
        self._cd = self.hold_days - 1
        return self._held


# ---------------- C3: 删失感知 Tweedie ----------------

def tweedie_correct(z: pd.Series, s2: pd.Series, locked_mask: Optional[pd.Series] = None,
                    censor_inflate: float = 1.0) -> pd.Series:
    """theta_hat = z + s2_eff * d/dz log m(z)。m 用当日截面 KDE 估计。
    locked_mask 为 True 的名字 s2 乘 (1+censor_inflate)。"""
    z = z.dropna()
    zz = z.to_numpy(dtype=float)
    if len(zz) < 50:
        return z
    zs = (zz - zz.mean()) / (zz.std() + 1e-12)
    kde = gaussian_kde(zs, bw_method="scott")
    eps = 0.05
    dlogm = (np.log(kde(zs + eps) + 1e-300) - np.log(kde(zs - eps) + 1e-300)) / (2 * eps)
    s2v = s2.reindex(z.index).fillna(s2.median()).to_numpy(dtype=float)
    if locked_mask is not None:
        infl = locked_mask.reindex(z.index).fillna(False).to_numpy(bool)
        s2v = np.where(infl, s2v * (1 + censor_inflate), s2v)
    # 标准化域内的收缩，再映回原尺度
    s2_std = s2v / (zz.std() ** 2 + 1e-12)
    corrected = zs + s2_std * dlogm
    return pd.Series(corrected * (zz.std() + 1e-12) + zz.mean(), index=z.index)


def grinold_baseline(z: pd.Series) -> pd.Series:
    """Grinold 风格基线：3σ 截尾（IC 缩放不改排序，略）。"""
    z = z.dropna()
    mu, sd = z.mean(), z.std()
    return z.clip(mu - 3 * sd, mu + 3 * sd)


# ---------------- C2: 影子价格 ----------------

class LockProbability:
    """监督学习 P(明日锁板 | 今日特征)：逻辑回归，训练窗口 60 日滚动。"""

    def __init__(self, market: MarketData, train_window: int = 60):
        self.m = market
        self.w = train_window
        self._model = None
        self._last_fit = -999

    def _features(self, s, stocks):
        win = self.m.window_returns(s, 21)[stocks]
        r1 = win.tail(1).iloc[0].to_numpy(float)
        mom5 = win.tail(5).mean().to_numpy(float)
        vol20 = win.std(ddof=0).to_numpy(float)
        up_y = self.m.at_up_limit.iloc[s - 1].reindex(stocks).fillna(0).to_numpy(float)
        X = np.column_stack([r1, mom5, vol20, up_y])
        return np.nan_to_num(X, nan=0.0)

    def predict(self, t, stocks) -> np.ndarray:
        if t - self._last_fit >= 5:  # 每 5 日重训
            Xs, ys = [], []
            for s in range(t - self.w, t - 1):
                st = self.m.universe(s)
                lock_next = (self.m.at_up_limit.iloc[s + 1].reindex(st).fillna(0) > 0.5) | \
                            (self.m.at_down_limit.iloc[s + 1].reindex(st).fillna(0) > 0.5)
                Xs.append(self._features(s, st)); ys.append(lock_next.to_numpy(bool))
            X = np.vstack(Xs); y = np.concatenate(ys)
            self._model = LogisticRegression(max_iter=200).fit(X, y)
            self._last_fit = t
        return self._model.predict_proba(self._features(t, stocks))[:, 1]
