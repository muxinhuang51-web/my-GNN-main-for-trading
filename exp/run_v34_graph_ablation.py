"""v3.4 图必要性消融：同一条 v3.3 带收割管线（band 60-90, h=5, 共识），换表征来源。

Arm A（已有）：RGCN 嵌入（行业边+特征）        -> outputs_v2/v33/band_60_90
Arm B（新）  ：原始 6 特征直接聚类（无编码器）  -> 图与编码器都不要
Arm C（新）  ：无边编码器（同架构同训练，空边集）-> 有编码器、无消息传递
A ≈ B ≈ C ⇒ 图是装饰；A >> C > B ⇒ 图/编码器各有贡献
"""
import json
import numpy as np, pandas as pd, torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.data import MarketData, assert_no_test_access
from engine.features import GraphBuilder, build_features
from engine.encoder import StockRGCN
from exp.run_v32_consensus import Consensus

OUT = config.RUNS_DIR / "v34"
LO, HI, HOLD = 0.60, 0.90, 5


class RawFeatConsensus(Consensus):
    """Arm B：分区与簇特征都来自原始 6 特征（无编码器无图）。"""

    def _clusters(self, ci, ks, t):
        key = (0, ks, t)  # ci 无意义，固定 0
        if key not in self._part:
            stocks = self.m.universe(t)
            x = build_features(self.m.window_returns(t, config.LOOKBACK), stocks)
            labels = KMeans(n_clusters=20, random_state=ks, n_init=1).fit_predict(x)
            ids = sorted(set(labels))
            feats = np.stack([x[labels == c].mean(axis=0) for c in ids])
            if len(self._part) > 3 * 75:
                oldest = min(self._part, key=lambda k: k[2])
                if oldest != key:
                    self._part.pop(oldest)
            self._part[key] = (stocks, labels, feats, ids)
        return self._part[key]


class EmptyGraphBuilder(GraphBuilder):
    """空边集：特征照常，消息传递失效（RGCNConv 退化为自环线性层）。"""

    def day_inputs(self, t, k_e):
        stocks, x, ei, et = super().day_inputs(t, 0)
        empty = torch.empty((2, 0), dtype=torch.long)
        return stocks, x, empty, torch.empty((0,), dtype=torch.long)


def train_noedge_encoder(market, seed, device):
    """与 train_encoder 同协议（同窗口/早停/内部验证），只是边为空。"""
    out = config.MODELS_DIR / f"rgcn_noedge_all_seed{seed}_20211231.pt"
    if out.exists():
        return out
    torch.manual_seed(seed); np.random.seed(seed)
    builder = EmptyGraphBuilder(market)
    positions = market.positions_in_range(pd.Timestamp("2016-01-01"), config.TRAIN_END, config.LOOKBACK + 1)
    split = int(len(positions) * 0.9)
    train_pos, val_pos = positions[:split], positions[split:]
    model = StockRGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.ENCODER_LR, weight_decay=1e-5)
    best, best_state, patience = -np.inf, None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, config.ENCODER_EPOCHS + 1):
        model.train()
        for i in rng.permutation(len(train_pos)):
            t = train_pos[i]
            stocks, x, ei, et = builder.day_inputs(t, 0)
            y = market.realized_returns(t).reindex(stocks).to_numpy(dtype=np.float32)
            y_t = torch.from_numpy(y).to(device); mask = torch.isfinite(y_t)
            if mask.sum() < 50: continue
            opt.zero_grad()
            pred = model(x.to(device), ei.to(device), et.to(device))
            loss = torch.nn.functional.mse_loss(pred[mask], y_t[mask])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); ics = []
        with torch.no_grad():
            for t in val_pos:
                stocks, x, ei, et = builder.day_inputs(t, 0)
                pred = model(x.to(device), ei.to(device), et.to(device)).cpu().numpy()
                real = market.realized_returns(t).reindex(stocks).to_numpy(dtype=float)
                m = np.isfinite(pred) & np.isfinite(real)
                if m.sum() > 10: ics.append(np.corrcoef(pred[m], real[m])[0, 1])
        vic = float(np.nanmean(ics))
        print(f"[noedge seed={seed}] epoch {epoch}: innerIC={vic:.4f}", flush=True)
        if vic > best: best, best_state, patience = vic, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= config.ENCODER_PATIENCE: break
    torch.save(best_state, out)
    return out


class NoEdgeCache:
    """EmbeddingCache 的空边版。"""

    def __init__(self, market, model_path, device):
        from engine.encoder import load_encoder
        self.market, self.device = market, device
        self.builder = EmptyGraphBuilder(market)
        self.model = load_encoder(model_path, device)
        import hashlib
        tag = hashlib.md5(model_path.name.encode()).hexdigest()[:10]
        self.dir = config.CACHE_DIR / f"emb_noedge_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k_e = 0

    def get(self, t):
        import os
        key = self.market.dates[t].strftime("%Y%m%d")
        path = self.dir / f"{key}.npz"
        if path.exists():
            try:
                blob = np.load(path, allow_pickle=True)
                return list(blob["stocks"]), blob["emb"]
            except Exception:
                pass
        stocks, x, ei, et = self.builder.day_inputs(t, 0)
        with torch.no_grad():
            emb = self.model(x.to(self.device), ei.to(self.device), et.to(self.device), return_embedding=True)
        emb = emb.cpu().numpy().astype(np.float32)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, stocks=np.array(stocks), emb=emb)
        os.replace(tmp, path)
        return stocks, emb


class BandC(Consensus):
    def __init__(self, market, caches, km_seeds):
        super().__init__(market, caches, km_seeds)

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        parts = []
        for ci in range(len(self.caches)):
            for ks in self.km_seeds:
                sc = self._stock_scores(ci, ks, t)
                if sc is not None:
                    parts.append(sc)
        if not parts:
            return self._held
        consensus = pd.concat(parts, axis=1).mean(axis=1)
        tr = self.m.tradable_at_decision(t)
        consensus = consensus[[s for s in consensus.index if bool(tr.get(s, False))]].dropna()
        ranks = consensus.rank(pct=True)
        self._held = list(ranks[(ranks >= LO) & (ranks <= HI)].index)
        self._cd = HOLD - 1
        return self._held


def main():
    market = MarketData("all")
    dev = torch.device("cpu")
    assert_no_test_access(config.VALID_END, "v34")

    # Arm B：原始特征共识带（3 KMeans 种子）
    name = "armB_rawfeat_band60_90"
    if not (OUT / name / "run_summary.json").exists():
        strat = RawFeatConsensus(market, [None], km_seeds=(42, 7, 123))
        strat.caches = [None]  # 单一"表征"，共识只跨 KMeans 种子
        # RawFeatConsensus._clusters 忽略 ci；_stock_scores 需要 caches 长度 1
        print(f"[run] {name}", flush=True)
        run_backtest(market, BandC.select.__get__(strat), config.VALID_START, config.VALID_END,
                     OUT / name, {"arm": "B_rawfeat"})

    # Arm C：无边编码器 x3 共识带
    paths = [train_noedge_encoder(market, s, dev) for s in (42, 7, 123)]
    caches = [NoEdgeCache(market, p, dev) for p in paths]
    name = "armC_noedge_band60_90"
    if not (OUT / name / "run_summary.json").exists():
        strat = BandC(market, caches, km_seeds=(42, 7, 123))
        print(f"[run] {name}", flush=True)
        run_backtest(market, strat.select, config.VALID_START, config.VALID_END,
                     OUT / name, {"arm": "C_noedge"})

    print("\n=== 图必要性消融（band 60-90, h=5, 全市场验证期）===")
    ref = json.loads((config.RUNS_DIR / "v33" / "band_60_90" / "run_summary.json").read_text())
    rows = [{"arm": "A_RGCN(行业边)", "net": ref["metrics"]["sharpe_excess_net"]}]
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"arm": f.parent.name, "gross": sharpe_annualized(ge),
                     "nw_t": newey_west_tstat(ge), "net": p["metrics"]["sharpe_excess_net"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
