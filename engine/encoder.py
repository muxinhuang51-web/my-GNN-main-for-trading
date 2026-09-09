"""RGCN 编码器的无泄漏训练。

防火墙规则：
- 部署期（回测区间）第一天之前的数据才允许参与训练与选型
- checkpoint 选择用训练区间末尾 10% 作为内部验证集（绝不触碰部署期）
- 每个 (scope, k_e, seed, train_end) 组合独立训练 → 训练图与推理图一致
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import RGCNConv

from . import config
from .data import MarketData, LeakageError
from .features import GraphBuilder

NUM_RELATIONS = 2  # 0=行业, 1=相关性


class StockRGCN(torch.nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = config.HIDDEN_DIM):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, NUM_RELATIONS)
        self.conv2 = RGCNConv(hidden, hidden, NUM_RELATIONS)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, return_embedding=False):
        h = torch.relu(self.conv1(x, edge_index, edge_type))
        emb = self.conv2(h, edge_index, edge_type)
        if return_embedding:
            return emb
        return self.head(emb).squeeze(-1)


def _ic(pred: np.ndarray, real: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(real)
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(pred[mask], real[mask])[0, 1])


def encoder_path(scope: str, k_e: int, seed: int, train_end: pd.Timestamp) -> Path:
    return config.MODELS_DIR / f"rgcn_{scope}_ke{k_e}_seed{seed}_{train_end.strftime('%Y%m%d')}.pt"


def train_encoder(
    market: MarketData,
    k_e: int,
    seed: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    deploy_start: pd.Timestamp,
    device: torch.device,
    epochs: int = config.ENCODER_EPOCHS,
) -> Path:
    """训练一个编码器并落盘；若已存在直接返回。"""
    if train_end >= deploy_start:
        raise LeakageError(
            f"train_end {train_end.date()} >= deploy_start {deploy_start.date()}：训练期侵入部署期"
        )
    out = encoder_path(market.scope, k_e, seed, train_end)
    if out.exists():
        return out
    config.ensure_dirs()
    torch.manual_seed(seed)
    np.random.seed(seed)

    builder = GraphBuilder(market)
    positions = market.positions_in_range(train_start, train_end, config.LOOKBACK + 1)
    if len(positions) < 100:
        raise RuntimeError(f"训练日不足: {len(positions)}")
    split = int(len(positions) * 0.9)
    train_pos, inner_val_pos = positions[:split], positions[split:]

    model = StockRGCN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.ENCODER_LR, weight_decay=1e-5)
    best_ic, best_state, patience = -np.inf, None, 0
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_pos))
        losses = []
        for i in order:
            t = train_pos[i]
            stocks, x, ei, et = builder.day_inputs(t, k_e)
            y = market.realized_returns(t).reindex(stocks).to_numpy(dtype=np.float32)
            y_t = torch.from_numpy(y).to(device)
            mask = torch.isfinite(y_t)
            if mask.sum() < 50:
                continue
            opt.zero_grad()
            pred = model(x.to(device), ei.to(device), et.to(device))
            loss = torch.nn.functional.mse_loss(pred[mask], y_t[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))

        model.eval()
        ics = []
        with torch.no_grad():
            for t in inner_val_pos:
                stocks, x, ei, et = builder.day_inputs(t, k_e)
                pred = model(x.to(device), ei.to(device), et.to(device)).cpu().numpy()
                real = market.realized_returns(t).reindex(stocks).to_numpy(dtype=float)
                ics.append(_ic(pred, real))
        val_ic = float(np.nanmean(ics))
        print(f"[encoder {market.scope} ke={k_e} seed={seed}] epoch {epoch}: loss={np.mean(losses):.5f} innerIC={val_ic:.4f}")
        if val_ic > best_ic:
            best_ic, best_state, patience = val_ic, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= config.ENCODER_PATIENCE:
                break

    torch.save(best_state, out)
    meta = {
        "scope": market.scope, "k_e": k_e, "seed": seed,
        "train_start": str(train_start.date()), "train_end": str(train_end.date()),
        "inner_val_ic": best_ic, "epochs_ran": epoch,
        "train_days": len(train_pos), "inner_val_days": len(inner_val_pos),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[encoder] 保存 {out.name} (innerIC={best_ic:.4f})")
    return out


def load_encoder(path: Path, device: torch.device) -> StockRGCN:
    model = StockRGCN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
