"""诊断：全市场簇级预测为何 IC 为正但选簇组合毛超额为负。

对 (all, k=20, ke=0, seed42) 逐日记录每个簇：预测分、全体成员实现收益、
仅可交易成员实现收益，以及 top/bottom 选簇组合的可交易收益。
"""
import numpy as np, pandas as pd, torch
from engine import config
from engine.data import MarketData, assert_no_test_access
from engine.encoder import encoder_path
from engine.cluster_strategy import EmbeddingCache, ClusterRotation

assert_no_test_access(config.VALID_END, "diag")
market = MarketData("all")
dev = torch.device("cpu")
cache = EmbeddingCache(market, encoder_path("all", 0, 42, config.TRAIN_END), 0, dev)
strat = ClusterRotation(market, cache, cluster_count=20, seed=42, device=dev)

positions = market.positions_in_range(config.VALID_START, config.VALID_END, config.LOOKBACK + 25)
rows = []
for n, t in enumerate(positions):
    sel = strat.select(t)          # 触发训练与打分
    if strat._last_scores is None:
        continue
    scores, ids, stocks, labels = strat._last_scores
    realized = market.realized_returns(t).reindex(stocks).to_numpy(dtype=float)
    tradable = market.tradable_at_decision(t).reindex(stocks).fillna(False).to_numpy(dtype=bool)
    order = np.argsort(-scores)
    for rank, pos in enumerate(order):
        member = labels == ids[pos]
        r_all = realized[member]; r_all = r_all[np.isfinite(r_all)]
        r_trd = realized[member & tradable]; r_trd = r_trd[np.isfinite(r_trd)]
        rows.append({"t": t, "rank": rank, "score": float(scores[pos]),
                     "ret_all": r_all.mean() if len(r_all) >= 5 else np.nan,
                     "ret_tradable": r_trd.mean() if len(r_trd) >= 5 else np.nan,
                     "n_members": int(member.sum()), "n_tradable": int((member & tradable).sum())})
    if n % 100 == 0:
        print(f"{n}/{len(positions)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("outputs_v2/diag_unmonetizable.csv", index=False)

def daily_ic(sub, col):
    ics = []
    for _, g in sub.groupby("t"):
        m = np.isfinite(g["score"]) & np.isfinite(g[col])
        if m.sum() >= 3:
            ics.append(np.corrcoef(g.loc[m, "score"], g.loc[m, col])[0, 1])
    return np.nanmean(ics)

print("\n=== IC 对比 ===")
print(f"IC(全体成员实现收益):    {daily_ic(df, 'ret_all'):+.4f}")
print(f"IC(仅可交易成员):        {daily_ic(df, 'ret_tradable'):+.4f}")

print("\n=== 按预测排名的实现收益（bp/日，跨日均值）===")
by_rank = df.groupby("rank")[ ["ret_all", "ret_tradable"] ].mean() * 1e4
print(by_rank.head(5).round(1).to_string())
print("  ...")
print(by_rank.tail(5).round(1).to_string())

top3 = df[df["rank"] < 3].groupby("t")["ret_tradable"].mean()
bot3 = df[df["rank"] >= 17].groupby("t")["ret_tradable"].mean()
print(f"\ntop3 簇可交易日均: {top3.mean()*1e4:+.1f} bp   bottom3: {bot3.mean()*1e4:+.1f} bp")
print(f"涨停锁定占比: top3 簇成员不可交易比例 = "
      f"{1 - df[df['rank']<3]['n_tradable'].sum()/df[df['rank']<3]['n_members'].sum():.1%}，"
      f"全体簇 = {1 - df['n_tradable'].sum()/df['n_members'].sum():.1%}")
