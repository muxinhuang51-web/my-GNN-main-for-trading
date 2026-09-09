"""v7 架构轴：GRU / Transformer / MLP 打分器，同一两行修正（可行性加权 + 带收割）。

协议：序列输入 = 过去 60 日的日收益序列（每步 1 维），2016-2021 预训练（可行性加权 MSE），
valid 期每 60 交易日滚动重训一次（trailing 2 年窗），冻结间隔内打分。{top, band} 双收割。
预测（写死待验）：band 口径各架构与 Ridge 打平（不变量）；架构差异 << 表征/标签/收割差异。
"""
import json
import numpy as np, pandas as pd, torch
import torch.nn as nn
from engine import config
from engine.backtest import run_backtest, sharpe_annualized, newey_west_tstat
from engine.data import MarketData, assert_no_test_access

OUT = config.RUNS_DIR / "v7_arch"
HOLD, TOP, SEQ = 5, 100, 60
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.head(h[-1]).squeeze(-1)


class TransNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 32)
        self.pos = nn.Parameter(torch.randn(SEQ, 32) * 0.02)
        layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1)).squeeze(-1)


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Flatten(), nn.Linear(SEQ, 64), nn.ReLU(),
                               nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def make_net(arch):
    return {"gru": GRUNet, "trans": TransNet, "mlp": MLPNet}[arch]()


class SeqArch:
    def __init__(self, market, arch, harvest):
        self.m, self.arch, self.harvest = market, arch, harvest
        self._held, self._cd = [], 0
        self._net, self._last_fit = None, -10**9

    def _seq_day(self, s):
        win = self.m.window_returns(s, SEQ)
        stocks = self.m.universe(s)
        X = win[stocks].to_numpy(float).T          # N x SEQ
        X = np.nan_to_num(X, nan=0.0)
        sd = X.std(axis=1, keepdims=True) + 1e-6   # 每股标准化，防尺度爆炸
        return stocks, (X / sd)[:, :, None].astype(np.float32)

    def _fit(self, t):
        torch.manual_seed(42)
        net = make_net(self.arch).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        window = 480                                # trailing ~2 年
        days = list(range(max(SEQ + 2, t - window), t, 2))  # 隔日采样减负
        for epoch in range(2):
            np.random.default_rng(epoch).shuffle(days)
            for s in days:
                stocks, X = self._seq_day(s)
                r = self.m.realized_returns(s).reindex(stocks).to_numpy(float)
                a = self.m.tradable_at_decision(s).reindex(stocks).fillna(False).to_numpy(bool)
                ok = np.isfinite(r) & a             # 修正①：可行性加权
                if ok.sum() < 100:
                    continue
                xb = torch.from_numpy(X[ok]).to(DEV)
                yb = torch.from_numpy(r[ok].astype(np.float32)).to(DEV)
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(xb), yb)
                loss.backward(); opt.step()
        net.eval()
        self._net = net
        self._last_fit = t

    def select(self, t):
        if self._cd > 0 and self._held:
            self._cd -= 1
            return self._held
        if self._net is None or t - self._last_fit >= 60:
            print(f"[fit] {self.arch} @t={t}", flush=True)
            self._fit(t)
        stocks, X = self._seq_day(t)
        with torch.no_grad():
            score = pd.Series(self._net(torch.from_numpy(X).to(DEV)).cpu().numpy(), index=stocks)
        tr = self.m.tradable_at_decision(t)
        score = score[[x for x in score.index if bool(tr.get(x, False))]].dropna()
        if self.harvest == "top":
            self._held = list(score.sort_values(ascending=False).index[:TOP])
        else:
            pct = score.rank(pct=True)
            self._held = list(pct[(pct >= 0.60) & (pct <= 0.90)].index)
        self._cd = HOLD - 1
        return self._held


def main():
    market = MarketData("all")
    assert_no_test_access(config.VALID_END, "v7")
    for arch in ("gru", "trans", "mlp"):
        for harvest in ("band", "top"):
            name = f"v7_{arch}_{harvest}"
            out = OUT / name
            if (out / "run_summary.json").exists():
                print(f"[skip] {name}", flush=True); continue
            strat = SeqArch(market, arch, harvest)
            print(f"[run] {name}", flush=True)
            run_backtest(market, strat.select, config.VALID_START, config.VALID_END, out,
                         {"arch": arch, "harvest": harvest})
    rows = []
    for f in sorted(OUT.glob("*/run_summary.json")):
        p = json.loads(f.read_text()); d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, "gross_shp": sharpe_annualized(ge),
                     "nw_t": newey_west_tstat(ge), "net_shp": p["metrics"]["sharpe_excess_net"]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
