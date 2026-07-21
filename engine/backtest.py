"""回测引擎 v2：成本、换手、t-1 可交易性、统一指标。

修复旧版全部审计问题：
- 决策只用 <= t-1 信息（可交易性来自 t-1 停牌/涨停，不看 t 日 NaN）
- 停牌持仓当日计 0 收益（冻结），而非从分母剔除
- 成本按换手计（买 BUY_COST / 卖 SELL_COST），逐日记录换手率
- 无效日计平收益（0 超额），不从收益序列删除
- 全部策略共用同一 Sharpe 实现（超额收益制）
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config
from .data import MarketData


def sharpe_annualized(excess_daily: np.ndarray) -> float:
    arr = np.asarray(excess_daily, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / arr.std(ddof=1) * np.sqrt(252))


def newey_west_tstat(series: np.ndarray, lags: int = 5) -> float:
    """日频序列均值的 Newey-West t 值。"""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    mean = x.mean()
    e = x - mean
    gamma0 = (e @ e) / n
    variance = gamma0
    for lag in range(1, lags + 1):
        cov = (e[lag:] @ e[:-lag]) / n
        variance += 2 * (1 - lag / (lags + 1)) * cov
    return float(mean / np.sqrt(variance / n))


def block_bootstrap_sharpe_ci(excess: np.ndarray, block: int = 10, n_boot: int = 2000, seed: int = 0):
    x = np.asarray(excess, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 40:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(x) / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, len(x) - block + 1, n_blocks)
        sample = np.concatenate([x[s : s + block] for s in starts])[: len(x)]
        stats.append(sharpe_annualized(sample))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def max_drawdown(returns_net: np.ndarray) -> float:
    arr = np.asarray(returns_net, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    wealth = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(wealth)
    return float(((wealth - peak) / peak).min())


def run_backtest(
    market: MarketData,
    select_fn: Callable[[int], List[str]],
    start: pd.Timestamp,
    end: Optional[pd.Timestamp],
    out_dir: Path,
    run_params: Dict,
    ic_fn: Optional[Callable[[int], float]] = None,
    min_history: Optional[int] = None,
) -> Dict:
    """通用回测循环。

    select_fn(t) -> 持仓股票列表（内部只允许使用 <= t-1 的信息；
    可交易性过滤由本函数统一执行，策略无须也无法看到 t 日数据）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    min_history = min_history if min_history is not None else config.LOOKBACK + 25
    positions = market.positions_in_range(start, end, min_history)
    records = []
    prev_holdings: Dict[str, float] = {}  # 持仓权重（含现金余量：1 - sum(weights)）
    daily_ics = []

    for t in positions:
        date = market.dates[t]
        t0 = time.perf_counter()
        # 1. 强制持仓：t-1 无法卖出（停牌/跌停锁死）的持仓必须带入 t 日，
        #    复牌/解锁后的跳空收益由持有人承担（对抗审查确认项 #1）。
        sellable = market.sellable_at_decision(t)
        forced = {s: w for s, w in prev_holdings.items() if not bool(sellable.get(s, True))}
        forced_weight = sum(forced.values())

        # 2. 策略选股（t-1 信息）+ 统一买入可行性过滤（所有策略族一致）
        raw_selection = select_fn(t)
        tradable = market.tradable_at_decision(t)
        buyable = [s for s in raw_selection
                   if s not in forced and (bool(tradable.get(s, False)) or s in prev_holdings)]

        free_weight = max(0.0, 1.0 - forced_weight)
        if len(buyable) < 10:
            targets = dict(forced)  # 无法建仓：持有强制仓位，其余现金
        else:
            per_stock = free_weight / len(buyable)
            targets = {**forced, **{s: per_stock for s in buyable}}

        # 3. 换手与成本（相对昨日持仓权重；强制仓位不产生交易）
        names = set(targets) | set(prev_holdings)
        buys = sum(max(0.0, targets.get(s, 0.0) - prev_holdings.get(s, 0.0)) for s in names)
        sells = sum(max(0.0, prev_holdings.get(s, 0.0) - targets.get(s, 0.0)) for s in names)
        cost = buys * config.BUY_COST + sells * config.SELL_COST
        turnover = (buys + sells) / 2

        # 4. 结算：组合加权收益；停牌股当日冻结计 0（其跳空在复牌日实现）；现金计 0。
        realized = market.realized_returns(t)
        rets = {s: (float(realized.get(s)) if np.isfinite(realized.get(s, np.nan)) else 0.0)
                for s in targets}
        gross = float(sum(targets[s] * rets[s] for s in targets))
        net = gross - cost
        bench = market.benchmark_return(t, market.universe(t))
        ic = float(ic_fn(t)) if ic_fn is not None else np.nan
        daily_ics.append(ic)

        # 纸面对照：同一批目标股票的简单等权均值（NaN 剔除、无冻结、无漂移）——
        # 与引擎 gross 的差值即引擎机制（冻结/强制持仓/漂移）的逐日归因
        paper = float(np.nanmean([realized.get(s, np.nan) for s in targets])) if targets else 0.0
        records.append({
            "date": str(date.date()), "n_stocks": len(targets), "gross": gross, "net": net,
            "bench": bench, "excess_net": net - bench, "turnover": turnover, "ic": ic,
            "forced_weight": forced_weight, "paper_gross": paper,
            "frozen_n": int(sum(1 for s in targets if not np.isfinite(realized.get(s, np.nan)))),
        })
        # 5. 权重按当日收益漂移，作为明日的 prev_holdings
        total_growth = 1.0 + gross
        prev_holdings = {s: targets[s] * (1.0 + rets[s]) / total_growth for s in targets}
        if len(records) % 50 == 0:
            print(f"[bt] {date.date()} ({len(records)}/{len(positions)}) net={net:+.4f} to={turnover:.2f} {time.perf_counter()-t0:.1f}s")

    frame = pd.DataFrame(records)
    frame.to_csv(out_dir / "daily.csv", index=False)

    excess = frame["excess_net"].to_numpy()
    net = frame["net"].to_numpy()
    ics = np.array([i for i in daily_ics if np.isfinite(i)])
    years = max(len(frame) / 252.0, 1e-9)
    metrics = {
        "days": int(len(frame)),
        "ann_return_net": float(np.prod(1 + np.where(np.isfinite(net), net, 0.0)) ** (1 / years) - 1),
        "sharpe_excess_net": sharpe_annualized(excess),
        "sharpe_net": sharpe_annualized(net),
        "max_drawdown_net": max_drawdown(net),
        "mean_daily_turnover": float(np.nanmean(frame["turnover"])),
        "win_rate_excess": float((excess > 0).mean()),
        "mean_ic": float(ics.mean()) if len(ics) else None,
        "ic_ir": float(ics.mean() / ics.std(ddof=1) * np.sqrt(252)) if len(ics) > 2 else None,
        "nw_tstat_excess": newey_west_tstat(excess),
        "sharpe_ci95": block_bootstrap_sharpe_ci(excess),
    }
    payload = {"params": run_params, "metrics": metrics,
               "period": {"start": str(start.date()), "end": str(end.date()) if end is not None else None},
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return metrics
