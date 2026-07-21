"""从 raw 数据构建回测面板：前复权收益、可交易性掩码、PIT 股票池/行业/成分。"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import config
from .api import load_stage, update_manifest


def _pivot(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="trade_date", columns="ts_code", values=value_col, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def build_price_panels() -> Dict[str, pd.DataFrame]:
    """前复权收盘/开盘价矩阵与收益矩阵。

    约定：
    - 后复权价 hfq = price * adj_factor（避免前复权随最新因子漂移，回测用后复权等价）
    - 收益 r_t = hfq_close_t / hfq_close_{t'} - 1，t' 为上一有交易记录日（停牌跨期收益记在复牌日）
    - 停牌/未上市日为 NaN，绝不填 0
    """
    daily = load_stage(config.RAW_DIR / "daily")
    adj = load_stage(config.RAW_DIR / "adj_factor")
    if daily.empty or adj.empty:
        raise RuntimeError("daily 或 adj_factor 数据为空，请先完成拉取")

    merged = daily.merge(adj, on=["ts_code", "trade_date"], how="left")
    missing_adj = merged["adj_factor"].isna().mean()
    if missing_adj > 0.001:
        print(f"[警告] {missing_adj:.2%} 行缺复权因子，这些行使用前值填充")
    merged = merged.sort_values(["ts_code", "trade_date"])
    merged["adj_factor"] = merged.groupby("ts_code")["adj_factor"].ffill()
    merged["hfq_close"] = merged["close"] * merged["adj_factor"]
    merged["hfq_open"] = merged["open"] * merged["adj_factor"]

    close_wide = _pivot(merged, "hfq_close")
    open_wide = _pivot(merged, "hfq_open")

    # 收益基于各股上一有效收盘（ffill 后 pct_change），停牌日本身保持 NaN
    close_ffill = close_wide.ffill()
    returns = close_ffill.pct_change(fill_method=None)
    returns = returns.where(close_wide.notna())  # 停牌日无收益
    first_valid = close_wide.notna().cumsum() <= 1
    returns = returns.where(~(first_valid & close_wide.notna()))  # 上市首个观测日无前值

    config.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    close_wide.to_parquet(config.PANEL_DIR / "hfq_close.parquet")
    open_wide.to_parquet(config.PANEL_DIR / "hfq_open.parquet")
    returns.to_parquet(config.PANEL_DIR / "returns.parquet")
    update_manifest(
        "panel_prices",
        {"dates": len(returns), "stocks": returns.shape[1],
         "start": str(returns.index.min().date()), "end": str(returns.index.max().date())},
    )
    print(f"[panel] returns: {returns.shape[0]} 日 x {returns.shape[1]} 股")
    return {"returns": returns, "close": close_wide, "open": open_wide}


def build_tradability_panels() -> None:
    """可交易性：停牌掩码 + 涨跌停触板标志（决策时只允许用 t-1 及之前的列）。"""
    daily = load_stage(config.RAW_DIR / "daily")
    close = pd.read_parquet(config.PANEL_DIR / "hfq_close.parquet")

    # 停牌：有交易记录 = 可交易。daily 缺行即停牌（与 suspend_d 交叉验证，若有）
    traded = _pivot(daily.assign(traded=1.0), "traded").reindex(index=close.index, columns=close.columns)
    suspend_mask = traded.isna() & close.ffill().notna()  # 上市后但无交易 = 停牌
    suspend_mask.to_parquet(config.PANEL_DIR / "suspend_mask.parquet")

    limit_dir = config.RAW_DIR / "stk_limit"
    if any(limit_dir.glob("*.parquet")):
        limit = load_stage(limit_dir)
        merged = daily.merge(limit, on=["ts_code", "trade_date"], how="left")
        merged["at_up_limit"] = (merged["close"] >= merged["up_limit"] - 1e-6).astype(float)
        merged["at_down_limit"] = (merged["close"] <= merged["down_limit"] + 1e-6).astype(float)
        up_wide = _pivot(merged, "at_up_limit").reindex(index=close.index, columns=close.columns)
        down_wide = _pivot(merged, "at_down_limit").reindex(index=close.index, columns=close.columns)
    else:
        # 无权限近似：按涨跌幅阈值判断触板（主板 10%、创业板/科创板 20%、ST 5%——用 9.7%/19.5%/4.7% 容差）
        print("[panel] stk_limit 缺失，按涨跌幅阈值近似触板")
        pct = daily.assign(pct=(daily["close"] / daily["pre_close"] - 1))
        board_limit = pct["ts_code"].map(_board_limit_pct)
        pct["at_up_limit"] = (pct["pct"] >= board_limit * 0.97).astype(float)
        pct["at_down_limit"] = (pct["pct"] <= -board_limit * 0.97).astype(float)
        up_wide = _pivot(pct, "at_up_limit").reindex(index=close.index, columns=close.columns)
        down_wide = _pivot(pct, "at_down_limit").reindex(index=close.index, columns=close.columns)
    up_wide.fillna(0.0).to_parquet(config.PANEL_DIR / "at_up_limit.parquet")
    down_wide.fillna(0.0).to_parquet(config.PANEL_DIR / "at_down_limit.parquet")
    print("[panel] 可交易性面板完成")


def _board_limit_pct(ts_code: str) -> float:
    code = ts_code.split(".")[0]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def build_universe_panels() -> None:
    """PIT 股票池：上市满 N 日且未退市；PIT 行业映射；指数成分（月度快照展开到日）。"""
    basic = pd.read_parquet(config.RAW_DIR / "stock_basic.parquet")
    returns = pd.read_parquet(config.PANEL_DIR / "returns.parquet")
    dates = returns.index

    basic = basic.copy()
    basic["list_dt"] = pd.to_datetime(basic["list_date"], errors="coerce")
    basic["delist_dt"] = pd.to_datetime(basic["delist_date"], errors="coerce")
    listable = {}
    for _, row in basic.iterrows():
        if row["ts_code"] not in returns.columns or pd.isna(row["list_dt"]):
            continue
        start = row["list_dt"] + pd.Timedelta(days=180)  # 上市满 180 天才入池，避开新股期
        end = row["delist_dt"] if pd.notna(row["delist_dt"]) else None
        mask = (dates >= start) & ((dates < end) if end is not None else True)
        listable[row["ts_code"]] = mask
    universe = pd.DataFrame(listable, index=dates).reindex(columns=returns.columns).fillna(False)
    universe.to_parquet(config.PANEL_DIR / "universe_pit.parquet")
    print(f"[panel] PIT 股票池: 平均每日 {universe.sum(axis=1).mean():.0f} 只")

    # PIT 行业：sw_member 的 in_date/out_date 区间判定
    member = pd.read_parquet(config.RAW_DIR / "sw_member.parquet")
    member = member.rename(columns={"con_code": "ts_code"})
    member["in_dt"] = pd.to_datetime(member["in_date"], errors="coerce")
    member["out_dt"] = pd.to_datetime(member["out_date"], errors="coerce")
    member[["ts_code", "industry_name", "in_dt", "out_dt"]].to_parquet(
        config.PANEL_DIR / "sw_industry_intervals.parquet"
    )

    # 指数成分：月度权重表 → 各月末快照，engine 端按"决策日之前最近一期快照"查询
    for name in config.INDEX_CODES:
        path = config.RAW_DIR / f"index_weight_{name}.parquet"
        if path.exists():
            weights = pd.read_parquet(path)
            weights["trade_date"] = pd.to_datetime(weights["trade_date"])
            weights.to_parquet(config.PANEL_DIR / f"index_members_{name}.parquet")
    update_manifest("panel_universe", {"avg_universe": float(universe.sum(axis=1).mean())})
    print("[panel] 股票池/行业/成分面板完成")


def industry_map_asof(date: pd.Timestamp) -> pd.Series:
    """给定日期的 PIT 行业映射（ts_code -> industry_name）。"""
    intervals = pd.read_parquet(config.PANEL_DIR / "sw_industry_intervals.parquet")
    active = intervals[
        (intervals["in_dt"] <= date)
        & (intervals["out_dt"].isna() | (intervals["out_dt"] > date))
    ]
    return active.drop_duplicates("ts_code", keep="last").set_index("ts_code")["industry_name"]


def index_members_asof(name: str, date: pd.Timestamp) -> list:
    """给定日期的指数成分（用决策日之前最近一期月度快照）。"""
    weights = pd.read_parquet(config.PANEL_DIR / f"index_members_{name}.parquet")
    past = weights[weights["trade_date"] <= date]
    if past.empty:
        return []
    latest = past["trade_date"].max()
    code_col = "con_code" if "con_code" in weights.columns else "ts_code"
    return past.loc[past["trade_date"] == latest, code_col].tolist()
