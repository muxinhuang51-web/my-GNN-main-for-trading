"""面板加载与逐日截面数据供给（严格 PIT）。

核心对象 MarketData：
- returns/close/open/suspend/limit 面板
- universe(date, scope): 决策日 t 的可用股票 = PIT 在池 ∧ t-1 前有足够历史 ∧（指数成分，若 scope 限定）
- 所有取数接口带 max_date 断言，杜绝未来信息。
"""

from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config


class LeakageError(RuntimeError):
    pass


class MarketData:
    def __init__(self, scope: str = "all"):
        """scope: all / csi300 / csi500。全部 scope 均剔除北交所（.BJ）：
        30% 涨跌停与极端数据质量问题会污染簇结构与收益统计。"""
        p = config.PANEL_DIR
        self.scope = scope
        self.returns = pd.read_parquet(p / "returns.parquet")
        keep = [c for c in self.returns.columns if not c.endswith(".BJ")]
        self.returns = self.returns[keep]
        self.suspend = pd.read_parquet(p / "suspend_mask.parquet").reindex_like(self.returns).fillna(False)
        self.at_up_limit = pd.read_parquet(p / "at_up_limit.parquet").reindex_like(self.returns).fillna(0.0)
        self.at_down_limit = pd.read_parquet(p / "at_down_limit.parquet").reindex_like(self.returns).fillna(0.0)
        self.universe_pit = pd.read_parquet(p / "universe_pit.parquet").reindex_like(self.returns).fillna(False)
        self.dates = self.returns.index
        self.stocks = list(self.returns.columns)
        self._industry_intervals = pd.read_parquet(p / "sw_industry_intervals.parquet")
        self._index_members = {}
        if scope in ("csi300", "csi500"):
            self._index_members[scope] = pd.read_parquet(p / f"index_members_{scope}.parquet")

    # ---------- 严格 PIT 取数 ----------
    def window_returns(self, t: int, lookback: int) -> pd.DataFrame:
        """[t-lookback, t) 的收益窗口——不含 t 日。"""
        if t < lookback:
            raise LeakageError("窗口越界")
        return self.returns.iloc[t - lookback : t]

    def realized_returns(self, t: int) -> pd.Series:
        """t 日实现收益。只允许结算/评估调用（调用点集中在 backtest.settle）。"""
        return self.returns.iloc[t]

    def tradable_at_decision(self, t: int) -> pd.Series:
        """买入可行性（t-1 收盘决策）：t-1 未停牌且 t-1 未涨停触板。"""
        prev = t - 1
        ok = (~self.suspend.iloc[prev]) & (self.at_up_limit.iloc[prev] < 0.5)
        return ok

    def sellable_at_decision(self, t: int) -> pd.Series:
        """卖出可行性（在 t-1 收盘执行卖出）：t-1 有成交且未跌停锁死。
        停牌或跌停锁死的持仓必须被强制持有进入 t 日（复牌跳空由持有人承担）。"""
        prev = t - 1
        return (~self.suspend.iloc[prev]) & (self.at_down_limit.iloc[prev] < 0.5)

    def universe(self, t: int) -> List[str]:
        """决策日 t 的股票池（只用 <= t-1 的信息）。"""
        date = self.dates[t]
        in_pit = self.universe_pit.iloc[t - 1]
        window = self.window_returns(t, config.LOOKBACK)
        enough_history = window.notna().sum() >= config.FEATURE_MIN_OBS
        mask = in_pit & enough_history
        if self.scope in self._index_members:
            members = self._members_asof(self.scope, self.dates[t - 1])
            mask &= pd.Series(self.returns.columns.isin(members), index=self.returns.columns)
        return list(self.returns.columns[mask])

    def industry_map(self, t: int) -> pd.Series:
        """t-1 日有效的申万一级行业映射。"""
        date = self.dates[t - 1]
        iv = self._industry_intervals
        active = iv[(iv["in_dt"] <= date) & (iv["out_dt"].isna() | (iv["out_dt"] > date))]
        return active.drop_duplicates("ts_code", keep="last").set_index("ts_code")["industry_name"]

    def _members_asof(self, name: str, date: pd.Timestamp) -> set:
        weights = self._index_members[name]
        code_col = "con_code" if "con_code" in weights.columns else "ts_code"
        past = weights[weights["trade_date"] <= date]
        if past.empty:
            return set()
        return set(past.loc[past["trade_date"] == past["trade_date"].max(), code_col])

    def benchmark_return(self, t: int, selected_universe: List[str]) -> float:
        """可投资基准：股票池 ∩ t-1 可买 的等权收益。

        与策略施加完全相同的可交易性过滤，避免涨停延续收益只进基准
        不进组合造成的系统性不对称（对抗审查确认项 #2）。"""
        tradable = self.tradable_at_decision(t)
        investable = [s for s in selected_universe if bool(tradable.get(s, False))]
        vals = self.returns.iloc[t].reindex(investable)
        return float(vals.mean(skipna=True))

    def positions_in_range(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp], min_history: int) -> List[int]:
        idx = []
        for t in range(min_history, len(self.dates)):
            d = self.dates[t]
            if start is not None and d < start:
                continue
            if end is not None and d > end:
                continue
            idx.append(t)
        return idx


def assert_no_test_access(end_date: pd.Timestamp, context: str) -> None:
    """防火墙断言：非 final 流程禁止触碰 TEST 期。"""
    if end_date >= config.TEST_START:
        raise LeakageError(
            f"{context}: 请求的数据截止 {end_date.date()} 已进入 TEST 期 "
            f"({config.TEST_START.date()} 之后)。调参/开发阶段禁止访问。"
        )
