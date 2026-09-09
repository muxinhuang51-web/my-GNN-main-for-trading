"""引擎单元测试：指标正确性、成本/换手计算、泄漏防火墙。不依赖真实数据。"""

import numpy as np
import pandas as pd
import pytest

from engine.backtest import max_drawdown, newey_west_tstat, sharpe_annualized


def test_sharpe_known_value():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 100000)
    s = sharpe_annualized(x)
    assert abs(s - 0.001 / 0.01 * np.sqrt(252)) < 0.05


def test_sharpe_handles_nan_and_degenerate():
    assert np.isnan(sharpe_annualized(np.array([0.01])))
    assert np.isnan(sharpe_annualized(np.array([0.01, 0.01, 0.01])))
    s = sharpe_annualized(np.array([0.01, np.nan, -0.01, 0.02, -0.005]))
    assert np.isfinite(s)


def test_max_drawdown():
    assert max_drawdown(np.array([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_newey_west_positive_for_positive_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.002, 0.01, 500)
    assert newey_west_tstat(x) > 2


def test_firewall_blocks_test_period():
    from engine.config import TEST_START
    from engine.data import LeakageError, assert_no_test_access

    with pytest.raises(LeakageError):
        assert_no_test_access(TEST_START + pd.Timedelta(days=1), "unit-test")
    assert_no_test_access(TEST_START - pd.Timedelta(days=1), "unit-test")


def test_encoder_firewall():
    from engine.data import LeakageError
    from engine.encoder import train_encoder

    class FakeMarket:
        scope = "fake"

    with pytest.raises(LeakageError):
        train_encoder(
            FakeMarket(), 0, 42,
            pd.Timestamp("2016-01-01"), pd.Timestamp("2022-06-30"),
            deploy_start=pd.Timestamp("2022-01-01"), device=None,
        )


def test_backtest_costs_and_turnover_on_synthetic_market():
    """两只股票轮换持仓：验证换手率与成本扣减的精确数值。"""
    from engine import config as engine_config
    from engine.backtest import run_backtest

    dates = pd.bdate_range("2022-01-03", periods=90)
    stocks = ["A", "B", "C", "D"]
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 4)), index=dates, columns=stocks)

    class SyntheticMarket:
        scope = "synthetic"

        def __init__(self):
            self.returns = returns
            self.dates = dates

        def positions_in_range(self, start, end, min_history):
            return list(range(min_history, len(dates)))

        def universe(self, t):
            return stocks

        def window_returns(self, t, lookback):
            return returns.iloc[max(0, t - lookback) : t]

        def realized_returns(self, t):
            return returns.iloc[t]

        def tradable_at_decision(self, t):
            return pd.Series(True, index=stocks)

        def benchmark_return(self, t, uni):
            return float(returns.iloc[t].mean())

    market = SyntheticMarket()
    # 策略：奇数日持 A/B，偶数日持 C/D → 每日全换仓，换手率应为 1.0
    def select_fn(t):
        return ["A", "B"] if t % 2 else ["C", "D"]

    # 引擎要求 >=10 只才建仓；放宽为持 10 只中的固定 2 组——改用 12 只股票版本
    stocks12 = [f"S{i}" for i in range(12)]
    returns12 = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 12)), index=dates, columns=stocks12)
    market.returns = returns12
    market.universe = lambda t: stocks12
    market.window_returns = lambda t, lookback: returns12.iloc[max(0, t - lookback) : t]
    market.realized_returns = lambda t: returns12.iloc[t]
    market.tradable_at_decision = lambda t: pd.Series(True, index=stocks12)
    market.sellable_at_decision = lambda t: pd.Series(True, index=stocks12)
    market.benchmark_return = lambda t, uni: float(returns12.iloc[t].mean())

    group1, group2 = stocks12[:10], stocks12[2:]  # 重叠 8 只 → 每日换手 2/10 = 0.2

    def select_overlap(t):
        return group1 if t % 2 else group2

    out = engine_config.RUNS_DIR / "_unit_test"
    metrics = run_backtest(
        market, select_overlap, dates[10], dates[-1], out, {"unit": True}, min_history=5
    )
    daily = pd.read_csv(out / "daily.csv")
    # 第一天全买入（换手 0.5 单边计），此后每日换手 = (0.2 买 + 0.2 卖)/2 = 0.2
    later = daily["turnover"].iloc[1:]
    assert np.allclose(later, 0.2, atol=0.02), later.describe()  # 权重随收益漂移的容差
    expected_cost = 0.2 * engine_config.BUY_COST + 0.2 * engine_config.SELL_COST
    diff = daily["gross"].iloc[1:] - daily["net"].iloc[1:]
    assert np.allclose(diff, expected_cost, atol=5e-5)
    # 指标存在且有限
    assert np.isfinite(metrics["sharpe_excess_net"])
    assert metrics["days"] == len(daily)


def test_corr_neighbor_table_matches_pandas():
    """向量化配对相关性 vs pandas.corr 逐对结果一致（含缺失值）。"""
    from engine.features import corr_neighbor_table

    rng = np.random.default_rng(3)
    n_days, n_stocks = 60, 30
    data = rng.normal(0, 0.02, (n_days, n_stocks))
    mask = rng.random((n_days, n_stocks)) < 0.15
    data[mask] = np.nan
    frame = pd.DataFrame(data, columns=[f"S{i}" for i in range(n_stocks)])

    table = corr_neighbor_table(frame, list(frame.columns), min_overlap=20)
    ref = frame.corr(min_periods=20)

    for i in range(n_stocks):
        top = [j for j in table[i] if j >= 0][:5]
        for j in top:
            expected = ref.iloc[i, j]
            row = frame.iloc[:, [i, j]].dropna()
            got_corr = np.corrcoef(row.iloc[:, 0], row.iloc[:, 1])[0, 1]
            assert abs(got_corr - expected) < 1e-6
        # 排序正确性：表内第一个邻居的 |corr| 不低于任何未入表邻居
        if top:
            in_table = set(int(j) for j in table[i] if j >= 0)
            best_in = abs(ref.iloc[i, top[0]])
            others = [abs(ref.iloc[i, j]) for j in range(n_stocks)
                      if j != i and j not in in_table and np.isfinite(ref.iloc[i, j])]
            if others:
                assert best_in >= max(others) - 1e-6


def _make_market(returns, suspend=None, up=None, down=None):
    """构造带可交易性面板的合成市场。"""
    dates, stocks = returns.index, list(returns.columns)
    z = pd.DataFrame(0.0, index=dates, columns=stocks)
    f = pd.DataFrame(False, index=dates, columns=stocks)

    class M:
        scope = "synthetic"

        def __init__(self):
            self.returns = returns
            self.dates = dates
            self.suspend = suspend if suspend is not None else f.copy()
            self.at_up_limit = up if up is not None else z.copy()
            self.at_down_limit = down if down is not None else z.copy()

        def positions_in_range(self, start, end, min_history):
            return [t for t in range(min_history, len(dates))
                    if (start is None or dates[t] >= start) and (end is None or dates[t] <= end)]

        def universe(self, t):
            return stocks

        def window_returns(self, t, lookback):
            return returns.iloc[max(0, t - lookback):t]

        def realized_returns(self, t):
            return returns.iloc[t]

        def tradable_at_decision(self, t):
            return (~self.suspend.iloc[t - 1]) & (self.at_up_limit.iloc[t - 1] < 0.5)

        def sellable_at_decision(self, t):
            return (~self.suspend.iloc[t - 1]) & (self.at_down_limit.iloc[t - 1] < 0.5)

        def benchmark_return(self, t, uni):
            tr = self.tradable_at_decision(t)
            inv = [s for s in uni if bool(tr.get(s, False))]
            return float(returns.iloc[t].reindex(inv).mean(skipna=True))

    return M()


def test_forced_hold_realizes_resume_gap():
    """停牌持仓不可幻影退出：复牌跳空 -60% 必须由组合承担。"""
    from engine import config as engine_config
    from engine.backtest import run_backtest

    dates = pd.bdate_range("2022-01-03", periods=40)
    stocks = [f"S{i}" for i in range(12)]
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0.0, 0.001, (len(dates), 12)), index=dates, columns=stocks)
    suspend = pd.DataFrame(False, index=dates, columns=stocks)
    # S0: 第 15-24 天停牌（收益 NaN），第 25 天复牌 -60%
    returns.iloc[15:25, 0] = np.nan
    suspend.iloc[15:25, 0] = True
    returns.iloc[25, 0] = -0.60

    market = _make_market(returns, suspend=suspend)

    # 策略：持有 S0 进入停牌期（t<=16 仍选入），t>=17 才试图抛弃——此时已不可卖
    def select_fn(t):
        return stocks[:11] if t <= 16 else stocks[1:12]

    out = engine_config.RUNS_DIR / "_unit_forced_hold"
    run_backtest(market, select_fn, dates[10], dates[-1], out, {"unit": True}, min_history=5)
    daily = pd.read_csv(out / "daily.csv")
    resume_row = daily[daily["date"] == str(dates[25].date())].iloc[0]
    # 复牌日组合必须承担约 -60% * (1/11 权重) 的损失
    assert resume_row["gross"] < -0.04, resume_row["gross"]
    # 停牌中段（策略已试图抛弃后）S0 被强制持有
    frozen_days = daily[(daily["date"] >= str(dates[17].date())) & (daily["date"] <= str(dates[25].date()))]
    assert (frozen_days["forced_weight"] > 0.05).all(), frozen_days[["date", "forced_weight"]]


def test_down_limit_holding_cannot_be_sold():
    """t-1 跌停锁死的持仓必须带入 t 日，承担次日续跌。"""
    from engine import config as engine_config
    from engine.backtest import run_backtest

    dates = pd.bdate_range("2022-01-03", periods=30)
    stocks = [f"S{i}" for i in range(12)]
    returns = pd.DataFrame(0.0, index=dates, columns=stocks)
    down = pd.DataFrame(0.0, index=dates, columns=stocks)
    # S0 第 15 天跌停 -10% 且锁死，第 16 天续跌 -10%
    returns.iloc[15, 0] = -0.10
    down.iloc[15, 0] = 1.0
    returns.iloc[16, 0] = -0.10

    market = _make_market(returns, down=down)

    def select_fn(t):
        return stocks[:11] if t <= 15 else stocks[1:12]  # 第 16 天想抛 S0

    out = engine_config.RUNS_DIR / "_unit_down_limit"
    run_backtest(market, select_fn, dates[10], dates[-1], out, {"unit": True}, min_history=5)
    daily = pd.read_csv(out / "daily.csv")
    d16 = daily[daily["date"] == str(dates[16].date())].iloc[0]
    assert d16["forced_weight"] > 0.05          # S0 被强制持有
    assert d16["gross"] < -0.007                # 续跌由组合承担


def test_benchmark_excludes_limit_up():
    """基准必须剔除 t-1 涨停股（可投资一致性）。"""
    dates = pd.bdate_range("2022-01-03", periods=5)
    stocks = ["A", "B", "C"]
    returns = pd.DataFrame([[0.0, 0.0, 0.0]] * 5, index=dates, columns=stocks, dtype=float)
    up = pd.DataFrame(0.0, index=dates, columns=stocks)
    up.iloc[2, 0] = 1.0            # A 在 t=2 涨停
    returns.iloc[3] = [0.10, 0.01, 0.01]  # A 次日 +10% 只应留在市场里，不进基准

    market = _make_market(returns, up=up)
    bench = market.benchmark_return(3, stocks)
    assert abs(bench - 0.01) < 1e-12  # 只平均 B、C


def test_bj_stocks_excluded():
    """MarketData 必须剔除北交所股票。"""
    from engine.data import MarketData

    market = MarketData("all")
    assert not any(c.endswith(".BJ") for c in market.stocks)
    assert len(market.stocks) > 5000
