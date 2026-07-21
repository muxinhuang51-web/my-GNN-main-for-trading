"""数据面板校验器：全绿才允许进入引擎/训练阶段。"""

import numpy as np
import pandas as pd

from . import config


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")
    return bool(condition)


def run_all() -> bool:
    print("=== 数据面板校验 ===")
    ok = True
    returns = pd.read_parquet(config.PANEL_DIR / "returns.parquet")
    universe = pd.read_parquet(config.PANEL_DIR / "universe_pit.parquet")
    basic = pd.read_parquet(config.RAW_DIR / "stock_basic.parquet")

    # 1. 覆盖范围
    ok &= check(
        "时间跨度 >= 9 年",
        (returns.index.max() - returns.index.min()).days >= 9 * 365,
        f"({returns.index.min().date()} ~ {returns.index.max().date()})",
    )
    daily_counts = returns.notna().sum(axis=1)
    ok &= check("每日有效股票数中位数 >= 2000", daily_counts.median() >= 2000, f"(中位数 {daily_counts.median():.0f})")

    # 2. 收益分布合理性（复权正确性的整体检验：未复权会产生大量假性 <-10% 收益）
    flat = returns.values.ravel()
    flat = flat[np.isfinite(flat)]
    below_limit = (flat < -0.22).mean()  # 超过 20cm 跌停的收益应极少（仅退市整理期等）
    ok &= check("超越跌停幅度的收益占比 < 0.05%", below_limit < 5e-4, f"({below_limit:.4%})")

    # 零收益对账：精确 0 必须对应原始数据的真实平盘（close==pre_close），否则说明存在填充伪造
    # （A 股 1 分钱价位下平盘常见，占比 1%-4% 属正常，故不用占比阈值而用对账）
    from .api import load_stage

    stacked = returns.stack()
    zeros = stacked[stacked == 0.0]
    if len(zeros) > 0:
        sample = zeros.sample(min(500, len(zeros)), random_state=0)
        daily = load_stage(config.RAW_DIR / "daily").set_index(["trade_date", "ts_code"])
        match = 0
        for (date, code), _ in sample.items():
            key = (date.strftime("%Y%m%d"), code)
            if key in daily.index and abs(daily.loc[key, "close"] - daily.loc[key, "pre_close"]) < 1e-9:
                match += 1
        ok &= check("零收益对账为真实平盘 >= 99%", match / len(sample) >= 0.99, f"({match}/{len(sample)})")
    else:
        ok &= check("零收益对账为真实平盘 >= 99%", True, "(无零收益)")

    # 3. 分红事件抽检：复权后除息日不应出现深跌（以高分红银行股为例）
    sample = [c for c in ("601398.SH", "600036.SH", "000001.SZ") if c in returns.columns]
    if sample:
        min_ret = returns[sample].min().min()
        ok &= check("高分红样本股复权后单日最低收益 > -25%", min_ret > -0.25, f"(min {min_ret:.2%})")

    # 4. PIT 股票池含退市股
    delisted = basic.loc[basic["list_status"] == "D", "ts_code"]
    delisted_in_panel = [c for c in delisted if c in returns.columns]
    ok &= check("面板包含退市股（幸存者偏差修复）", len(delisted_in_panel) > 50, f"({len(delisted_in_panel)} 只)")

    # 5. PIT 断言：退市股在退市日之后不在池内
    violations = 0
    basic_idx = basic.set_index("ts_code")
    for code in delisted_in_panel[:200]:
        delist_dt = pd.to_datetime(basic_idx.loc[code, "delist_date"], errors="coerce")
        if pd.isna(delist_dt):
            continue
        after = universe.loc[universe.index >= delist_dt, code]
        violations += int(after.any())
    ok &= check("退市后不在池（抽检 200 只）", violations == 0, f"({violations} 例违规)")

    # 6. 行业区间表可用
    intervals = pd.read_parquet(config.PANEL_DIR / "sw_industry_intervals.parquet")
    covered = intervals["ts_code"].nunique()
    ok &= check("行业映射覆盖 >= 3000 只", covered >= 3000, f"({covered} 只)")

    # 7. 指数成分历史
    for name in config.INDEX_CODES:
        path = config.PANEL_DIR / f"index_members_{name}.parquet"
        if path.exists():
            weights = pd.read_parquet(path)
            months = weights["trade_date"].dt.to_period("M").nunique()
            ok &= check(f"{name} 成分历史 >= 100 个月", months >= 100, f"({months} 个月)")
        else:
            ok &= check(f"{name} 成分文件存在", False)

    print("=== 结果:", "全部通过" if ok else "存在失败项，禁止进入下一阶段", "===")
    return bool(ok)
