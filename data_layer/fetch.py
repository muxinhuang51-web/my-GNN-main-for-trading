"""各数据阶段的拉取逻辑。

阶段清单（全部支持断点续传）：
  trade_cal    交易日历（1 次调用）
  stock_basic  股票列表含退市（3 次调用：L/D/P）→ PIT 股票池的基础
  daily        日线行情（按 trade_date，~2800 次）
  adj_factor   复权因子（按 trade_date，~2800 次）
  suspend_d    停复牌（按 trade_date，~2800 次；无权限时可从 daily 缺失推断）
  stk_limit    涨跌停价（按 trade_date，~2800 次；无权限时按前收 ±10%/20% 近似）
  index_weight 指数成分权重（按月，CSI300/CSI500）
  sw_industry  申万行业分类及成员（含 in_date/out_date → PIT 行业）
"""

import time
from typing import List, Optional

import pandas as pd

from . import config
from .api import TushareClient, fetch_by_trade_date, update_manifest


def get_trade_dates(client: TushareClient) -> List[str]:
    """交易日列表（YYYYMMDD），并落盘日历。"""
    config.ensure_dirs()
    cal_path = config.RAW_DIR / "trade_cal.parquet"
    end_date = config.END_DATE or time.strftime("%Y%m%d")
    if cal_path.exists():
        cal = pd.read_parquet(cal_path)
        if cal["cal_date"].max() >= end_date:
            return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].tolist())
    cal = client.query(
        "trade_cal", exchange="SSE", start_date=config.START_DATE, end_date=end_date
    )
    cal.to_parquet(cal_path, index=False)
    update_manifest("trade_cal", {"rows": len(cal), "start": config.START_DATE, "end": end_date})
    return sorted(cal.loc[cal["is_open"] == 1, "cal_date"].tolist())


def fetch_stock_basic(client: TushareClient) -> pd.DataFrame:
    """股票基本信息：在市 + 退市 + 暂停上市，含上市/退市日期。"""
    frames = []
    for status in ("L", "D", "P"):
        frame = client.query(
            "stock_basic",
            list_status=status,
            fields="ts_code,name,area,industry,market,list_date,delist_date,list_status",
        )
        frames.append(frame)
    basic = pd.concat(frames, ignore_index=True)
    out = config.RAW_DIR / "stock_basic.parquet"
    basic.to_parquet(out, index=False)
    update_manifest("stock_basic", {"rows": len(basic), "listed": int((basic.list_status == "L").sum()), "delisted": int((basic.list_status == "D").sum())})
    print(f"[stock_basic] 共 {len(basic)} 只（含退市 {(basic.list_status == 'D').sum()} 只）")
    return basic


def fetch_daily_stages(client: TushareClient, dates: List[str], stages: Optional[List[str]] = None) -> None:
    """按交易日拉取行情类数据。"""
    stage_specs = {
        "daily": dict(fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount"),
        "adj_factor": dict(fields="ts_code,trade_date,adj_factor"),
        "suspend_d": dict(fields=None),
        "stk_limit": dict(fields="ts_code,trade_date,up_limit,down_limit"),
    }
    for stage in stages or list(stage_specs):
        spec = stage_specs[stage]
        try:
            fetch_by_trade_date(
                client, stage, dates, config.RAW_DIR / stage, fields=spec["fields"]
            )
            update_manifest(stage, {"dates": len(dates)})
        except Exception as error:
            print(f"[跳过] {stage} 拉取失败（可能是权限不足）：{str(error)[:120]}")
            print(f"        suspend_d 可由 daily 缺失推断；stk_limit 可按前收 ±10%/20% 近似。")


def fetch_index_weights(client: TushareClient) -> None:
    """指数成分权重：按半年窗口分段拉取（接口单次限 ~ 数千行）。"""
    end_date = config.END_DATE or time.strftime("%Y%m%d")
    for name, code in config.INDEX_CODES.items():
        out_path = config.RAW_DIR / f"index_weight_{name}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            if existing["trade_date"].max() >= end_date[:6] + "01":
                print(f"[index_weight:{name}] 已最新，跳过")
                continue
        frames = []
        years = range(int(config.START_DATE[:4]), int(end_date[:4]) + 1)
        for year in years:
            for half in ((f"{year}0101", f"{year}0630"), (f"{year}0701", f"{year}1231")):
                frame = client.query("index_weight", index_code=code, start_date=half[0], end_date=half[1])
                if frame is not None and not frame.empty:
                    frames.append(frame)
        weights = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["trade_date", "con_code"])
        weights.to_parquet(out_path, index=False)
        update_manifest(f"index_weight_{name}", {"rows": len(weights), "months": weights["trade_date"].str[:6].nunique()})
        print(f"[index_weight:{name}] {len(weights)} 行, {weights['trade_date'].str[:6].nunique()} 个月")


def fetch_sw_industry(client: TushareClient) -> None:
    """申万一级行业分类与成员（带 in_date/out_date，可 PIT 回溯）。"""
    classify = client.query("index_classify", level="L1", src="SW2021")
    if classify is None or classify.empty:
        classify = client.query("index_classify", level="L1", src="SW")
    classify.to_parquet(config.RAW_DIR / "sw_classify.parquet", index=False)

    members = []
    for _, row in classify.iterrows():
        frame = client.query("index_member", index_code=row["index_code"], is_new="N")
        frame_new = client.query("index_member", index_code=row["index_code"], is_new="Y")
        for f in (frame, frame_new):
            if f is not None and not f.empty:
                f = f.copy()
                f["industry_name"] = row["industry_name"]
                members.append(f)
    member_df = pd.concat(members, ignore_index=True).drop_duplicates(subset=["index_code", "con_code", "in_date"])
    member_df.to_parquet(config.RAW_DIR / "sw_member.parquet", index=False)
    update_manifest("sw_industry", {"industries": len(classify), "member_rows": len(member_df)})
    print(f"[sw_industry] {len(classify)} 个一级行业, {len(member_df)} 条成员记录")


def probe_permissions() -> dict:
    """探测 token 对各接口的权限，返回 {接口: ok/错误信息}。"""
    client = TushareClient()
    probes = {
        "trade_cal": dict(exchange="SSE", start_date="20240101", end_date="20240110"),
        "stock_basic": dict(list_status="L", fields="ts_code,list_date"),
        "daily": dict(trade_date="20240105"),
        "adj_factor": dict(trade_date="20240105"),
        "suspend_d": dict(trade_date="20240105"),
        "stk_limit": dict(trade_date="20240105"),
        "index_weight": dict(index_code="399300.SZ", start_date="20240101", end_date="20240131"),
        "index_classify": dict(level="L1", src="SW2021"),
        "index_member": dict(index_code="801010.SI"),
    }
    results = {}
    for api_name, params in probes.items():
        try:
            frame = client.query(api_name, **params)
            rows = 0 if frame is None else len(frame)
            results[api_name] = f"OK ({rows} rows)"
        except Exception as error:
            results[api_name] = f"FAIL: {str(error)[:100]}"
    return results
