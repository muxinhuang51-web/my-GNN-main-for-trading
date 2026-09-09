"""生成 experiment-summary-v2.md：三池双轴曲线 + 配对统计 + 基线对照。

用法：python -m exp.report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import newey_west_tstat, sharpe_annualized

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs_v2"
SCOPES = ["csi300", "csi500", "all"]


def load_scope(scope: str) -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / f"t1_{scope}").glob("*/run_summary.json")):
        p = json.loads(f.read_text())
        d = pd.read_csv(f.parent / "daily.csv")
        ge = (d["gross"] - d["bench"]).to_numpy()
        rows.append({"name": f.parent.name, **p["params"], **p["metrics"],
                     "shp_g": sharpe_annualized(ge), "ge_bp": float(np.nanmean(ge) * 1e4),
                     "daily_path": str(f.parent / "daily.csv")})
    return pd.DataFrame(rows)


def pooled_daily_tstat(df: pd.DataFrame, names) -> float:
    """同格多次运行的逐日毛超额先跨运行取均值，再对均值序列做 NW t——
    正确处理跨种子同日相关性（比对每次运行独立算 t 更保守）。"""
    series = []
    for n in names:
        row = df[df["name"] == n]
        if row.empty:
            continue
        d = pd.read_csv(row.iloc[0]["daily_path"], index_col="date")
        series.append(d["gross"] - d["bench"])
    if not series:
        return float("nan")
    aligned = pd.concat(series, axis=1)
    return newey_west_tstat(aligned.mean(axis=1).to_numpy())


def axis_table(df: pd.DataFrame, axis: str, fixed: dict) -> pd.DataFrame:
    cl = df[df["name"].str.startswith("cluster_")].copy()
    for key, value in fixed.items():
        cl = cl[cl[key] == value]
    out = []
    for val, g in cl.groupby(axis):
        out.append({
            axis: val, "n_runs": len(g),
            "gross_sharpe": g["shp_g"].mean(), "gross_sharpe_std": g["shp_g"].std(),
            "net_sharpe": g["sharpe_excess_net"].mean(),
            "ic": g["mean_ic"].mean(), "ic_std": g["mean_ic"].std(),
            "turnover": g["mean_daily_turnover"].mean(),
            "pooled_nw_t": pooled_daily_tstat(df, g["name"].tolist()),
        })
    return pd.DataFrame(out).set_index(axis).round(3)


def baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[~df["name"].str.startswith("cluster_")].copy()
    base["family"] = base["name"].str.replace(r"_(es\d+|s\d+|seed\d+)", "", regex=True)
    out = []
    for fam, g in base.groupby("family"):
        out.append({"family": fam, "n": len(g),
                    "gross_sharpe": g["shp_g"].mean(), "std": g["shp_g"].std(),
                    "net_sharpe": g["sharpe_excess_net"].mean(),
                    "pooled_nw_t": pooled_daily_tstat(df, g["name"].tolist())})
    if not out:
        return pd.DataFrame([{"family": "(尚未完成)", "n": 0}]).set_index("family")
    return pd.DataFrame(out).set_index("family").round(3)


def main() -> int:
    lines = ["# 实验结果汇总 v2（修复版引擎，验证期 2022-2023）", ""]
    lines += ["> 协议：train 2016-2021 / valid 2022-2023（本表全部为验证期）/ test 2024+ 冻结未触碰。",
              "> 引擎：t-1 可交易性、强制持仓、可投资基准、成本 买5bp/卖15bp、北交所剔除。",
              "> 每格 = 3 编码器种子 x 3 下游种子；pooled_nw_t = 跨运行同日均值序列的 Newey-West t 值。", ""]
    for scope in SCOPES:
        if not (OUT / f"t1_{scope}").exists():
            continue
        df = load_scope(scope)
        lines += [f"## {scope}", ""]
        lines += ["### 聚类数 k 轴（k_e=0）", "", axis_table(df, "k", {"k_e": 0}).to_markdown(), ""]
        lines += ["### 相关边 k_e 轴（k=20）", "", axis_table(df, "k_e", {"k": 20}).to_markdown(), ""]
        lines += ["### 基线", "", baseline_table(df).to_markdown(), ""]
    report = "\n".join(lines)
    # 注意：叙事报告在 experiment-summary-v2.md（手写维护）；本脚本只生成机器表格
    (ROOT / "experiment-tables-v2.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
