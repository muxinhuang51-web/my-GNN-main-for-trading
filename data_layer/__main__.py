"""数据层 CLI。

用法：
  python -m data_layer probe            # 探测 token 权限
  python -m data_layer fetch            # 全量拉取（断点续传，可反复运行）
  python -m data_layer panel            # 构建面板
  python -m data_layer validate         # 校验
"""

import sys

from . import config


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "probe"
    config.ensure_dirs()

    if command == "probe":
        from .fetch import probe_permissions

        results = probe_permissions()
        print("=== tushare 接口权限探测 ===")
        for api_name, status in results.items():
            print(f"  {api_name:15s} {status}")
        return 0

    if command == "fetch":
        from .api import TushareClient
        from .fetch import (
            fetch_daily_stages,
            fetch_index_weights,
            fetch_stock_basic,
            fetch_sw_industry,
            get_trade_dates,
        )

        client = TushareClient()
        dates = get_trade_dates(client)
        print(f"[fetch] 交易日 {len(dates)} 个 ({dates[0]} ~ {dates[-1]})")
        fetch_stock_basic(client)
        fetch_sw_industry(client)
        fetch_index_weights(client)
        fetch_daily_stages(client, dates)
        print("[fetch] 全部阶段完成")
        return 0

    if command == "panel":
        from .panel import build_price_panels, build_tradability_panels, build_universe_panels

        build_price_panels()
        build_tradability_panels()
        build_universe_panels()
        return 0

    if command == "validate":
        from .validate import run_all

        return 0 if run_all() else 1

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
