"""tushare 客户端封装：限速、重试、断点续传。"""

import json
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from . import config


def _load_env_file() -> None:
    """将项目根目录 .env 中的键值注入环境（不覆盖已有环境变量）。"""
    import os

    env_path = config.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class TushareClient:
    """带令牌桶限速和自适应退避的 pro_api 封装。"""

    def __init__(self, calls_per_minute: int = config.CALLS_PER_MINUTE):
        _load_env_file()
        import tushare as ts

        self.pro = ts.pro_api(config.get_token())
        self.min_interval = 60.0 / calls_per_minute
        self._last_call = 0.0

    def query(self, api_name: str, **params) -> pd.DataFrame:
        """单次调用：限速 + 命中频控时指数退避重试。"""
        for attempt in range(config.MAX_RETRIES):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                return self.pro.query(api_name, **params)
            except Exception as error:  # tushare 抛的是普通 Exception，带中文信息
                message = str(error)
                # 频控类错误：等待后重试；权限类错误：直接抛出
                if any(kw in message for kw in ("每分钟", "频率", "限制", "超过")):
                    backoff = 30 * (attempt + 1)
                    print(f"[限速] {api_name} 命中频控，等待 {backoff}s: {message[:80]}")
                    time.sleep(backoff)
                    continue
                if attempt < config.MAX_RETRIES - 1 and "积分" not in message and "权限" not in message:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise RuntimeError(f"{api_name} 重试 {config.MAX_RETRIES} 次仍失败")


class DateCheckpoint:
    """按 trade_date 断点续传：state 文件记录已完成日期。"""

    def __init__(self, stage: str):
        config.ensure_dirs()
        self.path = config.STATE_DIR / f"{stage}.done"
        self._done = set(self.path.read_text().split()) if self.path.exists() else set()

    def is_done(self, date: str) -> bool:
        return date in self._done

    def mark(self, date: str) -> None:
        self._done.add(date)
        with self.path.open("a") as f:
            f.write(date + "\n")


def fetch_by_trade_date(
    client: TushareClient,
    api_name: str,
    dates: Iterable[str],
    out_dir: Path,
    fields: Optional[str] = None,
    empty_ok: bool = True,
) -> None:
    """逐交易日拉取并按日落盘 parquet，支持断点续传。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = DateCheckpoint(f"{api_name}")
    dates = [d for d in dates if not checkpoint.is_done(d)]
    if not dates:
        print(f"[{api_name}] 已全部完成，跳过")
        return
    print(f"[{api_name}] 待拉取 {len(dates)} 个交易日")
    start_time = time.monotonic()
    for i, date in enumerate(dates, 1):
        params = {"trade_date": date}
        if fields:
            params["fields"] = fields
        frame = client.query(api_name, **params)
        if frame is None or frame.empty:
            if not empty_ok:
                print(f"[警告] {api_name} {date} 返回空")
        else:
            frame.to_parquet(out_dir / f"{date}.parquet", index=False)
        checkpoint.mark(date)
        if i % 100 == 0:
            rate = i / (time.monotonic() - start_time) * 60
            remain = (len(dates) - i) / max(rate, 1e-9)
            print(f"[{api_name}] {i}/{len(dates)} ({rate:.0f} 日/分钟, 剩余约 {remain:.0f} 分钟)")
    print(f"[{api_name}] 完成")


def load_stage(out_dir: Path) -> pd.DataFrame:
    """合并某阶段的全部按日 parquet。"""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def update_manifest(entry_name: str, info: dict) -> None:
    """记录每个阶段的拉取信息，保证快照可追溯。"""
    manifest = {}
    if config.MANIFEST_PATH.exists():
        manifest = json.loads(config.MANIFEST_PATH.read_text())
    info = dict(info)
    info["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest[entry_name] = info
    config.MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
