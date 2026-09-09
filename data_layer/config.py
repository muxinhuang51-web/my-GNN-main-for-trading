"""数据层全局配置：路径、日期范围、接口参数。"""

import os
from pathlib import Path

# 项目根目录与数据目录（v2 与旧 data/ 隔离，避免混用带偏差的旧数据）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_v2"
RAW_DIR = DATA_DIR / "raw"
PANEL_DIR = DATA_DIR / "panel"
STATE_DIR = DATA_DIR / "state"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# 拉取范围：2015 年起，覆盖多种市场状态（2015 崩盘、2018 熊、2019-2021 牛、2022-2024 震荡）
START_DATE = "20150101"
END_DATE = None  # None = 拉到最新交易日

# 时间防火墙（所有调参只允许使用 train+valid；test 冻结）
SPLIT_TRAIN = ("2016-01-01", "2021-12-31")
SPLIT_VALID = ("2022-01-01", "2023-12-31")
SPLIT_TEST = ("2024-01-01", None)

# 指数成分股池
INDEX_CODES = {
    "csi300": "399300.SZ",
    "csi500": "000905.SH",
}

# tushare 限速：自适应退避的起始参数（不同积分档位限速不同）
CALLS_PER_MINUTE = 190  # 保守值，命中限速时自动退避
MAX_RETRIES = 5

TOKEN_ENV = "TUSHARE_TOKEN"


def get_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"环境变量 {TOKEN_ENV} 未设置。请运行: export {TOKEN_ENV}=<你的token>"
        )
    return token


def ensure_dirs() -> None:
    for d in (RAW_DIR, PANEL_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
