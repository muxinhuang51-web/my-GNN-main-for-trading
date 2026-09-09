"""引擎全局约定：时间防火墙、成本、路径。"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data_v2" / "panel"
CACHE_DIR = ROOT / "data_v2" / "cache"
MODELS_DIR = ROOT / "data_v2" / "models"
RUNS_DIR = ROOT / "outputs_v2"

# 时间防火墙。一切调参/选型/选种子只允许用 TRAIN+VALID。
# TEST 在结果冻结前只允许由 final 报告脚本访问一次。
TRAIN_END = pd.Timestamp("2021-12-31")
VALID_START = pd.Timestamp("2022-01-01")
VALID_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

# 交易成本（A 股惯例：佣金+冲击 5bp 买入；卖出含印花税 15bp）
BUY_COST = 0.0005
SELL_COST = 0.0015
COST_STRESS = [0.001, 0.003]  # 压力档（每边）

# 特征与图
LOOKBACK = 60
FEATURE_MIN_OBS = 40          # 窗口内至少 40 个有效收益才生成特征，否则该股当日出池
CORR_CACHE_TOPK = 40          # 相关邻居缓存上限，k_e<=40 直接切片
INDUSTRY_MAX_NEIGHBORS = 20

# 编码器
HIDDEN_DIM = 64
ENCODER_EPOCHS = 20
ENCODER_LR = 1e-3
ENCODER_PATIENCE = 3

SEED_DEFAULT = 42


def ensure_dirs():
    for d in (CACHE_DIR, MODELS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
