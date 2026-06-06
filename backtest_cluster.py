import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from models.cluster_predictor import (
    ClusterReturnPredictor,
    build_cluster_feature_samples,
    predict_cluster_returns,
    train_cluster_predictor,
)
from models.embedding_model import extract_embeddings, load_embedding_model


# 本文件负责“簇级轮动”回测：
# 1. 用历史窗口构造股票特征和关系图
# 2. 用训练好的 GNN 导出股票 embedding
# 3. 对 embedding 聚类，得到股票簇
# 4. 用过去窗口训练簇收益预测器
# 5. 选择预测收益最高的簇，计算当日组合收益


def set_seed(seed_value: int) -> None:
    """固定随机种子，尽量让 KMeans 和 torch 训练结果可复现。"""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def load_returns_csv(path: str) -> pd.DataFrame:
    """读取日收益率表，并整理成 index=日期、columns=股票代码的矩阵。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"收益率文件不存在: {path}")
    dataframe = pd.read_csv(path)
    date_col = "trade_date" if "trade_date" in dataframe.columns else dataframe.columns[0]
    dates = pd.to_datetime(dataframe[date_col], errors="coerce")
    returns = dataframe.drop(columns=[date_col], errors="ignore")
    returns.columns = [str(column) for column in returns.columns]
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns.index = dates
    returns = returns[~returns.index.isna()].sort_index()
    return returns


def load_industry_mapping(path: str, stock_codes: Sequence[str]) -> pd.DataFrame:
    """读取行业映射，只保留当前股票池里的股票。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"行业映射不存在: {path}")
    dataframe = pd.read_csv(path)
    dataframe["ts_code"] = dataframe["ts_code"].astype(str)
    return dataframe[dataframe["ts_code"].isin(stock_codes)].copy()


def load_concept_mapping(data_dir: str, stock_codes: Sequence[str]) -> pd.DataFrame:
    """读取概念映射；如果数据文件不存在，则返回空表，后续图构造会自动跳过概念边。"""
    candidates = [
        os.path.join(data_dir, "stock_concept.csv"),
        os.path.join(data_dir, "concept_mapping.csv"),
        os.path.join(data_dir, "stock_concepts.csv"),
    ]
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        return pd.DataFrame(columns=["ts_code", "concept"])

    dataframe = pd.read_csv(path)
    if "ts_code" not in dataframe.columns:
        return pd.DataFrame(columns=["ts_code", "concept"])
    concept_col = next((column for column in dataframe.columns if column != "ts_code"), None)
    if concept_col is None:
        return pd.DataFrame(columns=["ts_code", "concept"])

    result = dataframe[["ts_code", concept_col]].copy()
    result.columns = ["ts_code", "concept"]
    result["ts_code"] = result["ts_code"].astype(str)
    return result[result["ts_code"].isin(stock_codes)].copy()


def zscore_by_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """对每个特征列做横截面标准化。

    零标准差列（所有股票值相同）替换为 NaN，后续由调用方 fillna(0) 处理，
    此时该特征不含区分度，填 0 等价于将其从模型输入中置零。
    """
    mean = dataframe.mean(axis=0)
    std = dataframe.std(axis=0).replace(0, np.nan)
    return (dataframe - mean) / std


def build_features_from_window(window_dataframe: pd.DataFrame) -> pd.DataFrame:
    """从历史收益窗口构造每只股票的节点特征。"""
    # 这些特征全部来自 time_index 之前的窗口，避免在特征里混入预测日收益。
    momentum_20 = (1 + window_dataframe.tail(20)).prod() - 1
    mean_5 = window_dataframe.tail(5).mean()
    mean_10 = window_dataframe.tail(10).mean()
    # 历史波动率：截取不超过窗口实际长度的天数，避免 lookback < 60 时静默退化。
    volatility_20 = window_dataframe.tail(min(20, len(window_dataframe))).std(ddof=0)
    volatility_60 = window_dataframe.tail(min(60, len(window_dataframe))).std(ddof=0)
    last_return = window_dataframe.tail(1).iloc[0]
    feature_frame = pd.DataFrame(
        {
            "mom20": momentum_20,
            "mean5": mean_5,
            "mean10": mean_10,
            "vol20": volatility_20,
            "vol60": volatility_60,
            "last_ret": last_return,
        }
    )
    return zscore_by_column(feature_frame)


def build_stock2index(stock_codes: Sequence[str]) -> Dict[str, int]:
    """生成股票代码到节点编号的映射，保证边索引和特征矩阵对齐。"""
    return {code: index for index, code in enumerate(stock_codes)}


def build_group_edges(
    mapping_df: pd.DataFrame,
    stock2index: Dict[str, int],
    stock_column: str,
    group_column: str,
    max_neighbors: int = 20,
) -> Optional[np.ndarray]:
    """根据行业或概念分组构造同组股票之间的有向边。"""
    if mapping_df.empty or stock_column not in mapping_df.columns or group_column not in mapping_df.columns:
        return None

    edge_pairs = set()
    grouped = mapping_df.groupby(group_column)[stock_column].apply(list).to_dict()
    for members in grouped.values():
        members = sorted({member for member in members if member in stock2index}, key=stock2index.get)
        for source in members:
            # 限制每个股票的同组邻居数量，避免大行业或大概念生成过密的图。
            peers = [member for member in members if member != source][:max_neighbors]
            for target in peers:
                edge_pairs.add((stock2index[source], stock2index[target]))
    if not edge_pairs:
        return None
    return np.array(sorted(edge_pairs), dtype=np.int64).T


def build_corr_edges(
    window_dataframe: pd.DataFrame,
    top_neighbor_count: int = 20,
    min_overlap: int = 20,
) -> Optional[np.ndarray]:
    """根据历史收益相关性构造股票之间的相似边。"""
    if top_neighbor_count <= 0:
        return None
    if len(window_dataframe) < min_overlap:
        return None

    corr = window_dataframe.astype(float).corr(min_periods=min_overlap).to_numpy(dtype=float)
    edge_pairs = set()
    for source_index in range(corr.shape[0]):
        row = corr[source_index].copy()
        row[source_index] = np.nan
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            continue
        candidates = np.flatnonzero(finite_mask)
        ranked = candidates[np.argsort(np.abs(row[candidates]))]
        for target_index in ranked[-top_neighbor_count:]:
            # 取绝对相关性最高的若干邻居，正相关和负相关都被视为强关系。
            edge_pairs.add((source_index, int(target_index)))

    if not edge_pairs:
        return None
    return np.array(sorted(edge_pairs), dtype=np.int64).T


def tensor_from_edges(edge_index: Optional[np.ndarray]) -> torch.Tensor:
    """把 numpy 边索引转成 PyTorch Tensor；缺失边时返回空边。"""
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_index, dtype=torch.long)


def build_data_for_index(
    returns: pd.DataFrame,
    industry_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    stock_codes: Sequence[str],
    time_index: int,
    lookback: int,
    top_neighbor_count: int,
) -> Data:
    """为某个交易日构造 PyG Data，作为 GNN embedding 模型的输入。"""
    if time_index < lookback:
        raise ValueError("time_index 小于 lookback，无法构造历史窗口")

    # 使用 [time_index - lookback, time_index) 的历史收益构造预测日特征。
    window = returns.iloc[time_index - lookback:time_index]
    feature_frame = build_features_from_window(window).reindex(stock_codes).fillna(0.0)
    stock2index = build_stock2index(stock_codes)

    # 三类关系边：行业边、概念边、历史相关性边。
    industry_edge = tensor_from_edges(
        build_group_edges(industry_df, stock2index, "ts_code", "industry", max_neighbors=20)
    )
    concept_edge = tensor_from_edges(
        build_group_edges(concept_df, stock2index, "ts_code", "concept", max_neighbors=20)
    )
    corr_edge = tensor_from_edges(
        build_corr_edges(window.reindex(columns=stock_codes), top_neighbor_count=top_neighbor_count)
    )

    # edge_type 与 edge_index 的列一一对应：0=行业，1=概念，2=相关性。
    edge_parts = [industry_edge, concept_edge, corr_edge]
    type_parts = [
        torch.zeros(industry_edge.size(1), dtype=torch.long),
        torch.ones(concept_edge.size(1), dtype=torch.long),
        torch.full((corr_edge.size(1),), 2, dtype=torch.long),
    ]
    edge_index = torch.cat(edge_parts, dim=1).contiguous()
    edge_type = torch.cat(type_parts, dim=0).contiguous()

    return Data(
        x=torch.tensor(feature_frame.to_numpy(dtype=float), dtype=torch.float),
        edge_index=edge_index,
        edge_type=edge_type,
    )


def date_positions(
    returns: pd.DataFrame,
    lookback: int,
    train_window: int,
    min_market_valid_stocks: int = 1000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[int]:
    """给出可回测的日期下标，确保每个日期都有足够 lookback 和训练窗口。"""
    start_index = lookback + train_window
    valid_counts = returns.notna().sum(axis=1)
    valid_positions = [
        index
        for index in range(start_index, len(returns))
        if valid_counts.iloc[index] >= min_market_valid_stocks
    ]
    if start_date is not None:
        start = pd.to_datetime(start_date)
        valid_positions = [index for index in valid_positions if returns.index[index] >= start]
    if end_date is not None:
        end = pd.to_datetime(end_date)
        valid_positions = [index for index in valid_positions if returns.index[index] <= end]
    return valid_positions


def compute_portfolio_return(returns_row: pd.Series, selected_stocks: Sequence[str], min_valid_stocks: int = 1) -> float:
    """计算当日选中股票的等权平均收益。"""
    if not selected_stocks:
        return 0.0
    valid_returns = returns_row.reindex(selected_stocks).dropna()
    if len(valid_returns) < min_valid_stocks:
        return float("nan")
    return float(valid_returns.mean())


def select_clusters_until_target_valid_stocks(
    cluster_ids: Sequence[int],
    predicted_cluster_returns: np.ndarray,
    stock_codes: Sequence[str],
    cluster_labels: np.ndarray,
    returns_row: pd.Series,
    target_valid_stocks: int,
) -> Tuple[List[int], List[str], int]:
    """按预测收益降序逐簇加入，直到有效股票数量达到目标。"""
    ranked_indices = np.argsort(predicted_cluster_returns)[::-1]
    selected_clusters: List[int] = []
    selected_stocks: List[str] = []
    selected_stock_set = set()
    valid_count = 0

    for cluster_index in ranked_indices:
        cluster_id = int(cluster_ids[cluster_index])
        cluster_stocks = [
            code for code, label in zip(stock_codes, cluster_labels) if int(label) == cluster_id
        ]
        cluster_valid_count = int(returns_row.reindex(cluster_stocks).notna().sum())
        if cluster_valid_count == 0:
            continue

        selected_clusters.append(cluster_id)
        for code in cluster_stocks:
            if code not in selected_stock_set:
                selected_stocks.append(code)
                selected_stock_set.add(code)

        valid_count += cluster_valid_count
        # 最后一个簇整体加入，valid_count 可能超过 target；论文中注明此行为即可。
        if valid_count >= target_valid_stocks:
            break

    if not selected_stocks:
        print(f"[警告] 所有 {len(cluster_ids)} 个簇有效股票数均为 0，当日组合为空，记为 0 收益")

    return selected_clusters, selected_stocks, valid_count


def get_day_embeddings(
    embedding_cache: Dict[Tuple[int, int], np.ndarray],
    model,
    returns: pd.DataFrame,
    industry_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    stock_codes: Sequence[str],
    day_index: int,
    lookback: int,
    top_neighbor_count: int,
    device: torch.device,
) -> np.ndarray:
    """读取或计算某日股票 embedding。"""
    cache_key = (day_index, top_neighbor_count)
    if cache_key not in embedding_cache:
        data = build_data_for_index(
            returns, industry_df, concept_df, stock_codes, day_index, lookback, top_neighbor_count
        )
        embedding_cache[cache_key] = extract_embeddings(model, data, device)
    return embedding_cache[cache_key]


def get_day_clusters(
    cluster_cache: Dict[Tuple[int, int, int, int, int], Tuple[np.ndarray, np.ndarray, List[int]]],
    embedding_cache: Dict[Tuple[int, int], np.ndarray],
    model,
    returns: pd.DataFrame,
    industry_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    stock_codes: Sequence[str],
    day_index: int,
    lookback: int,
    top_neighbor_count: int,
    cluster_count: int,
    seed_value: int,
    kmeans_n_init: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """读取或计算某日 KMeans 标签和簇级特征。"""
    cache_key = (day_index, top_neighbor_count, cluster_count, seed_value, kmeans_n_init)
    if cache_key not in cluster_cache:
        embeddings = get_day_embeddings(
            embedding_cache,
            model,
            returns,
            industry_df,
            concept_df,
            stock_codes,
            day_index,
            lookback,
            top_neighbor_count,
            device,
        )
        labels = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=kmeans_n_init).fit_predict(
            embeddings
        )
        cluster_features, cluster_ids = build_cluster_feature_samples(embeddings, labels)
        cluster_cache[cache_key] = (labels, cluster_features, cluster_ids)
    return cluster_cache[cache_key]


def compute_metrics(daily_returns: Sequence[float]) -> Dict[str, Optional[float]]:
    """根据日收益序列计算年化收益、Sharpe 和最大回撤。"""
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"annualized_return": None, "sharpe": None, "max_drawdown": None, "days": 0}

    # 使用 log1p + exp 组合避免 float64 溢出（np.prod 在长回测下可能超过 1e308）。
    log_cumulative = np.sum(np.log1p(arr))
    years = len(arr) / 252.0
    annualized_return = np.exp(log_cumulative / years) - 1 if years > 0 else np.nan
    mean_daily = arr.mean()
    std_daily = arr.std(ddof=1) if len(arr) > 1 else 0.0
    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else np.nan
    wealth = np.exp(np.cumsum(np.log1p(arr)))
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak

    return {
        "annualized_return": float(annualized_return) if np.isfinite(annualized_return) else None,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "max_drawdown": float(np.min(drawdown)),
        "days": int(len(arr)),
    }


def export_backtest_plots(result_df: pd.DataFrame, out_dir: str) -> None:
    """导出累计净值、日收益和回撤曲线。"""
    if result_df.empty or "daily_return" not in result_df.columns:
        print("[警告] 回测结果为空，跳过绘图")
        return

    plot_df = result_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df["daily_return"] = pd.to_numeric(plot_df["daily_return"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date", "daily_return"]).sort_values("date")
    if plot_df.empty:
        print("[警告] 回测结果没有有效日期或收益，跳过绘图")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df["cum_return"] = (1 + plot_df["daily_return"]).cumprod()
    peak = plot_df["cum_return"].cummax()
    plot_df["drawdown"] = (plot_df["cum_return"] - peak) / peak

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["date"], plot_df["cum_return"], label="cum_return")
    plt.xlabel("date")
    plt.ylabel("net value")
    plt.title("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cum_return.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["date"], plot_df["daily_return"], label="daily_return")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("date")
    plt.ylabel("return")
    plt.title("Daily Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "daily_return.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["date"], plot_df["drawdown"], label="drawdown")
    plt.xlabel("date")
    plt.ylabel("drawdown")
    plt.title("Drawdown")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown.png"), dpi=150)
    plt.close()

    print(f"[状态] 回测图表已保存到 {out_dir}/")


def train_predictor_for_day(
    model,
    returns: pd.DataFrame,
    industry_df: pd.DataFrame,
    concept_df: pd.DataFrame,
    stock_codes: Sequence[str],
    day_index: int,
    lookback: int,
    train_window: int,
    top_neighbor_count: int,
    cluster_count: int,
    seed_value: int,
    device: torch.device,
    min_cluster_valid_count: int,
    epochs: int,
    kmeans_n_init: int,
    embedding_cache: Dict[Tuple[int, int], np.ndarray],
    cluster_cache: Dict[Tuple[int, int, int, int, int], Tuple[np.ndarray, np.ndarray, List[int]]],
) -> Optional[ClusterReturnPredictor]:
    """为某个回测日训练簇收益预测器。

    训练数据来自 [day_index - train_window, day_index)。
    对每个历史训练日，先用该日之前的 lookback 窗口导出 embedding，
    再聚类并构造簇级特征，标签是该训练日的真实簇平均收益。
    """
    feature_batches = []
    label_batches = []
    for train_index in range(day_index - train_window, day_index):
        try:
            # 训练日的特征只来自 train_index 之前的历史窗口；embedding 和聚类结果做内存缓存。
            labels, cluster_features, cluster_ids = get_day_clusters(
                cluster_cache,
                embedding_cache,
                model,
                returns,
                industry_df,
                concept_df,
                stock_codes,
                train_index,
                lookback,
                top_neighbor_count,
                cluster_count,
                seed_value,
                kmeans_n_init,
                device,
            )
            # 标签使用 train_index 当天收益；这对应“历史窗口 -> 当天收益”的监督样本。
            returns_array = returns.iloc[train_index].to_numpy(dtype=float)
            features = []
            cluster_returns = []
            for cluster_position, cluster_id in enumerate(cluster_ids):
                member_indices = np.flatnonzero(labels == cluster_id)
                valid_returns = returns_array[member_indices]
                valid_returns = valid_returns[np.isfinite(valid_returns)]
                if valid_returns.size < min_cluster_valid_count:
                    continue
                features.append(cluster_features[cluster_position])
                cluster_returns.append(float(valid_returns.mean()))
            if not features:
                raise ValueError("训练日没有满足有效样本数的簇")
            features = np.vstack(features)
            cluster_returns = np.asarray(cluster_returns, dtype=float)
        except ValueError as error:
            print(f"[警告] 跳过训练日 {train_index}: {error}")
            continue
        feature_batches.append(features)
        label_batches.append(cluster_returns)

    if not feature_batches:
        return None

    x_train = np.vstack(feature_batches)
    y_train = np.concatenate(label_batches)
    # 当前版本每天从零初始化簇预测器；后续可改成继承昨日 checkpoint 做日频微调。
    predictor = ClusterReturnPredictor(input_dim=x_train.shape[1], hidden_dim=32, dropout_rate=0.1)
    return train_cluster_predictor(
        predictor,
        x_train,
        y_train,
        device,
        epochs=epochs,
        batch_size=32,
        learning_rate=1e-3,
    )


def evaluate_prediction_ic(predictions: np.ndarray, realized_cluster_returns: np.ndarray) -> float:
    """计算簇预测收益和真实簇收益之间的截面相关系数。"""
    mask = np.isfinite(predictions) & np.isfinite(realized_cluster_returns)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(predictions[mask], realized_cluster_returns[mask])[0, 1])


def json_safe(value):
    """把 numpy/pandas 标量和 NaN 转成 JSON 友好的 Python 类型。"""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.ndarray,)):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str, payload: Dict) -> None:
    """原子写入 JSON 文件，避免中断时留下半截文件。"""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def write_jsonl(path: str, records: Sequence[Dict]) -> None:
    """写入 JSON Lines，便于逐日复盘和后续脚本读取。"""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(json_safe(record), ensure_ascii=False))
            file.write("\n")
    os.replace(tmp_path, path)


def build_cluster_decisions(
    cluster_ids: Sequence[int],
    predicted_cluster_returns: np.ndarray,
    stock_codes: Sequence[str],
    cluster_labels: np.ndarray,
    returns_row: pd.Series,
    selected_clusters: Sequence[int],
    realized_by_id: Dict[int, float],
) -> List[Dict]:
    """整理每个簇的预测、真实收益和是否入选。"""
    selected_cluster_set = {int(cluster_id) for cluster_id in selected_clusters}
    decisions = []
    ranked_indices = np.argsort(predicted_cluster_returns)[::-1]
    for rank, cluster_index in enumerate(ranked_indices, start=1):
        cluster_id = int(cluster_ids[cluster_index])
        member_codes = [
            code for code, label in zip(stock_codes, cluster_labels) if int(label) == cluster_id
        ]
        valid_count = int(returns_row.reindex(member_codes).notna().sum())
        decisions.append(
            {
                "rank": rank,
                "cluster_id": cluster_id,
                "predicted_return": float(predicted_cluster_returns[cluster_index]),
                "realized_return": realized_by_id.get(cluster_id),
                "member_count": int(len(member_codes)),
                "valid_return_count": valid_count,
                "selected": cluster_id in selected_cluster_set,
            }
        )
    return decisions


def build_selected_stock_details(
    selected_stocks: Sequence[str],
    stock_codes: Sequence[str],
    cluster_labels: np.ndarray,
    returns_row: pd.Series,
) -> List[Dict]:
    """整理入选股票所属簇和预测日真实收益，方便复查具体持仓。"""
    stock_to_position = {code: position for position, code in enumerate(stock_codes)}
    details = []
    for code in selected_stocks:
        position = stock_to_position.get(code)
        daily_return = returns_row.get(code, np.nan)
        details.append(
            {
                "ts_code": code,
                "cluster_id": int(cluster_labels[position]) if position is not None else None,
                "daily_return": float(daily_return) if np.isfinite(daily_return) else None,
            }
        )
    return details


def run_backtest(
    data_dir: str = "data",
    model_path: str = "best_model.pt",
    lookback: int = 60,
    train_window: int = 20,
    top_neighbor_count: int = 0,
    cluster_count: int = 20,
    top_k_clusters: int = 3,
    seed_value: int = 42,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = "outputs/cluster_backtest",
    min_cluster_valid_count: int = 5,
    min_portfolio_valid_stocks: int = 50,
    target_portfolio_valid_stocks: int = 100,
    min_market_valid_stocks: int = 1000,
    predictor_epochs: int = 3,
    kmeans_n_init: int = 1,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """运行簇级轮动回测。

    当前版本的核心特点：
    - GNN embedding 模型固定加载 `best_model.pt`
    - 每个回测日重新训练一个簇收益预测器
    - 每个回测日重新 KMeans 聚类
    - 按预测收益排序逐簇加入，直到有效股票数达到目标
    - 默认关闭相关性边，以优先保证运行速度和结果可复现性
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed_value)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 基础数据：收益矩阵、行业/概念映射、预训练 GNN embedding 模型。
    returns = load_returns_csv(os.path.join(data_dir, "daily_returns.csv"))
    stock_codes = list(returns.columns)
    industry_df = load_industry_mapping(os.path.join(data_dir, "industry_mapping.csv"), stock_codes)
    concept_df = load_concept_mapping(data_dir, stock_codes)
    embedding_model = load_embedding_model(model_path, in_channels=6, hidden_channels=64, num_relations=3, device=device)

    selected_records = []
    daily_returns = []
    cluster_ics = []
    embedding_cache: Dict[Tuple[int, int], np.ndarray] = {}
    cluster_cache: Dict[Tuple[int, int, int, int, int], Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    positions = date_positions(
        returns,
        lookback,
        train_window,
        min_market_valid_stocks=min_market_valid_stocks,
        start_date=start_date,
        end_date=end_date,
    )
    run_parameters = {
        "data_dir": data_dir,
        "model_path": model_path,
        "lookback": lookback,
        "train_window": train_window,
        "top_neighbor_count": top_neighbor_count,
        "cluster_count": cluster_count,
        "top_k_clusters": top_k_clusters,
        "top_k_clusters_note": "保留兼容参数；当前选股使用 target_portfolio_valid_stocks 逐簇加入。",
        "seed_value": seed_value,
        "start_date": start_date,
        "end_date": end_date,
        "out_dir": out_dir,
        "min_cluster_valid_count": min_cluster_valid_count,
        "min_portfolio_valid_stocks": min_portfolio_valid_stocks,
        "target_portfolio_valid_stocks": target_portfolio_valid_stocks,
        "min_market_valid_stocks": min_market_valid_stocks,
        "predictor_epochs": predictor_epochs,
        "kmeans_n_init": kmeans_n_init,
        "device": str(device),
    }
    run_context = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": run_parameters,
        "data": {
            "returns_path": os.path.join(data_dir, "daily_returns.csv"),
            "stock_count": int(len(stock_codes)),
            "date_count": int(len(returns)),
            "data_start_date": str(returns.index.min().date()) if len(returns) else None,
            "data_end_date": str(returns.index.max().date()) if len(returns) else None,
        },
        "backtest_dates": {
            "count": int(len(positions)),
            "start": str(returns.index[positions[0]].date()) if positions else None,
            "end": str(returns.index[positions[-1]].date()) if positions else None,
        },
        "output_files": {
            "daily_returns_csv": "daily_returns.csv",
            "daily_decisions_jsonl": "daily_decisions.jsonl",
            "run_config_json": "run_config.json",
            "run_summary_json": "run_summary.json",
            "metrics_json": "metrics.json",
        },
    }
    write_json(os.path.join(out_dir, "run_config.json"), run_context)

    print(
        f"[状态] 可回测日期数={len(positions)}，"
        f"min_market_valid_stocks={min_market_valid_stocks}，"
        f"target_portfolio_valid_stocks={target_portfolio_valid_stocks}"
    )

    for day_number, day_index in enumerate(positions, start=1):
        day_start_time = time.perf_counter()
        trade_date = returns.index[day_index]
        market_valid_count = int(returns.iloc[day_index].notna().sum())
        print(
            f"[状态] 回测日期 {trade_date.date()} (index={day_index}, "
            f"{day_number}/{len(positions)}, market_valid={market_valid_count})"
        )
        # 第一步：用过去 train_window 天训练当天的簇收益预测器。
        predictor = train_predictor_for_day(
            embedding_model,
            returns,
            industry_df,
            concept_df,
            stock_codes,
            day_index,
            lookback,
            train_window,
            top_neighbor_count,
            cluster_count,
            seed_value,
            device,
            min_cluster_valid_count,
            predictor_epochs,
            kmeans_n_init,
            embedding_cache,
            cluster_cache,
        )
        if predictor is None:
            print("[警告] 历史训练样本为空，跳过该日")
            continue

        # 第二步：读取或计算当前日聚类结果，并预测每个簇的收益。
        labels, cluster_features, cluster_ids = get_day_clusters(
            cluster_cache,
            embedding_cache,
            embedding_model,
            returns,
            industry_df,
            concept_df,
            stock_codes,
            day_index,
            lookback,
            top_neighbor_count,
            cluster_count,
            seed_value,
            kmeans_n_init,
            device,
        )
        predicted_cluster_returns = predict_cluster_returns(predictor, cluster_features, device)

        # 第四步：按预测收益从高到低逐簇加入，直到组合有效股票数达到目标。
        selected_clusters, selected_stocks, num_valid_returns = select_clusters_until_target_valid_stocks(
            cluster_ids,
            predicted_cluster_returns,
            stock_codes,
            labels,
            returns.iloc[day_index],
            target_valid_stocks=target_portfolio_valid_stocks,
        )
        portfolio_return = compute_portfolio_return(
            returns.iloc[day_index],
            selected_stocks,
            min_valid_stocks=min_portfolio_valid_stocks,
        )

        # 用当前日真实收益计算簇级 IC，仅用于评估预测质量，不参与选股。
        returns_array = returns.iloc[day_index].to_numpy(dtype=float)
        realized_by_id = {}
        for cluster_id in cluster_ids:
            member_indices = np.flatnonzero(labels == cluster_id)
            valid_returns = returns_array[member_indices]
            valid_returns = valid_returns[np.isfinite(valid_returns)]
            if valid_returns.size >= min_cluster_valid_count:
                realized_by_id[int(cluster_id)] = float(valid_returns.mean())
        aligned_realized = np.array([realized_by_id.get(int(cluster_id), np.nan) for cluster_id in cluster_ids])
        cluster_ic = evaluate_prediction_ic(predicted_cluster_returns, aligned_realized)
        cluster_decisions = build_cluster_decisions(
            cluster_ids,
            predicted_cluster_returns,
            stock_codes,
            labels,
            returns.iloc[day_index],
            selected_clusters,
            realized_by_id,
        )
        selected_stock_details = build_selected_stock_details(
            selected_stocks,
            stock_codes,
            labels,
            returns.iloc[day_index],
        )

        daily_returns.append(portfolio_return)
        cluster_ics.append(cluster_ic)
        # 记录每日选择结果，后续可用于分析簇轮动、选股数量和组合收益。
        selected_records.append(
            {
                "date": str(trade_date.date()),
                "daily_return": portfolio_return,
                "num_stocks": int(len(selected_stocks)),
                "num_valid_returns": int(num_valid_returns),
                "selected_clusters": selected_clusters,
                "selected_stocks": selected_stocks,
                "cluster_decisions": cluster_decisions,
                "selected_stock_details": selected_stock_details,
            }
        )
        print(
            f"[状态] 日期 {trade_date.date()} 选股数={len(selected_stocks)} "
            f"有效股票数={num_valid_returns} 组合日收益={portfolio_return:.6f} "
            f"cluster_ic={cluster_ic} 耗时={time.perf_counter() - day_start_time:.1f}s"
        )

        if len(selected_records) % 10 == 0:
            # 长回测时保留中间结果，避免运行中断后完全没有输出。
            partial_df = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"cluster_decisions", "selected_stock_details"}
                    }
                    for record in selected_records
                ]
            )
            partial_df.to_csv(os.path.join(out_dir, "daily_returns_partial.csv"), index=False)
            write_jsonl(os.path.join(out_dir, "daily_decisions_partial.jsonl"), selected_records)

    # 回测结束后保存完整逐日结果和汇总指标。
    result_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"cluster_decisions", "selected_stock_details"}
            }
            for record in selected_records
        ]
    )
    result_df.to_csv(os.path.join(out_dir, "daily_returns.csv"), index=False)
    write_jsonl(os.path.join(out_dir, "daily_decisions.jsonl"), selected_records)
    export_backtest_plots(result_df, out_dir)

    metrics = compute_metrics(daily_returns)
    finite_ics = [value for value in cluster_ics if np.isfinite(value)]
    metrics["mean_cluster_ic"] = float(np.mean(finite_ics)) if finite_ics else None
    with open(os.path.join(out_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)
    run_summary = dict(run_context)
    run_summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_summary["metrics"] = metrics
    write_json(os.path.join(out_dir, "run_summary.json"), run_summary)

    print(f"[状态] 回测完成，结果已保存到 {out_dir}/")
    return result_df, metrics


def run_daily_cluster_backtest(**kwargs) -> pd.DataFrame:
    """兼容 notebook 或外部脚本调用：只返回逐日回测结果。"""
    result_df, _ = run_backtest(**kwargs)
    return result_df


if __name__ == "__main__":
    _, backtest_metrics = run_backtest()
    print(backtest_metrics)
