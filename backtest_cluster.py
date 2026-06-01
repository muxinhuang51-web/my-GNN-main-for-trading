import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from models.cluster_predictor import (
    ClusterReturnPredictor,
    build_cluster_feature_samples,
    build_cluster_samples,
    predict_cluster_returns,
    train_cluster_predictor,
)
from models.embedding_model import extract_embeddings, load_embedding_model


def set_seed(seed_value: int) -> None:
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def load_returns_csv(path: str) -> pd.DataFrame:
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"行业映射不存在: {path}")
    dataframe = pd.read_csv(path)
    dataframe["ts_code"] = dataframe["ts_code"].astype(str)
    return dataframe[dataframe["ts_code"].isin(stock_codes)].copy()


def load_concept_mapping(data_dir: str, stock_codes: Sequence[str]) -> pd.DataFrame:
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
    mean = dataframe.mean(axis=0)
    std = dataframe.std(axis=0).replace(0, np.nan)
    return (dataframe - mean) / std


def build_features_from_window(window_dataframe: pd.DataFrame) -> pd.DataFrame:
    momentum_20 = (1 + window_dataframe.tail(20)).prod() - 1
    mean_5 = window_dataframe.tail(5).mean()
    mean_10 = window_dataframe.tail(10).mean()
    volatility_20 = window_dataframe.tail(20).std(ddof=0)
    volatility_60 = window_dataframe.tail(60).std(ddof=0)
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
    return {code: index for index, code in enumerate(stock_codes)}


def build_group_edges(
    mapping_df: pd.DataFrame,
    stock2index: Dict[str, int],
    stock_column: str,
    group_column: str,
    max_neighbors: int = 20,
) -> Optional[np.ndarray]:
    if mapping_df.empty or stock_column not in mapping_df.columns or group_column not in mapping_df.columns:
        return None

    edge_pairs = set()
    grouped = mapping_df.groupby(group_column)[stock_column].apply(list).to_dict()
    for members in grouped.values():
        members = sorted({member for member in members if member in stock2index}, key=stock2index.get)
        for source in members:
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
            edge_pairs.add((source_index, int(target_index)))

    if not edge_pairs:
        return None
    return np.array(sorted(edge_pairs), dtype=np.int64).T


def tensor_from_edges(edge_index: Optional[np.ndarray]) -> torch.Tensor:
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
    if time_index < lookback:
        raise ValueError("time_index 小于 lookback，无法构造历史窗口")

    window = returns.iloc[time_index - lookback:time_index]
    feature_frame = build_features_from_window(window).reindex(stock_codes).fillna(0.0)
    stock2index = build_stock2index(stock_codes)

    industry_edge = tensor_from_edges(
        build_group_edges(industry_df, stock2index, "ts_code", "industry", max_neighbors=20)
    )
    concept_edge = tensor_from_edges(
        build_group_edges(concept_df, stock2index, "ts_code", "concept", max_neighbors=20)
    )
    corr_edge = tensor_from_edges(
        build_corr_edges(window.reindex(columns=stock_codes), top_neighbor_count=top_neighbor_count)
    )

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
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[int]:
    start_index = lookback + train_window
    valid_positions = list(range(start_index, len(returns)))
    if start_date is not None:
        start = pd.to_datetime(start_date)
        valid_positions = [index for index in valid_positions if returns.index[index] >= start]
    if end_date is not None:
        end = pd.to_datetime(end_date)
        valid_positions = [index for index in valid_positions if returns.index[index] <= end]
    return valid_positions


def compute_portfolio_return(returns_row: pd.Series, selected_stocks: Sequence[str], min_valid_stocks: int = 1) -> float:
    if not selected_stocks:
        return 0.0
    valid_returns = returns_row.reindex(selected_stocks).dropna()
    if len(valid_returns) < min_valid_stocks:
        return float("nan")
    return float(valid_returns.mean())


def compute_metrics(daily_returns: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"annualized_return": None, "sharpe": None, "max_drawdown": None, "days": 0}

    cumulative = np.prod(1 + arr)
    years = len(arr) / 252.0
    annualized_return = cumulative ** (1 / years) - 1 if years > 0 else np.nan
    mean_daily = arr.mean()
    std_daily = arr.std(ddof=1) if len(arr) > 1 else 0.0
    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else np.nan
    wealth = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak

    return {
        "annualized_return": float(annualized_return) if np.isfinite(annualized_return) else None,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "max_drawdown": float(np.min(drawdown)),
        "days": int(len(arr)),
    }


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
) -> Optional[ClusterReturnPredictor]:
    feature_batches = []
    label_batches = []
    for train_index in range(day_index - train_window, day_index):
        try:
            data = build_data_for_index(
                returns, industry_df, concept_df, stock_codes, train_index, lookback, top_neighbor_count
            )
            embeddings = extract_embeddings(model, data, device)
            labels = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=10).fit_predict(embeddings)
            features, cluster_returns, _ = build_cluster_samples(
                embeddings,
                labels,
                returns.iloc[train_index].to_numpy(dtype=float),
                min_valid_count=min_cluster_valid_count,
            )
        except ValueError as error:
            print(f"[警告] 跳过训练日 {train_index}: {error}")
            continue
        feature_batches.append(features)
        label_batches.append(cluster_returns)

    if not feature_batches:
        return None

    x_train = np.vstack(feature_batches)
    y_train = np.concatenate(label_batches)
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
    mask = np.isfinite(predictions) & np.isfinite(realized_cluster_returns)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(predictions[mask], realized_cluster_returns[mask])[0, 1])


def run_backtest(
    data_dir: str = "data",
    model_path: str = "best_model.pt",
    lookback: int = 60,
    train_window: int = 60,
    top_neighbor_count: int = 20,
    cluster_count: int = 20,
    top_k_clusters: int = 3,
    seed_value: int = 42,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = "outputs/cluster_backtest",
    min_cluster_valid_count: int = 3,
    min_portfolio_valid_stocks: int = 1,
    predictor_epochs: int = 20,
    device: Optional[torch.device] = None,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed_value)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    returns = load_returns_csv(os.path.join(data_dir, "daily_returns.csv"))
    stock_codes = list(returns.columns)
    industry_df = load_industry_mapping(os.path.join(data_dir, "industry_mapping.csv"), stock_codes)
    concept_df = load_concept_mapping(data_dir, stock_codes)
    embedding_model = load_embedding_model(model_path, in_channels=6, hidden_channels=64, num_relations=3, device=device)

    selected_records = []
    daily_returns = []
    cluster_ics = []

    for day_index in date_positions(returns, lookback, train_window, start_date, end_date):
        trade_date = returns.index[day_index]
        print(f"[状态] 回测日期 {trade_date.date()} (index={day_index})")
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
        )
        if predictor is None:
            print("[警告] 历史训练样本为空，跳过该日")
            continue

        data = build_data_for_index(
            returns, industry_df, concept_df, stock_codes, day_index, lookback, top_neighbor_count
        )
        embeddings = extract_embeddings(embedding_model, data, device)
        labels = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=10).fit_predict(embeddings)
        cluster_features, cluster_ids = build_cluster_feature_samples(embeddings, labels)
        predicted_cluster_returns = predict_cluster_returns(predictor, cluster_features, device)

        top_count = min(max(1, top_k_clusters), len(cluster_ids))
        top_indices = np.argsort(predicted_cluster_returns)[-top_count:][::-1]
        selected_clusters = {cluster_ids[index] for index in top_indices}
        selected_stocks = [
            code for code, cluster_label in zip(stock_codes, labels) if int(cluster_label) in selected_clusters
        ]
        portfolio_return = compute_portfolio_return(
            returns.iloc[day_index],
            selected_stocks,
            min_valid_stocks=min_portfolio_valid_stocks,
        )

        try:
            _, realized_cluster_returns, realized_cluster_ids = build_cluster_samples(
                embeddings,
                labels,
                returns.iloc[day_index].to_numpy(dtype=float),
                min_valid_count=min_cluster_valid_count,
            )
            realized_by_id = dict(zip(realized_cluster_ids, realized_cluster_returns))
            aligned_realized = np.array([realized_by_id.get(cluster_id, np.nan) for cluster_id in cluster_ids])
            cluster_ic = evaluate_prediction_ic(predicted_cluster_returns, aligned_realized)
        except ValueError:
            cluster_ic = float("nan")

        daily_returns.append(portfolio_return)
        cluster_ics.append(cluster_ic)
        selected_records.append(
            {
                "date": str(trade_date.date()),
                "daily_return": portfolio_return,
                "num_stocks": int(len(selected_stocks)),
                "num_valid_returns": int(returns.iloc[day_index].reindex(selected_stocks).notna().sum()),
                "selected_clusters": sorted(selected_clusters),
                "selected_stocks": selected_stocks,
            }
        )
        print(
            f"[状态] 日期 {trade_date.date()} 选股数={len(selected_stocks)} "
            f"组合日收益={portfolio_return:.6f} cluster_ic={cluster_ic}"
        )

        if len(selected_records) % 10 == 0:
            pd.DataFrame(selected_records).to_csv(os.path.join(out_dir, "daily_returns_partial.csv"), index=False)

    result_df = pd.DataFrame(selected_records)
    result_df.to_csv(os.path.join(out_dir, "daily_returns.csv"), index=False)

    metrics = compute_metrics(daily_returns)
    finite_ics = [value for value in cluster_ics if np.isfinite(value)]
    metrics["mean_cluster_ic"] = float(np.mean(finite_ics)) if finite_ics else None
    with open(os.path.join(out_dir, "metrics.json"), "w") as file:
        json.dump(metrics, file, indent=2)

    print(f"[状态] 回测完成，结果已保存到 {out_dir}/")
    return result_df, metrics


def run_daily_cluster_backtest(**kwargs) -> pd.DataFrame:
    result_df, _ = run_backtest(**kwargs)
    return result_df


if __name__ == "__main__":
    _, backtest_metrics = run_backtest()
    print(backtest_metrics)
