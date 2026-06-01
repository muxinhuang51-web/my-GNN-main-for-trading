import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans

from models.embedding_model import load_embedding_model, extract_embeddings
from models.cluster_predictor import build_cluster_samples, ClusterReturnPredictor, train_cluster_predictor, predict_cluster_returns


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    print("[状态] 随机种子已设置")


def load_returns_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("收益率文件不存在")
    df = pd.read_csv(path)
    date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.columns = [str(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def run_daily_cluster_backtest(
    data_dir="data",
    model_path="best_model.pt",
    lookback=60,
    top_neighbor_count=20,
    cluster_count=20,
    seed_value=42,
    device=None,
    out_dir="outputs/cluster_backtest",
    start_date=None,
    end_date=None,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed_value)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    returns = load_returns_csv(os.path.join(data_dir, "daily_returns.csv"))
    stock_codes = list(returns.columns)
    # minimal industry/concept loaders from notebooks
    industry_df = pd.read_csv(os.path.join(data_dir, "industry_mapping.csv"))
    industry_df["ts_code"] = industry_df["ts_code"].astype(str)
    concept_candidates = [os.path.join(data_dir, "stock_concept.csv"), os.path.join(data_dir, "concept_mapping.csv"), os.path.join(data_dir, "stock_concepts.csv")]
    concept_path = next((p for p in concept_candidates if os.path.exists(p)), None)
    concept_df = pd.read_csv(concept_path) if concept_path else pd.DataFrame(columns=["ts_code", "concept"])

    # date slicing
    dates = returns.index.dropna()
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
    results = []  # daily returns

    model = load_embedding_model(model_path, in_channels=6, hidden_channels=64, num_relations=3, device=device)

    for i, date in enumerate(dates):
        if i < lookback:
            continue
        print(f"[状态] 回测日期 {date.date()} (index={i})")
        window = returns.iloc[i - lookback:i]
        # build features (reuse code from notebook)
        momentum_20 = (1 + window.tail(20)).prod() - 1
        mean_5 = window.tail(5).mean()
        mean_10 = window.tail(10).mean()
        vol20 = window.tail(20).std(ddof=0)
        vol60 = window.tail(60).std(ddof=0)
        last_ret = window.tail(1).iloc[0]
        feature_frame = pd.DataFrame({
            "mom20": momentum_20,
            "mean5": mean_5,
            "mean10": mean_10,
            "vol20": vol20,
            "vol60": vol60,
            "last_ret": last_ret,
        })
        feature_frame = (feature_frame - feature_frame.mean()) / feature_frame.std().replace(0, np.nan)
        feature_frame = feature_frame.reindex(stock_codes).fillna(0.0)
        # build simple edges (industry + concept)
        stock2index = {code: idx for idx, code in enumerate(stock_codes)}
        def build_group_edges(mapping_df, stock2index, stock_column, group_column, max_neighbors=20):
            edge_pairs = set()
            grouped = mapping_df.groupby(group_column)[stock_column].apply(list).to_dict()
            for members in grouped.values():
                members = [c for c in members if c in stock2index]
                for source in members:
                    peers = [c for c in members if c != source]
                    if len(peers) > max_neighbors:
                        peers = peers[:max_neighbors]
                    for target in peers:
                        edge_pairs.add((stock2index[source], stock2index[target]))
            if not edge_pairs:
                return torch.empty((2, 0), dtype=torch.long)
            return torch.tensor(sorted(edge_pairs), dtype=torch.long).t().contiguous()
        industry_edge = build_group_edges(industry_df, stock2index, "ts_code", "industry")
        concept_edge = build_group_edges(concept_df, stock2index, "ts_code", "concept")
        # no corr edges for speed
        edge_index = torch.cat([industry_edge, concept_edge], dim=1) if industry_edge.size(1) + concept_edge.size(1) > 0 else torch.empty((2,0), dtype=torch.long)
        edge_type = torch.cat([
            torch.zeros(industry_edge.size(1), dtype=torch.long) if industry_edge.size(1)>0 else torch.empty((0,), dtype=torch.long),
            torch.ones(concept_edge.size(1), dtype=torch.long) if concept_edge.size(1)>0 else torch.empty((0,), dtype=torch.long),
        ]) if edge_index.size(1)>0 else torch.empty((0,), dtype=torch.long)
        data = Data(x=torch.tensor(feature_frame.to_numpy(dtype=float), dtype=torch.float), edge_index=edge_index, edge_type=edge_type)
        embeddings = extract_embeddings(model, data, device)
        # clustering
        kmeans = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        # build cluster samples
        next_return = returns.iloc[i].to_numpy(dtype=float)
        try:
            cluster_features, cluster_labels_mean, cluster_id_list = build_cluster_samples(embeddings, cluster_labels, next_return)
        except ValueError as e:
            print(f"[警告] 构造簇级样本失败：{e}")
            continue
        predictor = ClusterReturnPredictor(input_dim=cluster_features.shape[1], hidden_dim=32, dropout_rate=0.1)
        predictor = train_cluster_predictor(predictor, cluster_features, cluster_labels_mean, device, epochs=10, batch_size=16, learning_rate=1e-3)
        cluster_preds = predict_cluster_returns(predictor, cluster_features, device)
        # select top cluster(s)
        top_k = max(1, int(cluster_count * 0.1))
        top_idx = np.argsort(cluster_preds)[-top_k:][::-1]
        selected_clusters = [cluster_id_list[idx] for idx in top_idx]
        selected_stocks = [code for code, label in zip(stock_codes, cluster_labels) if label in selected_clusters]
        # compute portfolio return: equal weight across selected stocks
        if not selected_stocks:
            daily_ret = 0.0
        else:
            sel_idx = [stock_codes.index(c) for c in selected_stocks]
            stock_next = returns.iloc[i].to_numpy(dtype=float)
            daily_ret = np.nanmean(stock_next[sel_idx])
        results.append({"date": date, "daily_return": daily_ret, "num_stocks": len(selected_stocks)})
        if (i - lookback) % 10 == 0:
            # checkpoint outputs
            pd.DataFrame(results).to_csv(os.path.join(out_dir, "daily_returns_partial.csv"), index=False)
    # finalize
    result_df = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, "daily_returns.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"[状态] 回测完成，输出 -> {out_csv}")
    return result_df


if __name__ == "__main__":
    run_daily_cluster_backtest()
import os
import json
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from models.embedding_model import load_embedding_model, extract_embeddings
from models.cluster_predictor import (
    build_cluster_samples,
    ClusterReturnPredictor,
    train_cluster_predictor,
    predict_cluster_returns,
)


def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


def load_returns_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"收益率文件不存在: {path}")
    df = pd.read_csv(path)
    date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    returns = df.drop(columns=[date_col], errors="ignore")
    returns.columns = [str(c) for c in returns.columns]
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns.index = dates
    return returns


def load_industry_mapping(path: str, stock_codes: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"行业映射不存在: {path}")
    df = pd.read_csv(path)
    df["ts_code"] = df["ts_code"].astype(str)
    df = df[df["ts_code"].isin(stock_codes)].copy()
    return df


def load_concept_mapping(data_dir: str, stock_codes: List[str]) -> pd.DataFrame:
    candidates = [
        os.path.join(data_dir, "stock_concept.csv"),
        os.path.join(data_dir, "concept_mapping.csv"),
        os.path.join(data_dir, "stock_concepts.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return pd.DataFrame(columns=["ts_code", "concept"])
    df = pd.read_csv(path)
    if "ts_code" not in df.columns:
        return pd.DataFrame(columns=["ts_code", "concept"])
    concept_col = next((c for c in df.columns if c != "ts_code"), None)
    if concept_col is None:
        return pd.DataFrame(columns=["ts_code", "concept"])
    df = df[["ts_code", concept_col]].copy()
    df.columns = ["ts_code", "concept"]
    df["ts_code"] = df["ts_code"].astype(str)
    df = df[df["ts_code"].isin(stock_codes)].copy()
    return df


def zscore_by_column(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=0)
    std = df.std(axis=0).replace(0, np.nan)
    return (df - mean) / std


def build_features_from_window(window_dataframe: pd.DataFrame) -> pd.DataFrame:
    momentum_20 = (1 + window_dataframe.tail(20)).prod() - 1
    mean_5 = window_dataframe.tail(5).mean()
    mean_10 = window_dataframe.tail(10).mean()
    vol20 = window_dataframe.tail(20).std(ddof=0)
    vol60 = window_dataframe.tail(60).std(ddof=0)
    last = window_dataframe.tail(1).iloc[0]
    feature_frame = pd.DataFrame({
        "mom20": momentum_20,
        "mean5": mean_5,
        "mean10": mean_10,
        "vol20": vol20,
        "vol60": vol60,
        "last_ret": last,
    })
    return zscore_by_column(feature_frame)


def build_stock2index(stock_codes: List[str]) -> dict:
    return {code: idx for idx, code in enumerate(stock_codes)}


def build_group_edges(mapping_df: pd.DataFrame, stock2index: dict, stock_column: str, group_column: str, max_neighbors: int = 20):
    edge_pairs = set()
    grouped = mapping_df.groupby(group_column)[stock_column].apply(list).to_dict()
    for members in grouped.values():
        members = [c for c in members if c in stock2index]
        for source in members:
            peers = [c for c in members if c != source]
            if len(peers) > max_neighbors:
                peers = peers[:max_neighbors]
            for target in peers:
                edge_pairs.add((stock2index[source], stock2index[target]))
    if not edge_pairs:
        return None
    edge_index = np.array(sorted(edge_pairs)).T
    return edge_index


def build_corr_edges(window_dataframe: pd.DataFrame, top_neighbor_count: int = 20):
    values = window_dataframe.to_numpy(dtype=float)
    if values.shape[0] < 5:
        return None
    corr = np.corrcoef(values.T)
    edge_pairs = set()
    for i in range(corr.shape[0]):
        row = corr[i].copy()
        row[i] = -np.inf
        idx = np.argsort(np.abs(row))[-top_neighbor_count:]
        for j in idx:
            if np.isfinite(row[j]):
                edge_pairs.add((i, j))
    if not edge_pairs:
        return None
    return np.array(sorted(edge_pairs)).T


def build_data_for_index(returns: pd.DataFrame, industry_df: pd.DataFrame, concept_df: pd.DataFrame, stock_codes: List[str], time_index: int, lookback: int, top_neighbor_count: int):
    window = returns.iloc[time_index - lookback:time_index]
    feature_frame = build_features_from_window(window).reindex(stock_codes).fillna(0.0)
    stock2index = build_stock2index(stock_codes)
    industry_edge = build_group_edges(industry_df, stock2index, "ts_code", "industry", max_neighbors=20)
    concept_edge = build_group_edges(concept_df, stock2index, "ts_code", "concept", max_neighbors=20)
    corr_edge = build_corr_edges(window, top_neighbor_count=top_neighbor_count)
    # build a simple container to feed embedding extractor (only the attributes used)
    class SimpleData:
        pass

    data = SimpleData()
    data.x = torch.tensor(feature_frame.to_numpy(dtype=float), dtype=torch.float)
    # pack edges as tensors expected by embedding extractor
    edges = []
    types = []
    if industry_edge is not None:
        edges.append(torch.tensor(industry_edge, dtype=torch.long))
        types.append(torch.zeros(industry_edge.shape[1], dtype=torch.long))
    if concept_edge is not None:
        edges.append(torch.tensor(concept_edge, dtype=torch.long))
        types.append(torch.ones(concept_edge.shape[1], dtype=torch.long))
    if corr_edge is not None:
        edges.append(torch.tensor(corr_edge, dtype=torch.long))
        types.append(torch.full((corr_edge.shape[1],), 2, dtype=torch.long))
    if edges:
        data.edge_index = torch.cat(edges, dim=1).contiguous()
        data.edge_type = torch.cat(types, dim=0).contiguous()
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.edge_type = torch.empty((0,), dtype=torch.long)
    return data


def compute_metrics(daily_returns: List[float]):
    arr = np.array(daily_returns)
    if len(arr) == 0:
        return {}
    # geometric annualized return
    cumulative = np.prod(1 + arr)
    years = len(arr) / 252.0
    ann_return = cumulative ** (1 / years) - 1 if years > 0 else float("nan")
    mean_daily = arr.mean()
    std_daily = arr.std(ddof=1) if len(arr) > 1 else 0.0
    sharpe = (mean_daily * 252) / (std_daily * np.sqrt(252)) if std_daily > 0 else float("nan")
    # max drawdown
    wealth = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / peak
    max_dd = float(np.min(drawdown))
    return {
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
        "max_drawdown": float(max_dd),
        "days": int(len(arr)),
    }


def run_backtest(
    data_dir: str = "data",
    model_path: str = "best_model.pt",
    lookback: int = 60,
    train_window: int = 60,
    cluster_count: int = 20,
    top_k_clusters: int = 3,
    seed_value: int = 42,
):
    set_seed(seed_value)
    returns = load_returns_csv(os.path.join(data_dir, "daily_returns.csv"))
    stock_codes = list(returns.columns)
    industry_df = load_industry_mapping(os.path.join(data_dir, "industry_mapping.csv"), stock_codes)
    concept_df = load_concept_mapping(data_dir, stock_codes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_embedding_model(model_path, in_channels=6, hidden_channels=64, num_relations=3, device=device)

    start_i = lookback + train_window
    daily_portfolio_returns = []
    daily_cluster_ics = []
    selected_records = []

    for i in range(start_i, len(returns) - 1):
        print(f"[状态] 回测日期索引 {i}, 日期 {returns.index[i].date()}")
        # build training dataset from previous train_window days
        X_list = []
        y_list = []
        for j in range(i - train_window, i):
            try:
                data_j = build_data_for_index(returns, industry_df, concept_df, stock_codes, j, lookback, top_neighbor_count=20)
                emb_j = extract_embeddings(model, data_j, device)
                kmeans = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=10)
                labels_j = kmeans.fit_predict(emb_j)
                next_returns_j = returns.iloc[j].to_numpy(dtype=float)
                feat, lab, _ = build_cluster_samples(emb_j, labels_j, next_returns_j)
                X_list.append(feat)
                y_list.append(lab)
            except Exception as e:
                print(f"[警告] 构造训练日 {j} 时发生错误: {e}")
                continue
        if not X_list:
            print("[状态] 未生成任何训练簇样本，跳过该日")
            continue
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        predictor = ClusterReturnPredictor(input_dim=X_train.shape[1], hidden_dim=32, dropout_rate=0.1)
        predictor = train_cluster_predictor(predictor, X_train, y_train, device, epochs=20, batch_size=32, learning_rate=1e-3)

        # predict for day i
        data_i = build_data_for_index(returns, industry_df, concept_df, stock_codes, i, lookback, top_neighbor_count=20)
        emb_i = extract_embeddings(model, data_i, device)
        kmeans_i = KMeans(n_clusters=cluster_count, random_state=seed_value, n_init=10)
        labels_i = kmeans_i.fit_predict(emb_i)
        cluster_feats_i, cluster_labels_mean_i, cluster_id_list_i = build_cluster_samples(emb_i, labels_i, returns.iloc[i].to_numpy(dtype=float))
        pred_cluster_returns = predict_cluster_returns(predictor, cluster_feats_i, device)

        # compute cluster IC (pearson) between pred and realized cluster mean returns
        try:
            if len(pred_cluster_returns) > 1:
                ic = float(np.corrcoef(pred_cluster_returns, cluster_labels_mean_i)[0, 1])
            else:
                ic = float('nan')
        except Exception:
            ic = float('nan')
        daily_cluster_ics.append(ic)

        # select top-k clusters
        top_idx = np.argsort(pred_cluster_returns)[-top_k_clusters:][::-1]
        selected_clusters = [cluster_id_list_i[idx] for idx in top_idx]
        selected_stocks = [
            code for code, lab in zip(stock_codes, labels_i) if lab in selected_clusters
        ]
        # compute portfolio return as equal-weighted average of selected stocks' realized return at day i
        realized_returns_i = returns.iloc[i].reindex(selected_stocks).to_numpy(dtype=float)
        realized_returns_i = realized_returns_i[~np.isnan(realized_returns_i)]
        if realized_returns_i.size == 0:
            port_ret = 0.0
        else:
            port_ret = float(np.nanmean(realized_returns_i))
        daily_portfolio_returns.append(port_ret)
        selected_records.append({"date": str(returns.index[i].date()), "n_selected": int(len(selected_stocks)), "selected_stocks": selected_stocks})
        print(f"[状态] 日期 {returns.index[i].date()} 选股数={len(selected_stocks)} 组合日收益={port_ret:.6f} cluster_ic={ic}")

    # 保存结果
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(selected_records).to_csv(os.path.join("outputs", "selected_stocks_cluster.csv"), index=False)
    pd.DataFrame({"daily_return": daily_portfolio_returns}, index=None).to_csv(os.path.join("outputs", "daily_returns_cluster.csv"), index=False)
    metrics = compute_metrics(daily_portfolio_returns)
    metrics["mean_cluster_ic"] = float(np.nanmean([v for v in daily_cluster_ics if not np.isnan(v)])) if daily_cluster_ics else None
    with open(os.path.join("outputs", "metrics_cluster.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[状态] 回测完成，结果已保存到 outputs/")
    return metrics


if __name__ == "__main__":
    # 简单命令行执行入口
    metrics = run_backtest()
    print(metrics)
