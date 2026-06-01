from typing import Dict, List, Tuple

import numpy as np
import torch


class ClusterReturnPredictor(torch.nn.Module):
    """簇收益预测器（两层 MLP）。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)  # 第一层线性
        self.dropout = torch.nn.Dropout(dropout_rate)  # 随机失活
        self.linear2 = torch.nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, features):
        hidden = torch.relu(self.linear1(features))  # 隐层激活
        hidden = self.dropout(hidden)  # 防止过拟合
        return self.linear2(hidden).squeeze(-1)


def group_indices_by_label(labels: np.ndarray) -> Dict[int, List[int]]:
    """把样本索引按簇标签分组。"""
    index_map: Dict[int, List[int]] = {}
    for index_value, label_value in enumerate(labels):
        if label_value < 0:
            continue
        index_map.setdefault(int(label_value), []).append(index_value)
    return index_map


def mean_pool_vectors(vectors: np.ndarray, indices: List[int]) -> np.ndarray:
    """计算指定索引集合的均值向量。"""
    if not indices:
        raise ValueError("簇内样本为空，无法均值池化")
    return vectors[indices].mean(axis=0)


def mean_pool_labels(labels: np.ndarray, indices: List[int]) -> float:
    """计算指定索引集合的均值标签。"""
    label_values = labels[indices]
    if np.all(np.isnan(label_values)):
        return float("nan")
    return float(np.nanmean(label_values))


def build_cluster_samples(embeddings: np.ndarray, cluster_labels: np.ndarray, next_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """从股票级样本构造簇级样本。"""
    print("[状态] 开始构造簇级样本...")
    index_map = group_indices_by_label(cluster_labels)
    feature_list: List[np.ndarray] = []
    label_list: List[float] = []
    cluster_id_list: List[int] = []
    # 比喻注释：把同一簇的股票当成一篮水果，先打成果汁再拿去评判口感
    for cluster_id, indices in index_map.items():
        cluster_feature = mean_pool_vectors(embeddings, indices)
        cluster_label = mean_pool_labels(next_returns, indices)
        if np.isnan(cluster_label):
            continue
        feature_list.append(cluster_feature)
        label_list.append(cluster_label)
        cluster_id_list.append(cluster_id)
    if not feature_list:
        raise ValueError("簇级样本为空，请检查聚类标签与收益率")
    print("[状态] 簇级样本构造完成")
    return np.vstack(feature_list), np.array(label_list), cluster_id_list


def iterate_minibatches(features: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """按 batch 迭代样本。"""
    total = features.size(0)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield features[start:end], labels[start:end]


def train_one_epoch(model: ClusterReturnPredictor, feature_tensor: torch.Tensor, label_tensor: torch.Tensor, optimizer, loss_function, batch_size: int) -> float:
    """训练单个 epoch。"""
    model.train()
    epoch_loss = 0.0  # 训练损失累计
    for batch_features, batch_labels in iterate_minibatches(feature_tensor, label_tensor, batch_size):
        optimizer.zero_grad()
        prediction = model(batch_features)
        loss = loss_function(prediction, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
    return epoch_loss


def train_cluster_predictor(model: ClusterReturnPredictor, features: np.ndarray, labels: np.ndarray, device: torch.device, epochs: int = 50, batch_size: int = 64, learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> ClusterReturnPredictor:
    """训练簇级预测器。"""
    print("[状态] 开始训练簇级预测器...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 优化器
    loss_function = torch.nn.MSELoss()  # 均方误差
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)  # 特征张量
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)  # 标签张量
    for epoch_index in range(1, epochs + 1):
        epoch_loss = train_one_epoch(model, feature_tensor, label_tensor, optimizer, loss_function, batch_size)
        if epoch_index % max(1, epochs // 5) == 0:
            print(f"[状态] 训练进度 {epoch_index}/{epochs}，loss={epoch_loss:.6f}")
    print("[状态] 簇级预测器训练完成")
    return model


def predict_cluster_returns(model: ClusterReturnPredictor, features: np.ndarray, device: torch.device) -> np.ndarray:
    """预测簇级收益。"""
    print("[状态] 开始预测簇级收益...")
    model.eval()
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)  # 特征张量
    with torch.no_grad():
        prediction = model(feature_tensor).detach().cpu().numpy()
    print("[状态] 簇级收益预测完成")
    return prediction
