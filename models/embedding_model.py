import os

import numpy as np
import torch
import torch.nn.functional as functional
from torch_geometric.nn import RGCNConv


class StockRGCNEmbedding(torch.nn.Module):
    """带嵌入导出的 RGCN 回归模型。"""

    def __init__(self, in_channels: int, hidden_channels: int, num_relations: int):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)  # 第一层关系卷积
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)  # 第二层关系卷积
        self.regressor = torch.nn.Linear(hidden_channels, 1)  # 线性回归头

    def forward(self, node_features, edge_index, edge_type, return_embedding: bool = False):
        hidden = self.conv1(node_features, edge_index, edge_type)  # 隐层表示
        hidden = functional.relu(hidden)  # 非线性激活
        embedding = self.conv2(hidden, edge_index, edge_type)  # 最终嵌入
        if return_embedding:
            return embedding
        output = self.regressor(embedding).squeeze(-1)  # 回归输出
        return output


def load_embedding_model(model_path: str, in_channels: int, hidden_channels: int, num_relations: int, device: torch.device):
    """加载模型并返回可导出嵌入的实例。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError("模型文件不存在，请检查 best_model.pt")
    print("[状态] 正在加载嵌入模型...")
    model = StockRGCNEmbedding(in_channels, hidden_channels, num_relations).to(device)  # 模型实例
    state_dict = torch.load(model_path, map_location=device)  # 权重字典
    model.load_state_dict(state_dict)  # 权重加载
    model.eval()
    print("[状态] 模型加载完成")
    return model


def extract_embeddings(model: StockRGCNEmbedding, data, device: torch.device) -> np.ndarray:
    """基于图数据提取节点嵌入。"""
    print("[状态] 开始导出嵌入...")
    # 比喻注释：像把每只股票的“指纹”从模型里取出来做档案
    model.eval()
    with torch.no_grad():
        embedding = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_type.to(device),
            return_embedding=True,
        )
    print("[状态] 嵌入导出完成")
    return embedding.detach().cpu().numpy()
