import torch
import numpy as np

def split_data(edge_index, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    将数据集划分为训练集、验证集和测试集
    Args:
        edge_index: 边的索引张量 [3, num_edges]
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    Returns:
        train_edges, val_edges, test_edges: 划分后的数据集
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    torch.manual_seed(seed)
    
    # 获取边的数量
    num_edges = edge_index.size(1)
    
    # 生成随机排列
    perm = torch.randperm(num_edges)
    
    # 计算每个集合的大小
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    
    # 划分数据
    train_indices = perm[:train_size]
    val_indices = perm[train_size:train_size + val_size]
    test_indices = perm[train_size + val_size:]
    
    # 获取对应的边
    train_edges = edge_index[:, train_indices]
    val_edges = edge_index[:, val_indices]
    test_edges = edge_index[:, test_indices]
    
    return train_edges, val_edges, test_edges