import pandas as pd
import torch

def load_node_csv(path, index_col):
    """
    Args:
        path (str): 数据集路径
        index_col (str): 数据集文件里的列索引
    Returns:
        dict: 列号和用户ID的索引、列好和电影ID的索引
    """
    df = pd.read_csv(path, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}  # enumerate()索引函数,默认索引从0开始
    return mapping

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=3):
    """
    Args:
        path (str): 数据集路径
        src_index_col (str): 用户列名
        src_mapping (dict): 行号和用户ID的映射
        dst_index_col (str): 电影列名
        dst_mapping (dict): 行号和电影ID的映射
        link_index_col (str): 交互的列名
        rating_threshold (int, optional): 决定选取多少评分交互的阈值，设置为4分
    Returns:
        torch.Tensor: 2*N的用户电影交互节点图
    """
    df = pd.read_csv(path)
    edge_index = None
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold  # 将数组转化为tensor张量
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
 
    return torch.tensor(edge_index)