import torch

# edge_index: shape [2, num_edges]
# 例如有3个 user（编号0~2），4个 item（编号0~3）
edge_index = torch.tensor([
    [0, 1, 2],  # source nodes (e.g., users)
    [1, 2, 0]   # target nodes (e.g., items)
])

num_src = 3   # user 节点数量
num_dst = 4   # item 节点数量

# 创建一个 3x4 的稠密邻接矩阵
adj = torch.zeros((num_src, num_dst))

# 对应位置设为 1
adj[edge_index[0], edge_index[1]] = 1

print("Adjacency matrix:")
print(adj)