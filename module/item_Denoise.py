import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import math
from torch_sparse import SparseTensor, matmul


class IDenoise(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device, k = 3):
        super(IDenoise, self).__init__()
        self.device = device
        self.dim = in_channels
        self.k = k

        self.encode = nn.Sequential(
            nn.Linear(in_channels*2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

        self.encode2 = nn.Sequential(
            nn.Linear(in_channels , hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

        self.decode = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.Dropout(p=0.2),
            nn.Tanh()
        )

        self.gat = GATConv(hidden_channels, hidden_channels)

    def cosine_positional_encoding(self, timestep):

        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.dim // 2, dtype=torch.float32) / (
                self.dim // 2))
        temp = timestep[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        return time_emb
    
    def forward(self, x_t, timestep, edge_index, y):

        time_emb = self.cosine_positional_encoding(timestep).to(self.device)
        adj = edge_index.clone()
        adj[1] = adj[1] + x_t.shape[0]

        x_and_t = torch.cat((x_t, time_emb), dim=1)
        z = self.encode(x_and_t)
        z_y = self.encode2(y)
        z= torch.cat((z, z_y), dim=0)
        z = self.lay(z, adj)
        pre_x0 = self.decode(z)
        pre_x0 = pre_x0[:x_t.shape[0]]
        return pre_x0

    def lay(self, x, edge_index):
        num_nodes = x.size(0)
        adj = edge_index.clone()
        # 1. 将边索引转换为稀疏张量并添加反向边（无自环）
        adj = torch.cat([adj, adj[[1, 0]]], dim=1)  # 双向边
        row, col = adj

        # 2. 计算对称归一化系数 (D^{-1/2} A D^{-1/2})
        deg = torch.bincount(row, minlength=num_nodes).float()  # 节点度
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0  # 处理孤立节点
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # 归一化系数

        # 3. 构建稀疏邻接矩阵
        SparseTensor_adj = SparseTensor(
            row=row,
            col=col,
            value=norm,
            sparse_sizes=(num_nodes, num_nodes))
        emb_k =x
        embs = [emb_k]

        for i in range(self.k):
            emb_k = matmul(SparseTensor_adj, emb_k)
            embs.append(emb_k)

        emb_k = torch.stack(embs, dim=1).mean(dim=1)
        return emb_k

    def build_symmetric_edge_index(self,edge_index):
        # 翻转方向（反向边）
        reversed_edge = edge_index[[1, 0]]
        # 拼接原始 + 反向边
        symmetric_edge_index = torch.cat([edge_index, reversed_edge], dim=1)

        return symmetric_edge_index