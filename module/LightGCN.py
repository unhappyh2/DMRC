import torch
from torch import nn, Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class LightGCN(MessagePassing):
    def __init__(self, K=3, add_self_loops=False, **kwargs):
        """
        Args:
            num_users (int): 用户数量
            num_items (int): 电影数量
            embedding_dim (int, optional): 嵌入维度，设置为64，后续可以调整观察效果
            K (int, optional): 传递层数，设置为3，后续可以调整观察效果
            add_self_loops (bool, optional): 传递时加不加自身节点，设置为不加
        """
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.add_self_loops = add_self_loops
 
    def forward(self, users_emb, items_emb, edge_index: SparseTensor):
        """
        Args:
            edge_index (SparseTensor): 邻接矩阵
        Returns:
            tuple (Tensor): e_u%^k, e_u^0, e_i^k, e_i^0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix

        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([ users_emb, items_emb]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # 多尺度扩散
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
 
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K
 
        users_emb_final, items_emb_final = torch.split(
            emb_final, [users_emb.shape[0], items_emb.shape[0]])  # splits into e_u^K and e_i^K
 
        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, users_emb, items_emb_final, items_emb
 
    def message(self, x_j: Tensor) -> Tensor:
        return x_j
 
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)