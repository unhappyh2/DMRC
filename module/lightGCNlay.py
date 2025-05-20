import torch.nn as nn
import torch

class lightGCN(nn.Module):
    def __init__(self, emb_dim , k = 3):
        super(lightGCN, self).__init__()
        self.k = k
        self.emb_dim = emb_dim

    def forward(self,user_emb,item_emb,edge_index, num_users, num_items):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        all_emb_list = [all_emb]
        norm_adj = self.norm_adj(edge_index, num_users + num_items)

        for _ in range(self.k):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            all_emb_list.append(all_emb)

        all_emb = sum(all_emb_list) / (self.k + 1)
        user_emb = all_emb[:num_users]
        item_emb = all_emb[num_users:]
        return user_emb, item_emb
        

    def norm_adj(self,edge_index, num_nodes):
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt[row] * edge_index * deg_inv_sqrt[col]
        return norm_adj
