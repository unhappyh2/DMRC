import torch.nn as nn
import torch
import random
import time
from tqdm import trange
from collections import defaultdict
from function.loss_func import bpr_loss
from function.evaluate import evaluation
from module.lightGCNlay import lightGCN

class MF(nn.Module):
    def __init__(self, device,batch_size, embedding_dim, num_users, num_items):
        super(MF, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.init_model()

    def init_model(self):
        self.users_emb = nn.Embedding(self.num_users, self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(self.num_items, self.embedding_dim).to(self.device)
        '''
        nn.init.normal_(self.users_emb.weight, std=0.1)  # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.items_emb.weight, std=0.1)
        '''
        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        self.emb_lays = nn.ModuleList()
        self.emb_lays.append(lightGCN(self.embedding_dim))

    def forward(self,batch_users,batch_pos_items,batch_neg_items):
        users_emb_final_batch = self.get_users_emb(batch_users)
        pos_items_emb_final_batch = self.get_items_emb(batch_pos_items)
        neg_items_emb_final_batch = self.get_items_emb(batch_neg_items)
        return users_emb_final_batch, pos_items_emb_final_batch, neg_items_emb_final_batch

    def get_users_emb(self, user_id):
        users_emb = self.users_emb.weight
        return users_emb[user_id]

    def get_items_emb(self, item_id):
        items_emb = self.items_emb.weight
        return items_emb[item_id]

    def Coach(self,epochs,train_edge,val_edge):
        LAMBDA = 1e-6
        LR = 1e-3
        K = 20
        ITERS_PER_EVAL = 20
        ITERS_PER_LR_DECAY = 200
        metrics = {}
        best_recall = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)

        for epoch in trange(epochs):
            self.train()
            user_emb_0, item_emb_0 = self.users_emb.weight, self.items_emb.weight
            emb_edge = torch.cat([train_edge[0], train_edge[1] + self.num_users])

            for emb_layer in self.emb_lays:
                user_emb_final, item_emb_final = emb_layer(user_emb_0, item_emb_0, emb_edge, self.num_users, self.num_items)

            for batch_i, num_batch, batch_users, batch_pos_items, batch_neg_items in self.batch_sample(train_edge):
                optimizer.zero_grad()
                time_start = time.time()
                users_emb_final_batch, pos_items_emb_final_batch, neg_items_emb_final_batch = user_emb_final[batch_users], item_emb_final[batch_pos_items], item_emb_final[batch_neg_items]

                users_emb_0_batch = self.users_emb.weight[batch_users]
                pos_items_emb_0_batch = self.items_emb.weight[batch_pos_items]
                neg_items_emb_0_batch = self.items_emb.weight[batch_neg_items]

                loss = bpr_loss(users_emb_final_batch, users_emb_0_batch, pos_items_emb_final_batch,
                              pos_items_emb_0_batch, neg_items_emb_final_batch, neg_items_emb_0_batch, LAMBDA)
                
                loss.backward()
                optimizer.step()
                time_end = time.time()

                print(f"\r[Iteration {epoch}/{epochs}] batch[{batch_i}\{num_batch}] time_use : {time_end - time_start} loss: {loss.item()}", end=' ', flush=True)
            print("end of epoch batch")
            if epoch % ITERS_PER_EVAL == 0:
                self.eval()
                print("start evaluation")
                recall, precision, ndcg = evaluation(
                    self, val_edge, [train_edge], K)
                metrics['recall@20'] = recall
                metrics['precision@20'] = precision
                metrics['ndcg@20'] = ndcg

                print(f"\n[Iteration {epoch}/{epochs}] \n")
                print(
                    f"loss = {loss.item()}")
                print(
                    f"val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")

                if recall > best_recall:
                    best_recall = recall
                    torch.save(self.state_dict(), f"model_pth/best_model.pth")
                    print(f"===[ best model saved at epoch {epoch} !!! ]===")




    def batch_sample(self,edge_index):
        # 1. 构建 train_user_dict，记录每个用户交互过哪些物品
        train_user_dict = defaultdict(set)
        users = edge_index[0].tolist()
        items = edge_index[1].tolist()
        for u, i in zip(users, items):
            train_user_dict[u].add(i)

        # 2. 打乱所有正样本 (u, i⁺)
        pos_pairs = list(zip(users, items))
        random.shuffle(pos_pairs)

        # 3. 划分 batch
        n_total = len(pos_pairs)
        n_batch = n_total // self.batch_size + 1

        for i in range(n_batch):
            batch_pairs = pos_pairs[i * self.batch_size: (i + 1) * self.batch_size]

            batch_users = []
            batch_pos_items = []
            batch_neg_items = []

            for u, i_pos in batch_pairs:
                # 4. 负采样：从未交互过的物品中随机选一个
                while True:
                    i_neg = random.randint(0, self.num_items - 1)
                    if i_neg not in train_user_dict[u]:
                        break
                batch_users.append(u)
                batch_pos_items.append(i_pos)
                batch_neg_items.append(i_neg)

            # 5. 返回一个 batch
            yield (
                i,
                n_batch,
                torch.LongTensor(batch_users),
                torch.LongTensor(batch_pos_items),
                torch.LongTensor(batch_neg_items)
            )




