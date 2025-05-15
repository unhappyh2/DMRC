import torch
import torch.nn as nn
from tqdm import trange
from utils.log import log_beginning, logger
from torch_sparse import SparseTensor
from function.loss_func import bpr_loss
from function.evaluate import evaluation
from module.user_Denoise import UDenoise
from module.item_Denoise import IDenoise
from module.GaussianDiffusion import GaussianDiffusion
from module.LightGCN import LightGCN

class mainModel(nn.Module):
    def __init__(self, device, database,timesteps,  batch_size, embedding_dim, num_users, num_items,num_genres=0):
        super(mainModel,self).__init__()
        self.model = "DiffRec "
        self.database = database
        self.k = 0
        self.timesteps = timesteps
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.latent_dim = self.embedding_dim*4
        self.num_users = num_users
        self.num_items = num_items
        self.num_genres = num_genres
        self.EmbedLayer = LightGCN().to(device)
        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0
        
        if self.num_genres != 0:
            self.genres_emb = nn.Embedding(
                num_embeddings=self.num_genres, embedding_dim=self.embedding_dim)
            nn.init.xavier_normal_(self.genres_emb.weight)

        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        '''
        nn.init.normal_(self.users_emb.weight, std=0.1)  # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.items_emb.weight, std=0.1)
        '''
        

        self.diffusionModel = GaussianDiffusion(
            noise_scale =0.1,
            noise_min =0.0001, 
            noise_max =0.02, 
            steps =self.timesteps
        ).to(self.device)

        self.Udenoiser = UDenoise(
            in_channels=self.embedding_dim,
            hidden_channels=self.latent_dim,
            out_channels=self.embedding_dim,
            device=self.device,
            k = 3
        ).to(self.device)

        self.Idenoiser = IDenoise(
            in_channels=self.embedding_dim,
            hidden_channels=self.latent_dim,
            out_channels=self.embedding_dim,
            device=self.device,
            k = 1
        ).to(self.device)

    def Coach(self, num_epochs, train_edge_index, val_edge_index, genres_edge_index=None):
        
        # val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
        #     self.num_users + self.num_items, self.num_users + self.num_items))
        
        LAMBDA = 1e-6
        LR = 1e-3
        K = 20
        ITERS_PER_EVAL = 50
        ITERS_PER_LR_DECAY = 200
        metrics = {}
        best_recall = 0
        log_beginning(self.model, self.database,self.k)
        
        optimizer_embed = torch.optim.Adam(list(self.users_emb.parameters()) + list(self.items_emb.parameters()), lr=LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_embed, gamma=0.95)
        
        optimizer_Udiff = torch.optim.Adam(self.Udenoiser.parameters(), lr=LR)

        loss_Idiff = None
        if self.num_genres != 0:
            optimizer_Idiff = torch.optim.Adam(self.Idenoiser.parameters(), lr=LR)
            self.genres_edge_index = genres_edge_index
            

        self.train()
        print("model Training...")
        for epoch in trange(num_epochs):

            # embedding part
            optimizer_embed.zero_grad()
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.embedding( train_edge_index)
                # mini batching
            user_indices, pos_item_indices, neg_item_indices = self.sample_mini_batch(
                train_edge_index, self.batch_size, self.num_items)
            user_indices, pos_item_indices, neg_item_indices = user_indices.to(
                self.device), pos_item_indices.to(self.device), neg_item_indices.to(self.device)
            users_emb_final_batch, users_emb_0_batch = users_emb_final[user_indices], users_emb_0[user_indices]
            pos_items_emb_final, pos_items_emb_0 = items_emb_final[
                pos_item_indices], items_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_final[
                neg_item_indices], items_emb_0[neg_item_indices]

                # 计算BPR损失
            emb_loss = bpr_loss(users_emb_final_batch, users_emb_0_batch, pos_items_emb_final,
                          pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
            
            emb_loss.backward()
            #print("self.x.grad:",self.x.grad)
            optimizer_embed.step()

            
            # diffusion part
            optimizer_Udiff.zero_grad()
            if self.num_genres != 0:
                optimizer_Idiff.zero_grad()

            users_emb_final = users_emb_final.detach()
            items_emb_final = items_emb_final.detach()

            loss_Udiff = self.diffusionModel.training_losses( self.Udenoiser, users_emb_final,
                                                              train_edge_index, items_emb_final)
            loss_Udiff = loss_Udiff.mean()

            if genres_edge_index is not None:
                loss_Idiff = self.diffusionModel.training_losses( self.Idenoiser, items_emb_final,
                                                                genres_edge_index, self.genres_emb.weight)
                loss_Idiff = loss_Idiff.mean()

            users_emb_denoise = self.diffusionModel.p_sample(
                self.Udenoiser, users_emb_final, self.timesteps, train_edge_index, items_emb_final)
            
            if genres_edge_index is not None:
                items_emb_denoise = self.diffusionModel.p_sample(
                    self.Idenoiser, items_emb_final, self.timesteps, genres_edge_index, self.genres_emb.weight)
            else:
                items_emb_denoise = items_emb_final

            users_emb_final_batch, users_emb_0_batch = users_emb_denoise[user_indices], users_emb_final[user_indices]
            pos_items_emb_final, pos_items_emb_0 = items_emb_denoise[
                pos_item_indices], items_emb_final[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = items_emb_denoise[
                neg_item_indices], items_emb_final[neg_item_indices]

            # 计算BPR损失
            diff_bpr_loss = bpr_loss(users_emb_final_batch, users_emb_0_batch, pos_items_emb_final,
                                pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)
            loss_Udiff = loss_Udiff + diff_bpr_loss

            loss_Udiff.backward()
            optimizer_Udiff.step()

            if genres_edge_index is not None:
                loss_Idiff.backward()
                optimizer_Idiff.step()


            # 验证模型
            if epoch % ITERS_PER_EVAL == 0:
                self.eval()
                recall, precision, ndcg = evaluation(
                    self, val_edge_index, [train_edge_index], K)
                metrics['recall@20']=recall
                metrics['precision@20']=precision
                metrics['ndcg@20']=ndcg

                print(f"\n[Iteration {epoch}/{num_epochs}] \n k = {self.k}")
                print(f"diff_loss[(Udiff,diff_bpr) and (Idiff)]: [({loss_Udiff.item()},{diff_bpr_loss}) and ({loss_Idiff.item()})], \nemb_loss: {round(emb_loss.item(), 5)},")
                print(f"val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")

                if recall > best_recall:
                    best_recall = recall
                    torch.save(self.state_dict(), f"model_pth/best_model.pth")
                    print(f"===[ best model saved at epoch {epoch} !!! ]===")
                    logger(metrics, epoch)

                self.train()
            
            if epoch % ITERS_PER_LR_DECAY == 0 and epoch != 0:
                scheduler.step()
            
        return
    
    def Eval(self, test_edge_index,train_edge_index):
        print("model Evaluating...")
        self.eval()

        with torch.no_grad():
            recall, precision, ndcg = evaluation(
                    self, test_edge_index, [train_edge_index], 20)
            print(f"test_recall@{20}: {round(recall, 5)}, test_precision@{20}: {round(precision, 5)}, test_ndcg@{20}: {round(ndcg, 5)}")
        return
    
    def forward(self, edge_index):

        #embedding
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.embedding(edge_index)

        # diffusion
        users_emb_denoise = self.diffusionModel.p_sample(
            self.Udenoiser, users_emb_final, self.timesteps, edge_index, items_emb_final)
        
        items_emb_denoise = items_emb_final

        # compute final embedding
        # users_emb_final = self.k * users_emb_final + (1 - self.k) * users_emb_denoise
        # items_emb_final = self.k * items_emb_final + (1 - self.k) * items_emb_denoise

        users_emb_final = torch.cat([users_emb_final, users_emb_denoise], dim=1)
        items_emb_final = torch.cat([items_emb_final, items_emb_denoise], dim=1)

        return users_emb_final, items_emb_final



    def embedding(self, edge_index):
        sparse_edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
            self.num_users + self.num_items, self.num_users + self.num_items))
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = self.EmbedLayer(
                                                                self.users_emb.weight, 
                                                                self.items_emb.weight, 
                                                                sparse_edge_index)
            
        return users_emb_final, users_emb_0, items_emb_final, items_emb_0

    def genre_embedding(self, edge_index):
        items_emb_final, items_emb_0, genres_emb_final, genres_emb_0 = self.EmbedLayer(self.items_emb.weight, self.genres_emb.weight, edge_index)
        return items_emb_final, items_emb_0



    def sample_mini_batch(self, edge_index, batch_size, num_items):
        """
        Args:
            edge_index (torch.Tensor): 边列表，形状为 [2, num_edges]
            batch_size (int): 批大小
            num_items (int): 物品数量
        Returns:
            tuple: user indices, positive item indices, negative item indices
        """
        src = edge_index[0]
        dst = edge_index[1]

        num_samples = min(src.size(0), batch_size)
        perm = torch.randperm(src.size(0), device=src.device)[:num_samples]
        src = src[perm]
        pos_dst = dst[perm]
        # 随机从已有的物品编号中采样负样本（编号范围是 0 ~ num_items-1）
        neg_dst = torch.randint(0, num_items, (num_samples,), device=src.device)
        return src, pos_dst, neg_dst
    
    

    
