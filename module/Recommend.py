import torch
import torch.nn as nn
from lightGCNlay import lightGCN
from GaussianDiffusion import GaussianDiffusion
from user_Denoise import UDenoise
from item_Denoise import IDenoise

class RcModel(nn.Module):
    def __init__(self, device, batch_size, embedding_dim, num_users, num_items):
        super(RcModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.latent_dim = embedding_dim * 2
        self.num_users = num_users
        self.num_items = num_items

        self.noise_scale = 0.1
        self.noise_min = 0.001
        self.noise_max = 0.02
        self.steps = 10
        self.init_model()


    def init_model(self):
        self.users_emb = nn.Embedding(self.num_users, self.embedding_dim).to(self.device)
        self.items_emb = nn.Embedding(self.num_items, self.embedding_dim).to(self.device)
        nn.init.xavier_normal_(self.users_emb.weight)
        nn.init.xavier_normal_(self.items_emb.weight)

        self.emb_lays = nn.ModuleList()
        self.emb_lays.append(lightGCN(self.embedding_dim))

        self.Udiffusion = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps, beta_fixed=True)
        self.Idiffusion = GaussianDiffusion(self.noise_scale, self.noise_min, self.noise_max, self.steps, beta_fixed=True)

        self.Udenoise = UDenoise(self.embedding_dim, self.latent_dim, self.embedding_dim, self.device)
        self.Idenoise = IDenoise(self.embedding_dim, self.latent_dim, self.embedding_dim, self.device)

    def forward(self):
        
        return
    
    def training(self):

        return