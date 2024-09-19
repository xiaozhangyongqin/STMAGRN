import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from memory import Memory
from decoder import AGCRNN_Decoder
from mambaEncoder import mambaEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].unsqueeze(2).repeat(x.size(0), 1, x.size(2), 1)
        return pe.detach()
        # return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class STMAGRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=2, mem_num=170, mem_dim=64, time_dim=24, adp_dim=20,
                 cl_decay_steps=2000, use_curriculum_learning=True):
        super(STMAGRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.minute_size = 1440
        self.weekday_size = 7
        self.embed_dim = time_dim
        self.adaptive_embedding_dim = adp_dim
        self.model_dim = self.embed_dim * 3 + self.adaptive_embedding_dim  # 80
        # self.model_dim = self.embed_dim  # 64
        # position_encoding and time embedding
        self.position_encoding = PositionalEncoding(self.embed_dim)
        self.daytime_embedding = nn.Embedding(self.minute_size, self.embed_dim)
        self.weekday_embedding = nn.Embedding(self.weekday_size, self.embed_dim)
         #  FC
        self.value_liner = nn.Linear(self.input_dim, self.embed_dim)
        
        # 寻找图特征
        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(12, self.num_nodes, self.adaptive_embedding_dim))
        )
        
         # memory augmented
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.m = Memory(T=self.horizon, num_nodes=self.num_nodes, model_dim=self.model_dim, mem_num=self.mem_num, mem_dim=self.mem_dim)
        
        # mambaEncoder
        self.mamba = mambaEncoder(model_dim=self.model_dim, d_state=16, d_conv=4, expand=2, dropout=0.1)
        # # encoder
        # self.time_former = nn.ModuleList(
        #     [SelfAttentionLayer(model_dim=self.model_dim * 1, feed_forward_dim=512, num_heads=8, dropout=0.15) for _ in
        #      range(1)]
        # )
        # self.space_former = nn.ModuleList([
        #     SelfAttentionLayer(model_dim=self.model_dim * 1, feed_forward_dim=512, num_heads=8, dropout=0.15) for _ in
        #     range(1)
        # ])
        
        # decoder
        self.decoder_dim = self.mem_dim
        self.decoder = AGCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim + self.decoder_dim,
                                      self.decoder_dim,
                                      self.cheb_k, self.num_layers)
        self.reg = nn.Conv2d(
            in_channels=12, out_channels=1, kernel_size=(1, 1), bias=True)
        #
        # # # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, x, x_cov, y_cov, labels=None, batches_seen=None):
        # x: B T N D : (64, 12, 170 , 1) PEMS08
        # y_cov: B T N D : (64, 12, 170 , 2) PEMS08
        origin_x = x
        origin_x_cov = x_cov
        node_embeddings1 = torch.matmul(self.m.ma.We1, self.m.ma.M[-1, :].T)
        node_embeddings2 = torch.matmul(self.m.ma.We2, self.m.ma.M[-1, :].T)
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        # X: FC
        x = self.value_liner(origin_x)  # B T N 24
        # X: position encoding
        x = x + self.position_encoding(origin_x)
        # X: time embedding
        x_day = self.daytime_embedding((origin_x_cov[..., 0] * self.minute_size).round().long())  # 24
        # X: week embedding
        x_week = self.weekday_embedding((origin_x_cov[..., 1]).long())  # 24
        # X: adpative node embedding
        x_adpative = self.adaptive_embedding.expand(
            size=(x.shape[0], *self.adaptive_embedding.shape)
        )  # dim = 28
        x = torch.cat([x, x_day, x_week, x_adpative], dim=-1)  # 64

        # mambaEncoder
        x_mb = self.mamba(x.reshape(-1, self.horizon, self.model_dim)).reshape(x.shape)

        x_ma = self.m(x_mb)

        # Decoder
        h_t = self.reg(x_ma)[:, 0, ...]
        ht_list = [(h_t, h_t)] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...], x_ma[:, t, ...]], dim=-1),
                                         ht_list, supports)  # 64  + 64
            go = self.proj(h_de)
            out.append(go)
            # 是否用真实值
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)

        return output