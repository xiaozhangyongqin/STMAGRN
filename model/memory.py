# import torch
# import torch.nn as nn

# class MemoryAugmented(nn.Module):
#     def __init__(self, num_nodes=170, mem_num=40, mem_dim=64, loop_times=2, fusion_factor=0.7):
#         super(MemoryAugmented, self).__init__()
#         self.mem_num = mem_num
#         self.mem_dim = mem_dim
#         self.num_nodes = num_nodes
#         self.loop_times = loop_times

#         #self.fusion_factor = nn.Parameter(torch.rand(1), requires_grad=True)
#         self.fusion_factor = fusion_factor

#         self.memory = nn.ParameterDict({
#             'Memory': nn.Parameter(torch.randn(mem_num, mem_dim), requires_grad=True),
#             'We1': nn.Parameter(torch.randn(num_nodes, mem_num), requires_grad=True),
#             'We2': nn.Parameter(torch.randn(num_nodes, mem_num), requires_grad=True)
#         })

#         self._init_weights()

#     def _init_weights(self):
#         for param in self.memory.values():
#             nn.init.xavier_normal_(param)

#     def query_memory(self, h_t):
#         query = h_t
#         value_list = [query]
#         att_score_list = []
#         for i in range(self.loop_times):
#             att_score = torch.softmax(torch.matmul(value_list[i], self.memory['Memory'].T), dim=-1)
#             value = torch.matmul(att_score, self.memory['Memory'])
#             value_list.append(value)
#             att_score_list.append(att_score)

#         _, ind = torch.topk(att_score_list[-1], k=2, dim=-1)
#         pos = self.memory['Memory'][ind[..., 0]]
#         neg1 = self.memory['Memory'][ind[..., 1]]
#         neg2 = self.memory['Memory'][ind[..., 1]]

#         x_aug = self.fusion_factor * value_list[1] + (1 - self.fusion_factor) * value_list[2]
#         # x_aug = value_list[1]

#         return x_aug, query, pos, neg1, neg2

#     def forward(self, x):
#         # print(self.fusion_factor)
#         return self.query_memory(x)

"""
 @Author: zhangyq
 @FileName: memory.py
 @DateTime: 2024/6/18 20:49
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn


class MemoryAugmented(nn.Module):
    def __init__(self, T=12, num_nodes=170, mem_num=64, mem_dim=64):
        super(MemoryAugmented, self).__init__()
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.mem_num = mem_num

        self.M = nn.Parameter(torch.FloatTensor(T, self.mem_num, self.mem_dim))
        self.We1 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.mem_dim))
        self.We2 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.mem_dim))
        nn.init.xavier_normal_(self.M)
        nn.init.xavier_normal_(self.We1)
        nn.init.xavier_normal_(self.We2)

    def forward(self, x):
        B, T, N, D = x.shape
        # T:
        ma_x = []
        for t in range(T):
            q = x[:, t, :]
            m = self.M[t, :]
            score = torch.softmax(torch.matmul(q, m.T), dim=-1)
            value = torch.matmul(score, m)
            ma_x.append(value)
        return torch.stack(ma_x, dim=1)

class Memory(nn.Module):
    def __init__(self, T=12, num_nodes=170, model_dim=64, mem_num=64, mem_dim=128):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.T = T
        self.ma = MemoryAugmented(T, num_nodes, mem_num, mem_dim)
        self.l1 = nn.Linear(model_dim, mem_dim)
        self.l2 = nn.Linear(model_dim, mem_dim)
        self.l3 = nn.Linear(mem_dim, mem_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        resudiual = x
        x = self.l1(x)
        resudiual = self.gelu(self.l2(resudiual))
        x = self.ma(x)
        x = x * resudiual
        x = self.l3(x)
        return x


if __name__ == "__main__":
    x = torch.randn(32, 12, 170, 64)
    y = torch.randn(32, 12, 170, 64)
    ma = MemoryAugmented(T=12, num_nodes=170, mem_dim=64)
    m = Memory(T=12, num_nodes=170, model_dim=64, mem_dim=128)
    print(m(x).shape)
    print(ma(x).shape)
    # print(x * y)
# """
#  @Author: zhangyq
#  @FileName: memory.py
#  @DateTime: 2024/6/18 20:49
#  @SoftWare: PyCharm
# """
# import torch
# import torch.nn as nn


# class MemoryAugmented(nn.Module):
#     def __init__(self, T=12, num_nodes=170, mem_dim=64):
#         super(MemoryAugmented, self).__init__()
#         self.mem_dim = mem_dim
#         self.num_nodes = num_nodes

#         self.M = nn.Parameter(torch.FloatTensor(T, 200, self.mem_dim))
#         self.We1 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.mem_dim))
#         self.We2 = nn.Parameter(torch.FloatTensor(self.num_nodes, self.mem_dim))
#         nn.init.xavier_normal_(self.M)
#         nn.init.xavier_normal_(self.We1)
#         nn.init.xavier_normal_(self.We2)

#     def forward(self, x):
#         B, T, N, D = x.shape
#         # T:
#         ma_x = []
#         Q = x
#         p = []
#         n = []
#         for t in range(T):
#             q = x[:, t, :]
#             m = self.M[t, :]
#             score = torch.softmax(torch.matmul(q, m.T), dim=-1)
#             value = torch.matmul(score, m)
#             ma_x.append(value)
#             _, ind = torch.topk(score, k=2, dim=-1)
#             pos = m[ind[..., 0]]
#             neg = m[ind[..., 1]]
#             p.append(pos)
#             n.append(neg)
#         pos = torch.stack(p, dim=1)
#         neg = torch.stack(n, dim=1)
#         return torch.stack(ma_x, dim=1), {'q': Q, 'pos': pos, 'neg': neg}


# class Memory(nn.Module):
#     def __init__(self, T=12, num_nodes=170, model_dim=64, mem_dim=128):
#         super(Memory, self).__init__()
#         self.mem_dim = mem_dim
#         self.num_nodes = num_nodes
#         self.T = T
#         self.ma = MemoryAugmented(T, num_nodes, mem_dim)
#         self.l1 = nn.Linear(model_dim, mem_dim)
#         self.l2 = nn.Linear(model_dim, mem_dim)
#         self.l3 = nn.Linear(mem_dim, model_dim)
#         self.gelu = nn.GELU()

#     def forward(self, x):
#         resudiual = x
#         x = self.l1(x)
#         resudiual = self.gelu(self.l2(resudiual))
#         x, dic = self.ma(x)
#         x = x * resudiual
#         x = self.l3(x)
#         return x, dic


# if __name__ == "__main__":
#     x = torch.randn(32, 12, 170, 64)
#     y = torch.randn(32, 12, 170, 64)
#     ma = MemoryAugmented(T=12, num_nodes=170, mem_dim=64)
#     m = Memory(T=12, num_nodes=170, model_dim=64, mem_dim=128)
#     out, dic = m(x)
#     print(out.shape, dic['q'].shape, dic['pos'].shape, dic['neg'].shape)
#     # print(x * y)