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
