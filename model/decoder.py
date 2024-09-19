import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv, support_set[-1]

class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 4 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)
        # self.Q = nn.Parameter(torch.FloatTensor(dim_in + dim_out, dim_out))
        # self.M = nn.Parameter(torch.FloatTensor(node_num, dim_out))

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: (h, c) where h and c are both B, num_nodes, hidden_dim
        # print(state.shape)
        h, c = state
        h, c = h.to(x.device), c.to(x.device)
        input_and_state = torch.cat((x, h), dim=-1)
        z_r, support_last = self.gate(input_and_state, supports)
        i, f, o, g = torch.split(z_r, self.hidden_dim, dim=-1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        # q = torch.matmul(input_and_state, self.Q)
        # w = torch.softmax(torch.matmul(q, self.M.T), dim=-1)
        # m = torch.matmul(w, self.M)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return (h_new, c_new), support_last

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.node_num, self.hidden_dim)
        c = torch.zeros(batch_size, self.node_num, self.hidden_dim)
        return (h, c)

class AGCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(AGCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, (B, N, hidden_dim))
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state, _ = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state[0]  # only h is used as the input to the next layer
        return current_inputs, output_hidden
