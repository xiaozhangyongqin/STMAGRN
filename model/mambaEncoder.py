# """
#  @Author: zhangyq
#  @FileName: test.py
#  @DateTime: 2024/6/12 14:54
#  @SoftWare: PyCharm
# """
# import torch
# import torch.nn as nn
# from mamba_ssm import Mamba


# # # This module uses roughly 3 * expand * d_model^2 parameters
# #     d_model=dim, # Model dimension d_model
# #     d_state=16,  # SSM state expansion factor
# #     d_conv=4,    # Local convolution width
# #     expand=2,    # Block expansion factor

# class mambaEncoder(nn.Module):
#     def __init__(self, model_dim=64, d_state=16, d_conv=4, expand=2, dropout=0.15):
#         super(mambaEncoder, self).__init__()
#         self.model_dim = model_dim
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand

#         self.mamba = Mamba(d_model=self.model_dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
#         self.mlp = nn.Linear(self.model_dim, self.model_dim)
#         self.ln = nn.LayerNorm(self.model_dim)
#         self.ln1 = nn.LayerNorm(self.model_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         # self.gelu = nn.GELU()

#     def forward(self, x):
#         residual = x
#         x = self.ln(x)
#         out = self.mamba(x)
#         out = self.dropout1(out)
#         z = out + residual
#         out = self.ln1(z)

#         out = self.mlp(out)
#         out = self.dropout2(out) + z
#         return out


# if __name__ == "__main__":

#     batch, length, N, dim = 32, 12, 170, 64
#     x = torch.randn(batch, length, N, dim).to("cuda")
#     x = x.reshape(-1, length, dim)
#     model = mambaEncoder(model_dim=dim, d_state=64, d_conv=4, expand=2, dropout=0.15).to("cuda")
#     y = model(x)
#     assert y.shape == x.shape
#     y = y.reshape(batch, length, N, dim)
#     print(f'y.shape:{y.shape}, x.shape{x.shape}')



"""
 @Author: zhangyq
 @FileName: test.py
 @DateTime: 2024/6/12 14:54
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


# # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor

class mambaEncoder(nn.Module):
    def __init__(self, model_dim=64, d_state=16, d_conv=4, expand=2, dropout=0.15):
        super(mambaEncoder, self).__init__()
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.mamba = Mamba(d_model=self.model_dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        # self.ffn = nn.Sequential(
        #     nn.Linear(self.model_dim, 128),
        #     nn.SiLU(),
        #     nn.Linear(128, self.model_dim)
        # )
        self.mlp = nn.Linear(self.model_dim, self.model_dim)
        self.ln = nn.LayerNorm(self.model_dim)
        self.ln1 = nn.LayerNorm(self.model_dim)
        # self.ln2 = nn.LayerNorm(self.model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.ln(x)
        out = self.mamba(x)
        out = self.dropout1(out)
        z = out + residual
        out = self.ln1(z)

        out = self.mlp(out)
        out = self.dropout2(out) + z
        return out


if __name__ == "__main__":

    batch, length, N, dim = 32, 12, 170, 64
    x = torch.randn(batch, length, N, dim).to("cuda")
    x = x.reshape(-1, length, dim)
    model = mambaEncoder(model_dim=dim, d_state=64, d_conv=4, expand=2, dropout=0.15).to("cuda")
    y = model(x)
    assert y.shape == x.shape
    y = y.reshape(batch, length, N, dim)
    print(f'y.shape:{y.shape}, x.shape{x.shape}')



