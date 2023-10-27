import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from Config import Config

configs = Config()


def get_attn_pad_mask(masked_data):
    bsz, measures, ts = masked_data.size()
    pad_attn_mask = masked_data[:, :, 0].data.eq(0).unsqueeze(1)  # [bzs, 1, measures]
    return pad_attn_mask.expand(bsz, measures, measures)  # [bzs, measures, measures]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MeasureEmbedding1(nn.Module):
    # Simple embedding, only matmul
    # input of size (batch, n_meas * n_nodes, ts)
    # output of size (batch, n_meas * n_nodes, n_embed)

    def __init__(self, n_embed, n_meas, n_nodes, ts):
        super().__init__()
        self.n_meas = n_meas
        self.n_nodes = n_nodes
        self.n_embed = n_embed
        self.ts = ts
        self.time_embed = nn.ModuleList(nn.Linear(ts, n_embed) for _ in range(n_meas))

    def forward(self, x, Z):
        # time embedding
        bsz = x.shape[0]
        x = x.view(bsz, self.n_meas, self.n_nodes, self.ts)
        x = [x[:, i, :, :] for i in range(self.n_meas)]
        x = torch.cat([layer(xp) for layer, xp in zip(self.time_embed, x)], dim=1)
        x = x.view(self.bsz, self.n_meas * self.n_nodes, self.n_embed)

        # spatial embedding
        x = x.view(bsz, self.n_meas, self.n_nodes, self.n_embed)
        x = torch.matmul(Z, x).view(self.bzs, self.n_meas * self.n_nodes, self.n_embed)
        return x


class MeasureEmbedding2(nn.Module):
    # complex embedding, conv1d for time embed and GCN for spatial embed
    # input of size (batch, n_meas * n_nodes, ts)
    # output of size (batch, n_meas * n_nodes, n_embed)

    def __init__(self, n_embed, n_meas, n_nodes, ts):
        super().__init__()
        self.n_meas = n_meas
        self.n_nodes = n_nodes
        self.n_embed = n_embed
        self.ts = ts

        self.time_embed = nn.ModuleList(nn.Sequential(
            nn.Conv1d(in_channels=i, out_channels=n_nodes, kernel_size=11),  # ts -10
            nn.ReLU(),
            nn.MaxPool1d(2),  # (ts - 10) / 2
            nn.Conv1d(in_channels=i, out_channels=n_nodes, kernel_size=int((ts - 10) / 2 - n_embed + 1)),
        ) for i in n_meas)
        self.spatial_embed = nn.Parameter(torch.FloatTensor(n_embed, n_embed), requires_grad=True)

        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.spatial_embed.size(1))
        self.spatial_embed.data.uniform_(-stdv, stdv)

    def forward(self, x, Z):
        # time embedding
        bsz = x.shape[0]
        meas = [0, self.n_meas]
        x = [x[:, meas[i]:meas[i+1], :] for i in range(len(self.n_meas) - 1)]
        x = [layer(xp) for layer, xp in zip(self.time_embed, x)]

        # spatial embedding
        adj = torch.diag(torch.pow(Z.sum(1), -0.5))
        adj = torch.mm(torch.mm(adj, Z), adj)
        x = [F.relu(torch.matmul(xp, self.spatial_embed)) for xp in x]
        x = torch.cat([torch.matmul(adj, xp) for xp in x], dim=1)
        x = x.view(bsz, -1, self.n_embed)

        return x


class MultiHeadAttention(nn.Module):
    # input of size (batch, n_meas * n_nodes, n_embed)
    # hid_dim：每个词输出的向量维度
    def __init__(self, n_embed, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_embed = n_embed
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert n_embed % n_heads == 0
        self.w_q = nn.Linear(n_embed, n_embed)
        self.w_k = nn.Linear(n_embed, n_embed)
        self.w_v = nn.Linear(n_embed, n_embed)
        self.fc = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([n_embed // n_heads]))

    def forward(self, query, key, value, attn_mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.n_embed // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.n_embed // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.n_embed // self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            attention = attention.masked_fill(attn_mask, -1e10)

        attention = self.dropout(torch.softmax(attention, dim=-1))
        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.n_embed // self.n_heads))
        out = self.dropout(self.fc(x))
        return x


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed, ffn_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, ffn_dim, dropout):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffwd = FeedForward(n_embed, ffn_dim, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, mask):
        x1 = self.ln1(x)
        x = x + self.sa(x1, x1, x1, mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class BertNet(nn.Module):
    def __init__(self, configs):
        super(BertNet, self).__init__()
        self.configs = configs
        self.meas_embed = MeasureEmbedding2(configs.n_embed, configs.n_meas, configs.n_nodes, configs.ts)
        self.Blocks = nn.Sequential(*[Block(configs.n_embed, configs.n_head, configs.ffn_dim, configs.dropout)
                                      for _ in range(configs.n_layers)])
        self.ln = nn.LayerNorm(configs.n_embed)
        # self.proj = nn.ModuleList(nn.Linear(configs.n_embed, 1) for _ in range(configs.n_meas))
        self.proj = nn.ModuleList(nn.Sequential(
            nn.Conv1d(in_channels=configs.n_nodes, out_channels=i, kernel_size=11),  # n_embed -10
            nn.ReLU(),
            nn.MaxPool1d(2),  # (ts - 10) / 2
            nn.Conv1d(in_channels=configs.n_nodes, out_channels=i, kernel_size=int((configs.n_embed - 10) / 2)),
        ) for i in configs.n_meas)

    def forward(self, x, mask_pos, Z):
        x = self.meas_embed(x, Z)
        attn_mask = get_attn_pad_mask(x)
        x = self.Blocks(x, attn_mask)

        # projection
        bsz = x.shape[0]
        meas = [0, self.n_meas]
        x = [x[:, meas[i]:meas[i + 1], :] for i in range(len(self.n_meas) - 1)]
        x = torch.cat([layer(xp) for layer, xp in zip(self.proj, x)], dim=1)
        x = x.view(bsz, self.configs.n_meas * self.configs.n_nodes, 1)

        mask_pos = mask_pos[:, :, None]
        mask_values_pred = torch.gather(x, 1, mask_pos)  # [bzs, max_len, 1]
        return x, mask_values_pred
