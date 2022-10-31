import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model%2!=0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.view([-1, self.max_len, self.d_model])[0]
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, d_motion, d_control, num_heads=1, dropout=0.0, device=None):
        super(Encoder, self).__init__()

        self.pos_encoder = PositionalEncoding(d_motion, dropout)

        self.W_Q = nn.Linear(d_motion, d_control * num_heads, bias=False)
        self.W_K = nn.Linear(d_motion, d_control * num_heads, bias=False)
        self.W_V = nn.Linear(d_motion, d_control * num_heads, bias=False)

        self.attention = nn.MultiheadAttention(
            embed_dim = d_control,
            num_heads = num_heads,
            device = device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.attention(self.W_Q(x), self.W_K(x), self.W_V(x))[0]
        out = nn.Softmax(dim=-1)(x)
        return out

class MotionParser(nn.Module):
    def __init__(self, d_motion, d_control, num_heads=1, dropout=0.0, device=None):
        super(MotionParser, self).__init__()
        self.layers = nn.ModuleList([Encoder(d_motion, d_control, num_heads, dropout, device)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parse_data(self, exp_traj):
        self.betas = self.forward(exp_traj)

    def get_beta(self, state, i):
        return self.betas[i] # TODO: Check dimension
