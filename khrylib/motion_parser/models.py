import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import math
from turtle import forward
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
        # print("positional encoding", x.size(), pe.size())
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, len_seq=10, keep_percent=1/2, device=None):
        super(SelfAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.len_seq = len_seq
        self.keep_percent = keep_percent
        self.device = device

        self.Query = nn.Linear(d_in, d_out)
        self.Key = nn.Linear(d_in, d_out)
        self.Value = nn.Linear(d_in, d_out)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.Query(x)
        key = self.Key(x)
        value = self.Value(x)
        score = torch.bmm(query.view(1, query.size(0), query.size(1)), key.view(1, key.size(1), key.size(0))) / np.sqrt(self.d_out) # tensor. length = input sequence length
        score = score.view(batch_size, batch_size).to(self.device)

        # hard attention
        keep_num = int(batch_size * self.keep_percent)
        filter_threshhold = torch.Tensor([torch.topk(s, keep_num).values.min() for s in score]).to(self.device)
        score = torch.where(score > filter_threshhold, score, score-score)
        
        score = score.view(1, batch_size, batch_size)
        attn = F.softmax(score, -1)
        # print("value device", value.device)
        context = torch.bmm(attn, value.view(1, query.size(0), query.size(1))).view(batch_size, -1)
        return context
        
class Encoder(nn.Module):
    def __init__(self, d_model, num_layers=3, nhead=4, device=None):
        super(Encoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, embedding_size=128, hidden_size=128, num_layers=3, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_size = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(d_model, embedding_size)
        self.activaton = nn.ReLU()
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)
        self.predictor = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        embedded = self.activaton(embedded)
        output, hidden = self.gru(embedded, hidden)
        pred = self.predictor(output[0])
        return pred, hidden

class MotionParser(nn.Module):
    def __init__(self, d_motion, d_model, d_control, num_heads=1, sa_keep_percent=1/5, dropout=0.0, device=None):
        super(MotionParser, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_motion, dropout)
        self.self_attention = SelfAttention(d_motion, d_model, keep_percent=sa_keep_percent)
        self.encoder = Encoder(d_model, num_layers=3, nhead=num_heads)
        self.decoder = Decoder(d_model, num_layers=1, hidden_size=d_model)
        self.predictor = nn.Linear(d_model, d_control)
        self.device = device
        self.hidden = None

    def parse_data(self, x):
        x = self.positional_encoding(x)
        x = self.self_attention(x)
        self.hidden = x

    def forward(self, state, hidden):
        # print("hidden size", self.hidden.size())
        x = hidden
        x = self.encoder(x)
        if state==None:
            state = torch.zeros(1, self.d_model).to(self.device)
        x1 = torch.zeros(x.size(0), x.size(1)).to(self.device)
        for i in range(x.size(0)):
            output, hidden = self.decoder(state, x[i].view(1, -1))
            state = hidden
            x1[i] += output.view(-1)
        x1 = self.predictor(x1)
        return x1

    def get_beta(self, x, i):
        return self.forward(x.view(1, -1), self.hidden[i].view(1, -1)).view(-1)

class DummyMotionParser(nn.Module):
    def __init__(self, d_motion, d_control, traj_length=10):
        super(DummyMotionParser, self).__init__()
        self.embedding = nn.Embedding(traj_length,d_control)

    def forward(self, x):
        pass

    def parse_data(self, exp_traj):
        pass

    def get_beta(self, state, i):
        return self.embedding(torch.as_tensor(i)).softmax(dim=0)

    def load_betas(self, betas):
        self.embedding.weight = torch.nn.Parameter(betas)
