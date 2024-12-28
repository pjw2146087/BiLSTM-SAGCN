#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from .SAGCNLayer import GCNLayer

# Attention module
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttBiLSTM(nn.Module):
    def __init__(
            self,
            n_classes: int,
            feature_size: int,
            rnn_size: int,
            rnn_layers: int,
            dropout: float,
            num_heads: int,
            qkv_bias=False,
            attn_drop=0.0, 
            drop=0.0
    ):
        super(AttBiLSTM, self).__init__()
        self.rnn_size = rnn_size
        self.time_tri=43
        self.BiLSTM = nn.LSTM(
            feature_size, rnn_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.attention = Attention(rnn_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.tanh = nn.Tanh()
        self.gcn_hidden_size=216
        self.gcn_layers=1
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes)
        )
        self.conv1=GCNLayer(input_size=self.time_tri,hidden_size=self.gcn_hidden_size,num_layers=self.gcn_layers,
num_heads=num_heads,is_attn=True)
        self.conv2=GCNLayer(input_size=self.gcn_hidden_size,hidden_size=n_classes,num_layers=self.gcn_layers,
num_heads=num_heads,is_attn=False)
        self.batchnorm = nn.BatchNorm1d(self.time_tri)
        self.layernorm = nn.LayerNorm(rnn_size)

    def forward(self, x,adj):
        x=x.unsqueeze(2)
        #x=self.batchnorm(x)
        # (batch_size, time,emb_size)
        rnn_out,_ = self.BiLSTM(x)
        H = rnn_out[:, :, : self.rnn_size] + rnn_out[:, :, self.rnn_size:]
        H = self.layernorm(H)
        # attention module
        #r= self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)
        h = self.tanh(H.mean(dim=2))  # (batch_size, rnn_size)
        g=self.conv1(adj,h)
        scores = self.conv2(adj,g)
        return scores