import torch
import torch.nn as nn
import numpy as np
from .SAGCNLayer import GCNLayer
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self, num_tags, vocab_size, gcn_hidden_size, num_layers, num_heads):
        super(GCNModel, self).__init__()
        self.conv1 = GCNLayer(input_size=vocab_size, hidden_size=gcn_hidden_size, num_layers=num_layers, num_heads=num_heads, is_attn=False)
        self.conv2 = GCNLayer(input_size=gcn_hidden_size, hidden_size=num_tags, num_layers=num_layers, num_heads=num_heads, is_attn=False)
        self.fc = nn.Linear(gcn_hidden_size, num_tags)

    def forward(self, inputs, adj=torch.randn(2, 2).to("cuda:0")):
        x = self.conv1(adj, inputs.squeeze())
        x = self.conv2(adj, x)
        #out = F.log_softmax(x, dim=1)
        return x
