import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
xavier_uniform_ = init.xavier_uniform_
zeros_ = init.zeros_
class GCNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads=8, qkv_bias=False, is_attn=True, attn_drop=0.3, droprate=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.bias = False
        self.is_attn = is_attn
        self.proj = nn.Linear(input_size, hidden_size, bias=self.bias)
        self.another1_gcn_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=self.bias) for i in range(num_layers)])
        self.another2_gcn_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=self.bias) for i in range(num_layers)])
        self.self_loof_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=self.bias) for i in range(num_layers)])
        self.num_heads = num_heads
        if is_attn:
            head_dim = hidden_size // num_heads
            self.scale = head_dim ** -0.5
            self.qkv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
            self.attn_drop = nn.Dropout(p=attn_drop)
        self.dropout = nn.Dropout(p=droprate)
        #self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用xavier_uniform_初始化权重
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # 将偏置项初始化为零
                init.zeros_(m.bias)
    def forward(self, adj, x):
        x = self.proj(x)
        first = (adj == 0.1).float()
        second = (adj == 0.6).float()
        denom = first.sum(-1, keepdim=True) + second.sum(-1, keepdim=True) + 1
        first_neigh = (adj == 0.1).float()
        second_neigh = (adj == 0.2).float()
        third_neigh = (adj == 0.3).float()
        four_neigh = (adj == 0.4).float()
        forth_neigh = (adj == 0.5).float()
        sixth_neigh = (adj == 0.6).float()
        # seven_neigh = (adj == 0.7).float()
        # eight_neigh = (adj == 0.8).float()
        #
        # ninth_neigh = (adj == 0.9).float()
        # tenth_neigh = (adj == 2.0).float()
        # elven_neigh = (adj == 2.1).float()
        # twelve_neigh = (adj == 2.2).float()
        denom = first_neigh.sum(-1, keepdim=True) \
                + second_neigh.sum(-1, keepdim=True) \
                + third_neigh.sum(-1, keepdim=True) \
                + four_neigh.sum(-1, keepdim=True) \
                + forth_neigh.sum(-1, keepdim=True) \
                + sixth_neigh.sum(-1, keepdim=True) + 1
        masks = [first, second]
        if self.is_attn:
            N, C = x.shape
            qkv = self.qkv(x).view(N, 2, self.num_heads, C // self.num_heads).permute(1, 2, 0, 3)
            q, k = qkv.unbind(0)
            attn = (torch.matmul(q, k.permute(0, 2, 1))) * self.scale
            attnclone=attn.clone()
            outputs = []
            for i in range(self.num_heads):
                dense = []
                for mask in masks:
                    mask = -1e9 * (1.0 - mask)
                    attnclone[i] = attnclone[i] + mask  # 避免就地操作
                    attnclone[i] = F.softmax(attnclone[i], dim=-1)
                    attnclone[i] = self.attn_drop(attnclone[i])
                    dense.append(attnclone[i])
                for l in range(self.num_layers):
                    self_node = self.self_loof_layers[l](x)
                    another1_neigh_Ax = self.another1_gcn_layers[l](torch.einsum(
                'kl, lz -> kz',dense[0].clone(), x))
                    another2_neigh_Ax = self.another2_gcn_layers[l](torch.einsum(
                'kl, lz -> kz',dense[1].clone(), x))
                    AxW = (self_node + another1_neigh_Ax + another2_neigh_Ax) / 3
                    gAxWb = F.relu(AxW)
                    x = self.dropout(gAxWb) if 0 < self.num_layers - 1 else gAxWb
                outputs.append(x)
            x = torch.mean(torch.stack(outputs), dim=0)
        else:
            for l in range(self.num_layers):
                self_node = self.self_loof_layers[l](x)
                another1_neigh_Ax = self.another1_gcn_layers[l](torch.matmul(first, x))
                another2_neigh_Ax = self.another2_gcn_layers[l](torch.matmul(second, x))
                if l != self.num_layers - 1:
                    AxW = (self_node + another1_neigh_Ax + another2_neigh_Ax) / denom
                else:
                    AxW = (self_node + another1_neigh_Ax + another2_neigh_Ax)
                gAxWb = F.relu(AxW)
                x = self.dropout(gAxWb) if l < self.num_layers - 1 else gAxWb
        return x
