import torch
import torch.nn as nn
import torch.nn.init as torch_init
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=2500):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model1, d_model2, dim_k, n_heads=1):
        super(CrossAttention, self).__init__()
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.dim_k =dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model1, dim_k)
        self.k = nn.Linear(d_model2, dim_k)
        self.v = nn.Linear(d_model2, dim_k)

        self.o = nn.Linear(dim_k, d_model2)
        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, y, adj):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)

        att_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        print(f"global {att_map.shape} {att_map}\n\n")        
        print(f"global+adj {att_map + adj}\n\n")
        att_map = self.act(att_map + adj)
        print(f"softmax {att_map.shape} {att_map}")
        temp = torch.matmul(att_map, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(temp).reshape(-1, y.shape[1], y.shape[2])

        return output


class DistanceAdj(nn.Module):
    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.bias = nn.Parameter(torch.FloatTensor(1))

    def forward(self, batch_size, max_seqlen):
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist))#.cuda()
        self.dist = torch.exp(- torch.abs(self.w * (self.dist**2) + self.bias))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1)#.cuda()

        return self.dist

dis_adj = DistanceAdj()
cross_attention = CrossAttention(128, 1024, 128)

x = torch.rand(2,8,1152)    
f_v = x[:, :, :1024]
f_a = x[:, :, 1024:]
adj = dis_adj(f_v.shape[0], f_v.shape[1])
#for i in range(10):
#    adj = dis_adj(f_v.shape[0], f_v.shape[1])
#    print(adj)

new_v = cross_attention(f_a, f_v, adj)



def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class CMA_LA(nn.Module):
    def __init__(self, modal_a, modal_b, hid_dim=128, d_ff=512, dropout_rate=0.1):
        super(CMA_LA, self).__init__()

        self.cross_attention = CrossAttention(modal_b, modal_a, hid_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(d_ff, 128, kernel_size=1),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x, y, adj):
        new_x = x + self.cross_attention(y, x, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)

        return new_x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes
        self.dis_adj = DistanceAdj()

        self.cross_attention = CMA_LA(modal_a=1024, modal_b=128, hid_dim=128, d_ff=512)
        self.classifier = nn.Conv1d(128, 1, 7, padding=0)
        self.apply(weight_init)

    def forward(self, x):
        f_v = x[:, :, :1024]
        f_a = x[:, :, 1024:]
        
        bs, seqlen = f_v.shape[0:2]
        
        adj = self.dis_adj(bs, seqlen)

        new_v = self.cross_attention(f_v, f_a, adj) ## b, f, t
        
        new_v = F.pad(new_v, (6, 0))
        logits = self.classifier(new_v) ## b, t
        logits = logits.squeeze(dim=1)
        logits = torch.sigmoid(logits)

        return logits