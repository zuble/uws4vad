import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.utils.logger import get_log
log = get_log(__name__)


#############
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hdim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 4, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(2*inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b,n,d=x.size()
        qkvt = self.to_qkv(x).chunk(4, dim = -1)   
        q, k, v, t = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkvt)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots)
        #log.debug(f"\tAtt1 {list(attn1.shape)}")
        
        tmp_ones = torch.ones(n).to(x.device)
        tmp_n = torch.linspace(1, n, n).to(x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b,self.heads, 1, 1)
        #log.debug(f"\tAtt2 {list(attn2.shape)}")
        
        out = torch.cat([torch.matmul(attn1, v),torch.matmul(attn2, t)],dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        #log.debug(f"\our {list(out.shape)}")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_hdim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, ff_hdim, dropout = dropout))
            ]))
    def forward(self, x):
        for depth, (attn, ff) in enumerate(self.layers):
            #log.debug(f"Depth[{depth}] pre-att {list(x.shape)}")
            x = attn(x) + x
            #log.debug(f"Depth[{depth}] pre-ff {list(x.shape)}")
            x = ff(x) + x
            #log.debug(f"Depth[{depth}] out {list(x.shape)}")
        return x
#############  