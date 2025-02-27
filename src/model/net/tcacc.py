import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.pstfwd.utils import PstFwdUtils
from src.model.net.layers import SConv, VLstm, Temporal

import math

from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)


## from pel4vad
class DistanceAdj2(nn.Module):
    def __init__(self, sigma, bias):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)    
        self.b.data.fill_(bias)
    def forward(self, batch_size, seq_len, dvc):
        arith = torch.arange(seq_len).reshape(-1, 1).float()
        dist = torch.cdist(arith, arith, p=1).float().to(dvc)
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = dist.unsqueeze(0).repeat(batch_size, 1, 1)
        return dist
'''    
class CrossAttention(nn.Module):
    def __init__(self, dfaud, dfrgb, xat_hdim, n_heads=1):
        super().__init__()
        self.dfaud = dfaud
        self.dfrgb = dfrgb
        self.xat_hdim = xat_hdim
        self.n_heads = n_heads
        self.q = nn.Linear(dfaud, xat_hdim)
        self.k = nn.Linear(dfrgb, xat_hdim)
        self.v = nn.Linear(dfrgb, xat_hdim)
        self.o = nn.Linear(xat_hdim, dfrgb)
        self.norm_fact = 1 / math.sqrt(xat_hdim)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, y, adj):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.xat_hdim // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.xat_hdim // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.xat_hdim // self.n_heads)
        #print(adj)
        att_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        #print(f"global {att_map.shape} {att_map}\n\n")        
        #print(f"global+adj {att_map + adj}\n\n")
        att_map = self.act(att_map + adj)
        #print(f"softmax {att_map.shape} {att_map}")
        temp = torch.matmul(att_map, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(temp).reshape(-1, y.shape[1], y.shape[2])

        return output ## B T FRGB

class CMA_LA(nn.Module):
    def __init__(self, dfrgb, dfaud, xat_hdim=128, ffn_hdim=512, do=0.1):
        super().__init__()
        self.cross_attention = CrossAttention(dfaud, dfrgb, xat_hdim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dfrgb, ffn_hdim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(do),
            nn.Conv1d(ffn_hdim, dfaud, kernel_size=1),
            nn.Dropout(do),
        )
        self.norm = nn.LayerNorm(dfrgb)
    def forward(self, x_rgb, x_aud, adj):
        new_x = x_rgb + self.cross_attention(x_aud, x_rgb, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x) 
        return new_x ## B FAUD T 
'''
class TCA(nn.Module):
    def __init__(self, 
            d_model, 
            dim_k, 
            dim_v, 
            n_heads, 
            norm=None
        ):
        super(TCA, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k  # same as dim_q
        self.n_heads = n_heads
        self.norm = norm
        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(dim_k)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, mask, adj=None):
        Q = self.q(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).view(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(x).view(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        g_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact ## eq 1
        if adj is not None: 
            ## Dynamic Position Encoding (DPE)
            ## is embedded into the similarity matrix M as a location prior, 
            ## i.e., M ← M + G, thus avoiding affecting the original feature distribution.
            g_map = g_map + adj
        
        l_map = g_map.clone()  ## ++
        ## Fills elements of l_map with -1e9 where mask is True.
        l_map = l_map.masked_fill_(mask.data.eq(0), -1e9)  ## ++

        g_map = self.act(g_map) ## eq 2
        l_map = self.act(l_map) ## eq 5
        x_g = torch.matmul(g_map, V).view(x.shape[0], x.shape[1], -1) ## eq 3
        x_l = torch.matmul(l_map, V).view(x.shape[0], x.shape[1], -1) ## eq 6

        alpha = torch.sigmoid(self.alpha) 
        x_o = alpha * x_g + (1 - alpha) * x_l ## eq 7
        if self.norm: ## Norm(Xo))
            ## However, we observe that either power normalization or L2 normalization 
            ## leads to a significant drop in performance for XD-Violence. 
            ## We argue that normalization may jeopardize the diversity distribution of this dataset 
            ## since it contains videos from different sources and types.
            ## While normalization is necessary for surveillance videos with fixed lenses
            x_o = torch.sqrt(F.relu(x_o)) - torch.sqrt(F.relu(-x_o))  # power norm
            x_o = F.normalize(x_o)  # l2 norm
        
        x_o = self.o(x_o).view(-1, x.shape[1], x.shape[2]) ## fh(Norm(Xo))
        return x_o
    
    
class XEncoder(nn.Module):
    def __init__(self, 
            d_model, 
            hid_dim, 
            out_dim, 
            n_heads, 
            win_size, 
            dropout, 
            gamma, 
            bias, 
            norm=None
        ):
        super(XEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = TCA(d_model, hid_dim, hid_dim, n_heads, norm)
        
        self.mlp1 = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(d_model // 2, out_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj2(gamma, bias)
    
    def get_mask(self, window_size, temporal_scale, seq_len):
        ## eq 4
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.
        m = m.repeat(self.n_heads, len(seq_len), 1, 1)
        return m
    
    def forward(self, x, seq_len):
        b, t, f = x.shape
        adj = self.loc_adj(b, t, x.device) ## b,t,t
        
        mask = self.get_mask(self.win_size, t, seq_len).to(x.device)
        x = x + self.self_attn(x, mask, adj)
        x_c = self.norm(x).permute(0, 2, 1) ## eq 8
        
        x_e = self.mlp1(x_c) ## eq 10
        x_s = self.mlp2(x_e) ## eq 10
        
        return x_s, x_e

    
class Network(nn.Module):
    def __init__(self, 
            dfeat, 
            _cfg, 
            rgs=None
        ):
        super().__init__()
        self.dfrgb, self.dfaud = dfeat[0], dfeat[1]
        
        self.self_attention = XEncoder(
            d_model=self.dfrgb, #1024
            hid_dim=_cfg.hid_dim, #128
            out_dim=_cfg.out_dim, #300
            n_heads=_cfg.n_heads, #1
            win_size=_cfg.win_size, #9
            dropout=_cfg.dropout, #0.1
            gamma=_cfg.gamma, # 0.6 / 0.06
            bias=_cfg.bias, # 0.2 / 0.02
            norm=_cfg.norm, # True / False
        )
        self.t = _cfg.t_step # 9 / 3
        
        self.slrgs = SConv(dfeat=_cfg.out_dim, ks=self.t)
        #self.slrgs = nn.Conv1d(_cfg.out_dim, 1, self.t, padding=0)
        #self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(1 / _cfg.temp))

    def forward(self, x):
        seq_len = torch.sum(torch.max(torch.abs(x), dim=2)[0] > 0, 1)
        log.debug(seq_len)
        x_s, x_e = self.self_attention(x, seq_len)
        logits = self.slrgs(x_s.permute(0,2,1))
        #sls = torch.sigmoid(logits)
        return {
            'feats': x_e.permute(0,2,1),
            'scores': logits, 
        }


class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        self.sig = nn.Sigmoid()
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        scores = self.sig(scores)
        return scores
    

## TODO: 
def fixed_smooth(logits, t_size):
    ins_preds = torch.zeros(0).cuda()
    assert t_size > 1
    if len(logits) % t_size != 0:
        delta = t_size - len(logits) % t_size
        logits = F.pad(logits, (0,  delta), 'constant', 0)

    seq_len = len(logits) // t_size
    for i in range(seq_len):
        seq = logits[i * t_size: (i + 1) * t_size]
        avg = torch.mean(seq, dim=0)
        avg = avg.repeat(t_size)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds

def slide_smooth(logits, t_size, mode='zero'):
    assert t_size > 1
    ins_preds = torch.zeros(0).cuda()
    padding = t_size - 1
    if mode == 'zero':
        logits = F.pad(logits, (0, padding), 'constant', 0)
    elif mode == 'constant':
        logits = F.pad(logits, (0, padding), 'constant', logits[-1])

    seq_len = int(len(logits) - t_size) + 1
    for i in range(seq_len):
        seq = logits[i: i + t_size]
        avg = torch.mean(seq, dim=0).unsqueeze(dim=0)
        ins_preds = torch.cat((ins_preds, avg))

    return ins_preds