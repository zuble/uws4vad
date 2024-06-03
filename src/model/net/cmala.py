import torch
import torch.nn as nn
import torch.nn.init as torch_init
from src.model.net.layers import BasePstFwd, SConv

import math#, numpy as np
#from scipy.spatial.distance import pdist, squareform

from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)

#class DistanceAdj(nn.Module):
#    def __init__(self):
#        super(DistanceAdj, self).__init__()
#        self.w = nn.Parameter(torch.FloatTensor(1))
#        self.bias = nn.Parameter(torch.FloatTensor(1))
#
#    def forward(self, batch_size, max_seqlen):
#        self.arith = np.arange(max_seqlen).reshape(-1, 1)
#        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
#        self.dist = torch.from_numpy(squareform(dist))#.cuda()
#        self.dist = torch.exp(- torch.abs(self.w * (self.dist**2) + self.bias))
#        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1)#.cuda()
#
#        return self.dist

## from pel4vad
class DistanceAdj2(nn.Module):
    def __init__(self, sigma, bias):
        super(DistanceAdj2, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)    
        self.b.data.fill_(bias)

    def forward(self, batch_size, seq_len):
        arith = torch.arange(seq_len).reshape(-1, 1).float()
        dist = torch.cdist(arith, arith, p=1).float()#.cuda()
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = dist.unsqueeze(0).repeat(batch_size, 1, 1)

        return dist
    
class CrossAttention(nn.Module):
    def __init__(self, dfaud, dfrgb, dim_k, n_heads=1):
        super(CrossAttention, self).__init__()
        self.dfaud = dfaud
        self.dfrgb = dfrgb
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(dfaud, dim_k)
        self.k = nn.Linear(dfrgb, dim_k)
        self.v = nn.Linear(dfrgb, dim_k)

        self.o = nn.Linear(dim_k, dfrgb)
        self.norm_fact = 1 / math.sqrt(dim_k)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, y, adj):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, y.shape[0], y.shape[1], self.dim_k // self.n_heads)

        #print(adj)
        att_map = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.norm_fact
        #print(f"global {att_map.shape} {att_map}\n\n")        
        #print(f"global+adj {att_map + adj}\n\n")
        att_map = self.act(att_map + adj)
        #print(f"softmax {att_map.shape} {att_map}")
        temp = torch.matmul(att_map, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(temp).reshape(-1, y.shape[1], y.shape[2])

        return output

class CMA_LA(nn.Module):
    def __init__(self, dfrgb, dfaud, hid_dim=128, d_ff=512, do=0.1):
        super(CMA_LA, self).__init__()

        self.cross_attention = CrossAttention(dfaud, dfrgb, hid_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dfrgb, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(do),
            nn.Conv1d(d_ff, dfaud, kernel_size=1),
            nn.Dropout(do),
        )
        self.norm = nn.LayerNorm(dfrgb)

    def forward(self, x_rgb, x_aud, adj):
        new_x = x_rgb + self.cross_attention(x_aud, x_rgb, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)

        return new_x


class Network(nn.Module):
    def __init__(self, dfeat, _cfg, _cls = None):
        super(Network, self).__init__()
        self.dfrgb, self.dfaud = dfeat[0], dfeat[1]
        self.hid_dim = _cfg.hid_dim
        self.d_ff = _cfg.d_ff
        self.do = _cfg.do
        
        self.dis_adj = DistanceAdj2(_cfg.sigma, _cfg.bias)
        self.cross_attention = CMA_LA(dfrgb=self.dfrgb, 
                                    dfaud=self.dfaud , 
                                    hid_dim=self.hid_dim, 
                                    d_ff=self.d_ff)
        self.sig = nn.Sigmoid()
        
        if _cls is not None:
            self.slcls = instantiate(_cls, dfeat=self.dfaud)
        else: self.slcls = SConv(dfeat=dfaud, ks=7)
        
    def forward(self, x):
        frgb = x[:, :, :self.dfrgb]
        faud = x[:, :, self.dfrgb:]
        log.debug(f" rgb_aud/ {frgb.shape} {faud.shape}")
        
        bs, seqlen = frgb.shape[0:2]
        adj = self.dis_adj(bs, seqlen)

        ## rgbf enhanced by audf -> ffn to macth audnf
        new_v = self.cross_attention(frgb, faud, adj) ## b, f, t
        log.debug(f"cross__att/ {new_v.shape}")
        
        sls = self.sig( self.slcls( new_v.permute(0, 2, 1) ) )
        log.debug(f"sls/ {sls.shape}")
        
        return {
            'sls': sls
        }

class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)
        
    def train(self, ndata, ldata, lossfx):
        super().rshp_out(ndata, 'sls', 'mean') ## crop0
        #log.info(f" pos_rshp: {ndata['sls'].shape}")
        
        L0 = lossfx['clas'](ndata['sls'], ldata)
        
        return L0
        
    def infer(self, ndata):
        ## output is excepted to be segment level 
        
        #log.debug(f"")
        #log.debug(f"sls: {ndata['sls']=}")
        
        return ndata['sls']    
    
    
    
    
## adj in cross attention
#tensor([[[0.9802, 0.9608, 0.8025, 0.5945],
#         [0.9608, 0.9802, 0.9608, 0.8025],
#         [0.8025, 0.9608, 0.9802, 0.9608],
#         [0.5945, 0.8025, 0.9608, 0.9802]],
#
#        [[0.9802, 0.9608, 0.8025, 0.5945],
#         [0.9608, 0.9802, 0.9608, 0.8025],
#         [0.8025, 0.9608, 0.9802, 0.9608],
#         [0.5945, 0.8025, 0.9608, 0.9802]]],
#global torch.Size([1, 2, 4, 4]) 
#tensor([[[  [ 0.0233, -0.0930,  0.0473,  0.3909],
#            [ 0.1152,  0.5969,  0.1038,  0.3224],
#            [ 0.3205,  0.1843, -0.0496, -0.0371],
#            [ 0.6672, -0.3107, -0.5163,  0.2209]],
#
#            [[ 0.2474,  0.1430, -0.0047, -0.4056],
#            [-0.7909, -0.1887,  0.1281, -0.2227],
#            [ 0.0912, -0.1879, -0.0470, -0.0511],
#            [-0.4467,  0.3301,  0.2303,  0.0705]]]], grad_fn=<MulBackward0>)
#global+adj 
#tensor([[[  [1.0035, 0.8678, 0.8498, 0.9854],
#            [1.0760, 1.5771, 1.0646, 1.1249],
#            [1.1230, 1.1450, 0.9306, 0.9237],
#            [1.2617, 0.4918, 0.4445, 1.2011]],
#
#            [[1.2276, 1.1037, 0.7978, 0.1889],
#            [0.1698, 0.7915, 1.0889, 0.5799],
#            [0.8937, 0.7728, 0.9332, 0.9097],
#            [0.1478, 1.1326, 1.1911, 1.0507]]]], grad_fn=<AddBackward0>)
#softmax torch.Size([1, 2, 4, 4]) 
#tensor([[[  [0.2693, 0.2352, 0.2310, 0.2645],
#            [0.2133, 0.3520, 0.2108, 0.2239],
#            [0.2727, 0.2788, 0.2250, 0.2235],
#            [0.3514, 0.1627, 0.1552, 0.3307]],
#
#            [[0.3463, 0.3059, 0.2253, 0.1225],
#            [0.1454, 0.2708, 0.3646, 0.2192],
#            [0.2536, 0.2248, 0.2639, 0.2577],
#            [0.1113, 0.2981, 0.3160, 0.2746]]]], grad_fn=<SoftmaxBackward>)