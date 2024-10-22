import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.pstfwd.utils import PstFwdUtils
from src.model.net.layers import SConv, VLstm, Temporal

import math#, numpy as np
#from scipy.spatial.distance import pdist, squareform

from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)


########
class SelfAnttention(nn.Module):
    def __init__(self, dfeat, att_dim=128, r=1, do=0.3, cls_dim=64):
        super().__init__()
        self.net_att = nn.Sequential(
            nn.Linear(dfeat, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, r),
            nn.Softmax(dim=1),
            nn.Dropout(do)
            )
        self.net_cls = nn.Sequential(
            nn.Linear(dfeat, cls_dim),
            nn.Linear(cls_dim, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, t, f = x.shape
        old_x = x ## b, t, audf
        att_wght = self.net_att(x) ## (b, t, r)
        mxyz = [ (old_x * att_wght[:,:,0].unsqueeze(2)).sum(dim=1) for i in range(att_wght.shape[2])]
        mcat = torch.cat(mxyz, dim=1)  # (b, r*f)
        x_cls = self.net_cls(mcat).view(b)
        return x_cls
#######


#class DistanceAdj(nn.Module):
#    def __init__(self):
#        super().__init__()
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

class Network(nn.Module):
    def __init__(self, dfeat, _cfg, rgs=None):
        super().__init__()
        self.dfrgb, self.dfaud = dfeat[0], dfeat[1]        
        self.xat_hdim = _cfg.xat_hdim
        #if self.xat_hdim != self.dfaud:
        #    self.xat_hdim = self.dfaud
            
        self.embedding = Temporal(self.dfaud, self.xat_hdim )
        
        self.ffn_hdim = self.dfrgb#//2 #_cfg.ffn_hdim
        self.do = _cfg.do
        
        self.dis_adj = DistanceAdj2(_cfg.sigma, _cfg.bias)
        self.cross_attention = CMA_LA(dfrgb=self.dfrgb, 
                                    dfaud=self.xat_hdim, #self.dfaud , 
                                    xat_hdim=self.xat_hdim, 
                                    ffn_hdim=self.ffn_hdim)
        self.slrgs = rgs
        if self.slrgs is None:
            self.slrgs = SConv(dfeat=self.xat_hdim, 
                                ks=7) #self.dfaud
        self.sig = nn.Sigmoid()
        #self.santt = SelfAnttention(self.dfaud)
        #self.cls = VLstm(self.dfaud, self.dfaud//2)
        
    def forward(self, x):
        frgb = x[:, :, :self.dfrgb]
        faud = x[:, :, self.dfrgb:]
        
        faud = self.embedding(faud)
        
        bs, seqlen = frgb.shape[0:2]
        adj = self.dis_adj(bs, seqlen, x.device) ## b, t, t

        ## rgbf enhanced by audf -> ffn to macth audnf
        new_v = self.cross_attention(frgb, faud, adj) ## b, dfaud, t 
        sls = self.sig( self.slrgs( new_v.permute(0, 2, 1) ) ) ## b, t
        #x_cls = self.santt( new_v.permute(0, 2, 1) )
        
        #vls = self.sig( self.cls(new_v.permute(0, 2, 1)) )
        
        log.debug(f" rgb_aud/ {frgb.shape} {faud.shape}")
        log.debug(f"cross_att/ {new_v.shape}")
        log.debug(f"sls/ {sls.shape}")
        #log.debug(f"x_cls/ {x_cls.shape}")
        #log.debug(f"vls/ {vls.shape}")
        return {
            #'vlscores': vls, ## <- for bce
            'scores': sls, # <- for clas
            #'feats':  new_v.permute(0, 2, 1) 
            #"vls": x_cls, ## <- for salient
        }
        
class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        return scores
    
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