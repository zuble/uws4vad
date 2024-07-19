import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.pstfwd.utils import PstFwdUtils

from omegaconf.dictconfig import DictConfig
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
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
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

        tmp_ones = torch.ones(n).to(x.device)
        tmp_n = torch.linspace(1, n, n).to(x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1))
        attn2 = torch.exp(-tg_tmp / torch.exp(torch.tensor(1.)))
        attn2 = (attn2 / attn2.sum(-1)).unsqueeze(0).unsqueeze(1).repeat(b,self.heads, 1, 1)

        out = torch.cat([torch.matmul(attn1, v),torch.matmul(attn2, t)],dim=-1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
#############    


class NormalHead(nn.Module):
    def __init__(self, in_feats=512, ratios=[16, 32], ks=[1, 1, 1]):
        super(NormalHead, self).__init__()

        reduction1 = in_feats // ratios[0] ## = b
        reduction2 = in_feats // ratios[1] ## = 32

        ## torch.nn.Conv1d( in_channels, out_channels, 
        #                   kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv1d(in_feats, reduction1, 
                               ks[0], padding=ks[0] // 2)
        
        self.bn1 = nn.BatchNorm1d(reduction1)
        self.conv2 = nn.Conv1d(reduction1, reduction2, 
                               ks[1], padding=ks[1] // 2)
        
        self.bn2 = nn.BatchNorm1d(reduction2)
        ## regressor layer 
        self.conv3 = nn.Conv1d(reduction2, 1, 
                                ks[2], padding=ks[2] // 2)
        self.sigmoid = nn.Sigmoid()
        
        self.act = nn.ReLU()
        self.bns = [self.bn1, self.bn2]

    def forward(self, x):
        '''
        x: BN * C * T
        return BN * C // 64 * T and BN * 1 * T
        '''
        ## b, 512, t
        outputs = []
        x = self.conv1(x)  ## b, reduction1, t
        outputs.append(x)
        x = self.conv2(self.act(self.bn1(x))) ## b, reduction2 , t
        outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x)))) ## b, 1, t
        outputs.append( x.squeeze(1) )
        return outputs


class Temporal(nn.Module):
    def __init__(self, dfeat, dout):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=dfeat, out_channels=dout, 
                    kernel_size=3,
                    stride=1, 
                    padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        return self.conv_1( x.permute(0, 2, 1) ).permute(0, 2, 1)


class Network(nn.Module):
    def __init__(self, dfeat, _cfg: DictConfig, rgs=None):
        super().__init__()
        
        self.dfeat = sum(dfeat)
        #self.ncrops = _cfg.ncrops
        
        self.embedding = Temporal(self.dfeat, _cfg.emb_dim)
        self.selfatt = Transformer(_cfg.emb_dim, _cfg.depth, _cfg.heads, _cfg.head_dim, _cfg.mlp_dim, _cfg.do)
        self.normal_head = NormalHead(in_feats=_cfg.mlp_dim, ratios=_cfg.nh_dimratios, ks=_cfg.nh_ks)

    def forward(self, x):
        log.debug(f"{x.shape=}")
            
        x_embd = self.embedding(x) ## b, t, 512
        x = self.selfatt(x_embd) ## b, t, 512
        
        nh_res = self.normal_head( x.permute(0, 2, 1) )
        anchors = [bn.running_mean for bn in self.normal_head.bns] ## [red1 ;; red2]
        variances = [bn.running_var for bn in self.normal_head.bns] ## [red1 ;; red2]

        return {
            'anchors': anchors, ## (red1, red2)
            'variances': variances, ## (red1, red2)
            'norm_scors': nh_res[-1], ## b,t 
            'norm_feats': nh_res[:-1] ## [(b, red1, t) ;; (b, red2, t)]          
        }


class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata):
        
        ## t    
        dists =self.pfu._get_mtrcs_dfm(ndata["norm_feats"], ndata["anchors"], ndata["variances"], infer=True)
        log.debug(f"{dists[0].shape=}, {dists[1].shape=}")
        ## t  
        dists_sum = sum(dists).squeeze(0) 
        log.debug(f"{dists_sum.shape=}")
        #log.error(f"{dists[0].max()=} {dists[1].max()=}")

        scores = self.pfu.uncrop(ndata['norm_scors'], 'mean')
        log.error(f"scores {scores.shape} ") #{scores.max()}
        
        out = scores * dists_sum
        log.error(f"scores {out.shape} ") #{scores.max()}
        
        return out

    
    
'''
class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)
    
    def train(self, ndata, ldata, lossfx):
        #super().logdat([ndata])
    
        ## FEAT
        abn_dists, nor_dists = super()._get_mtrcs_dfm(ndata["norm_feats"], ndata["anchors"], ndata["variances"])
        
        #sel_abn_feats = torch.zeros(0, device=self.dvc)
        #sel_nor_feats = torch.zeros(0, device=self.dvc)
        sel_abn_feats, sel_nor_feats = [], []
        for feats, abn_dist, nor_dist in zip(ndata["norm_feats"], abn_dists, nor_dists):
            
            feats = feats.permute(0,2,1)
            feats = super().uncrop(feats, 'mean') ## bs, t, f
            abn_feats, nor_feats = super().unbag(feats) ## bag, t, f
            
            ## bag,k,f
            tmp_abn, tmp_nor = super().sel_feats_by_dist(abn_feats, nor_feats, abn_dist, nor_dist)
            tmp_abn2, tmp_nor2 = super().sel_feats_sbs(abn_feats, nor_feats, abn_dist, nor_dist)
            super().is_equal(tmp_abn,tmp_abn2)
            super().is_equal(tmp_nor,tmp_nor2)
            
            sel_abn_feats.append(tmp_abn[..., None]) ## _, red1, 1 ;; _, red2, 1
            sel_nor_feats.append(tmp_nor[..., None]) ## _, red1, 1 ;; _, red2, 1
            
        L0 = lossfx['mpp'](ndata['anchors'], ndata["variances"], sel_abn_feats, sel_nor_feats)
        
        
        ## SCORE
        norm_scores = super().uncrop(ndata['norm_scors'], 'mean')
        norm_scores, _ = super().unbag(norm_scores)#ldata['labels']
        log.debug(f"{norm_scores.shape=}")
        L1 = lossfx["norm"]( norm_scores )
        
        #super().logdat([L0,L1])
        return super().merge(L0, L1) 
        
        ## DMF-based dist calculus, ret: [(b,t), (b,t)]
        #dists = [self.calc_mahalanobis_dist(norm_feat, anchor, var) for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"])]
        #dists = []
        #for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"]):
        #    #log.debug(f"{feats.shape}&{anchor.shape}&{var.shape} -> {dist.shape}")
        #    ## bs*nc,f,t -> bs*nc,t
        #    dist = super().calc_mahalanobis_dist(norm_feat, anchor[None, :, None], var[None, :, None])
        #    dist = super().uncrop(dist, 'mean') ## bs,t
        #    dists.append(dist)
        #    #log.debug(f"dist {dist.shape} max {dist.max()}  mean {dist.mean()}")#{dist}
        #    
        ### segment select for each BN layer
        #sel_abn_feats, sel_nor_feats = [], []
        #for feats, dists in zip(ndata["norm_feats"], dists):
        #    #bs*nc, f, t = feats.shape
        #    #assert bs == self.bs*self.ncrops
        #    feats = super().uncrop(feats, 'mean') ## bs, f, t
        #    abn_feats, nor_feats = super().unbag(feats, permute='021') ## bag, t, f
        #    
        #    abn_dists, nor_dists = super().unbag(dists) ## bag , t
        #    
        #    sel_nor_feat, sel_abn_feat = super().sel_feats_by_dist(abn_feats, nor_feats, abn_dists, nor_dists)
        #    sel_nor_feats.append(sel_nor_feat[..., None]) ## _, red1, 1 ;; _, red2, 1
        #    sel_abn_feats.append(sel_abn_feat[..., None]) ## _, red1, 1 ;; _, red2, 1
        
        ##########
        ## GENERAL 
        #sel_abn_feats, sel_nor_feats = [], []
        #for feats, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"]):
        #    #log.debug(f"{feats.shape}&{anchor.shape}&{var.shape} -> {dist.shape}")
        #    ## bs*nc,f,t -> bs*nc,t
        #    dists = super().calc_mahalanobis_dist(norm_feat, anchor[None, :, None], var[None, :, None])
        #    dists = super().uncrop(dists, 'mean') ## bs,t
        #    abn_dists, nor_dists = super().unbag(dists) ## bag , t
        #    
        #    feats = super().uncrop(feats, 'mean') ## bs, f, t
        #    abn_feats, nor_feats = super().unbag(feats, permute='021') ## bag, t, f
        #    
        #    sel_nor_feat, sel_abn_feat = super().sel_feats_by_dist(abn_feats, nor_feats, abn_dists, nor_dists)
        #    sel_nor_feats.append(sel_nor_feat[..., None]) ## _, red1, 1 ;; _, red1, 1
        #    sel_abn_feats.append(sel_abn_feat[..., None]) ## _, red2, 1 ;; _, red2, 1
        #
        #    log.debug(f"{abn_feats.shape=}, {nor_feats.shape=}")
        #    log.debug(f"{abn_dists.shape=}, {nor_dists.shape=}")
        #    log.debug(f"norm_sel {tmp_fnor_sel[..., None].shape} , abn_sel {tmp_fabn_sel[..., None].shape}")
        '''