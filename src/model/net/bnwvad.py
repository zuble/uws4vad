import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.net.layers import BasePstFwd

from hydra.utils import instantiate as instantiate
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

        tmp_ones = torch.ones(n)#.cuda()
        tmp_n = torch.linspace(1, n, n)#.cuda()
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
        outputs.append(x)
        return outputs


class Temporal(nn.Module):
    def __init__(self, dfeat, dout):
        super(Temporal, self).__init__()
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
    def __init__(self, dfeat, _cfg):
        super().__init__()
        
        self.dfeat = sum(dfeat)
        #self.ncrops = _cfg.ncrops
        
        self.embedding = Temporal(self.dfeat, _cfg.emb_dim)
        self.selfatt = Transformer(_cfg.emb_dim, _cfg.depth, _cfg.heads, _cfg.head_dim, _cfg.mlp_dim, _cfg.do)
        self.normal_head = NormalHead(in_feats=_cfg.mlp_dim, ratios=_cfg.nh_dimratios, ks=_cfg.nh_ks)

    def forward(self, x):
        #if len(x.size()) == 4: b, n, t, d = x.size(); x = x.reshape(b * n, t, d)
        #else: b, t, d = x.size(); n = 1
            
        x = self.embedding(x) ## b, t, 512
        x = self.selfatt(x) ## b, t, 512
        
        nh_res = self.normal_head( x.permute(0, 2, 1) )
        anchors = [bn.running_mean for bn in self.normal_head.bns] ## [red1 ;; red2]
        variances = [bn.running_var for bn in self.normal_head.bns] ## [red1 ;; red2]

        return {
            'anchors': anchors, ## (red1, red2)
            'variances': variances, ## (red1, red2)
            'norm_scors': nh_res[-1], ## b,1,t 
            'norm_feats': nh_res[:-1] ## [(b, red1, t) ;; (b, red2, t)]          
        }


class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)
        
        self.slsr = self._cfg.sls_ratio
        self.blsr = self._cfg.bls_ratio
        
    def logd(self, d):
        for key, value in d.items():
            if type(value) is list:
                for v in value:
                    log.debug(f"{key} {v.shape}")
            else:
                log.debug(f"{key} {value.shape}")
                
    def pos_neg_select(self, feats, distance):
        bs, c, t = feats.shape
        #bs, t = distance.shape
        
        ## assigns a dynamic k value for dists selection
        select_num_sample = int(t * self.slsr) ## for dists
        select_num_batch = int(bs // 2 * t * self.blsr)
        #log.warning(f"sample_sel: {select_num_sample} / batch_sel: {select_num_batch}")
        
        feats = feats.view(bs, self.ncrops, c, t).mean(1) ## b, c, t
        
        ###########
        ## ABNORMAL
        ## SLS: video/sammple-wise mask
        abn_distance = distance[bs // 2:] ## bag, t
        topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
        mask_sel_abn_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
        mask_sel_abn_sample.scatter_(1, topk_abnormal_sample, True) ## bag, t
        
        ## BLS: bag-wise mask
        abn_dist_flat = abn_distance.reshape(-1) ## bag * t
        topk_abnormal_batch = torch.topk(abn_dist_flat, select_num_batch, dim=-1)[1]
        mask_sel_abn_batch = torch.zeros_like(abn_dist_flat, dtype=torch.bool)
        mask_sel_abn_batch.scatter_(0, topk_abnormal_batch, True) ## bag
        
        ## what to do with those that are abnormal
        ## grab and orientate to a specif subspcae
        
        ## SBS
        mask_sel_abn = mask_sel_abn_batch | mask_sel_abn_sample.reshape(-1)
        
        ## ABNORMAL
        abn_feats = feats[bs // 2:].permute(0, 2, 1) ## bag, t, c
        abn_feats_flatten = abn_feats.reshape(-1, c) ## bag*t, c
        sel_abn_feats = abn_feats_flatten[mask_sel_abn]
        
        
        ## NORMAL
        nor_distance = distance[:bs // 2] ## bag , t
        nor_feats = feats[:bs // 2].permute(0, 2, 1) ## bag, t, c
        
        num_sel_abn = torch.sum(mask_sel_abn)
        
        k_nor = int(num_sel_abn / (bs // 2)) + 1
        topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]
        sel_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c))
        sel_nor_feats = sel_nor_feats.permute(1, 0, 2).reshape(-1, c)
        sel_nor_feats = sel_nor_feats[:num_sel_abn] ## match with abn
        
        ## (num_sel_abn, c) 
        return sel_nor_feats, sel_abn_feats
    
    def get_mahalanobis_dist(self, feats, anchor, var): 
        d = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
        return d #super().rshp_out2(d, 'mean')
    
    def train(self, ndata, ldata, lossfx):
        self.logd(ndata)
        
        ## DMF-based dist calculus, ret: [(b,t), (b,t)]
        #dists = [self.get_mahalanobis_dist(norm_feat, anchor, var) for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"])]
        dists = []
        for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"]):
            dist = self.get_mahalanobis_dist(norm_feat, anchor, var)
            #log.debug(f"{dist.shape=} {dist=}")
            dist = super().rshp_out2(dist, 'mean')
            log.debug(f"{dist.shape} {dist.max()} {dist.mean()}")
            dists.append(dist)
        
        ## segment select for each BN layer
        fnor_sel, fabn_sel = [], []
        for feat, distance in zip(ndata["norm_feats"], dists):
            tmp_fnor_sel, tmp_fabn_sel = self.pos_neg_select(feat, distance)
            fnor_sel.append(tmp_fnor_sel[..., None]); fabn_sel.append(tmp_fabn_sel[..., None])
            ## _, red1, 1 ;; _, red1, 1
            ## _, red2, 1 ;; _, red2, 1
            log.debug(f"norm_sel {tmp_fnor_sel[..., None].shape} , abn_sel {tmp_fabn_sel[..., None].shape}")
            
        
        L0 = lossfx['mpp'](ndata['anchors'], ndata["variances"], fnor_sel, fabn_sel)
        
        super().uncrop(ndata,'norm_scors', 'mean')
        L1 = lossfx["norm"]( ndata["norm_scors"][0:self.bs // 2] )
        
        return super().merge(L0, L1) 


    def infer(self, ndata):
        #dists = [self.get_mahalanobis_dist(norm_feat, anchor, var) for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"])]
        dists = []
        for norm_feat, anchor, var in zip(ndata["norm_feats"], ndata["anchors"], ndata["variances"]):
            dist = self.get_mahalanobis_dist(norm_feat, anchor, var)
            log.debug(f"dist pre {dist.shape}")
            dist = super().rshp_out2(dist, 'mean')
            log.debug(f"dist {dist.shape}")
            dists.append(dist)
        dists_sum = sum(dists)
        log.error(f"DISTS {dists[0].max()} {dists[1].max()}")
        log.error(f"DISTS SUM {dists_sum.max()}")
        
        super().uncrop(ndata,'norm_scors', 'mean')
        log.error(f"NORM {ndata['norm_scors'].max()}")
        
        return ndata["norm_scors"] #* dists_sum 





#x = torch.randn((4, 8, 1024)).cuda()
#
#net = WSAD(1024,'Train').cuda()
#_ = net(x)