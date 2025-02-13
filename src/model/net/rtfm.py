import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.pstfwd.utils import PstFwdUtils
from src.model.net.layers import Aggregate, SMlp, SConv, Transformer, Temporal

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils import get_log
log = get_log(__name__)



class Network(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig, rgs = None):
        super().__init__()
        self.dfeat = sum(dfeat)
        
        ## idea
        #if _fm is not None:
        #    self.fmodulator = instantiate(_fm, dfeat=self.dfeat)
        #else: self.fmodulator = Aggregate(self.dfeat)
        
        self.fm = Aggregate(self.dfeat, do=_cfg.do)
        #self.emb_hdim = 512 #self.dfeat//2
        #self.embedding = Temporal(self.dfeat, self.emb_hdim)
        #self.fm = Transformer(self.emb_hdim , depth=2, heads=4, dim_head=128, ff_hdim=self.emb_hdim, dropout=0.)
        
        self.slrgs = rgs
        #self.slrgs = SConv(dfeat=self.dfeat, ks=7)
        #if _cls is not None:
        #    self.slrgs = instantiate(_cls, dfeat=self.dfeat)
        #else: raise Exception
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, t, f = x.shape
        
        rgbf = x[:, :, :self.dfeat]
        #audf = x[:, :, self.dfeat:]
        
        #x = rgbf
        #x = self.embedding(x)
        x_new = self.fm( rgbf ) ## (b, t, f)
        log.debug(f'fm {x_new.shape}')
        
        scors = self.sig( self.slrgs(x_new) )

        return { ## standard output
            'scores': scors, 
            'feats': x_new 
        }
        

class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        return scores
    
        ## magnitudes
        ## 1
        ##feat_magn = torch.norm(feats, p=2, dim=2)  ## bs*nc, t
        #feat_magn = torch.linalg.norm(feats, ord=2, dim=2) ## bs*nc, t
        #feat_magn = super().uncrop(feat_magn, 'mean') ## (bs, t)
        #abn_fmgnt, nor_fmgnt = super().unbag(feat_magn, ldata["label"] ) ## bag, t

        ## 2/3
        #abn_fmgnt, nor_fmgnt = super()._get_mtrcs_magn(feats)
        #feats = super().uncrop(feats, force=True) ## bs,nc,t,f
        #abn_feats, nor_feats = super().unbag(feats, ldata["label"], '1023') ## nc,bag,t,f
        ## selection
        #abn_feat, idx_abn = super().sel_feats_by_magn(abn_feats, abn_fmgnt, avg=True) 
        #nor_feat, idx_nor = super().sel_feats_by_magn(nor_feats, nor_fmgnt, avg=True) 
        
        ##########
        ## GENERAL
        ## 3
        #assert feats.ndim in [3, 4], f"feats.ndim {feats.ndim} not in [3, 4]"
        #magnitudes_drop = magnitudes * self.do(torch.ones_like(magnitudes))
        #idx = torch.topk(magnitudes_drop, k, dim=1)[1]  ## (bag, k)
        #idx_feat = idx.unsqueeze(2).expand(-1, -1, f)  ## (bag, k, f)
        #
        #if per_crop:
        #    assert feats.ndim == 4
        #    nc, bs, t, f = feats.shape
        #    
        #    sel_feats = torch.zeros(0, device=feats.device)
        #    for i, feat in enumerate(feats):
        #        tmp = torch.gather(feat, dim=1, index=idx_feat)  # (bag, k, f)
        #        sel_feats = torch.cat((sel_feats, tmp))
        #    
        #    #log.debug(f"{feats.shape=} gather {idx_feat.shape=} over nc dim -> {sel_feats.shape=} ")
        #    if avg: ## (bag*nc, f)
        #        return sel_feats.mean(dim=1), idx 
        #    else: ## (bag*nc, k, f)
        #        return sel_feats, idx 
        #
        #elif feats.ndim == 3:
        #    bag, t, f = feats.shape
        #    
        #    sel_feats = torch.gather(feats, dim=1, index=idx_feat)  # (bag, k, f)
        #    #log.debug(f"{feats.shape=} gather {idx_feat.shape=} -> {sel_feats.shape=} ")
        #    if avg: ## (bag, f)
        #        return sel_feats.mean(dim=1), idx 
        #    else:
        #        return sel_feats, idx
        #if avg: ## (bag, f)
        #    return sel_feats.mean(dim=1), idx
        #else:
        #    return sel_feats, idx