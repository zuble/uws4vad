import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net.layers import BasePstFwd
from src.model.net.layers import Aggregate, SMlp

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)



class Network(nn.Module):
    def __init__(self, dfeat, _cfg, _cls: DictConfig = None):
        super().__init__()
        self.dfeat = sum(dfeat)
        
        ## idea
        #if _fm is not None:
        #    self.fmodulator = instantiate(_fm, dfeat=self.dfeat)
        #else: self.fmodulator = Aggregate(self.dfeat)
        
        self.aggregate = Aggregate(self.dfeat)
        self.do = nn.Dropout( _cfg.do )
        
        if _cls is not None:
            self.slcls = instantiate(_cls, dfeat=self.dfeat)
        else: raise Exception
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, t, f = x.shape
        
        rgbf = x[:, :, :self.dfeat]
        #audf = x[:, :, self.dfeat:]
        
        ## rgbf temporal refinement/ennahncment
        x_new = self.aggregate( rgbf.permute(0,2,1) ).permute(0,2,1) ## (b, t, f)
        x_new = self.do(x_new)
        log.debug(f'RTFM/aggregate {x_new.shape}')
        
        scors = self.sig( self.slcls(x_new) )

        return {
            'scors': scors, 
            'feats': x_new 
        }
        

class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)

    def train(self, ndata, ldata, lossfx):
        #super().logdat([ndata])
        
        ## FEAT
        feats = ndata['feats'] ## bs*nc,t,f
        
        abn_fmgnt, nor_fmgnt = self._get_mtrcs_magn(feats, labels=ldata["label"], apply_do=True)

        feats_pnc = self.uncrop(feats, force=True) ## bs,nc,t,f
        abn_feats, nor_feats = self.unbag(feats_pnc, labels=ldata["label"], permute='1023') ## nc,bag,t,f
        
        ## nc,bag,k,f ->  bag*nc, f (crop dim kept exposed) > key point, sufffers w/ nc=1 on ucf
        sel_abn_feats, idx_abn = self.gather_feats_per_crop(abn_feats, abn_fmgnt, avg=True)
        sel_nor_feats, idx_nor = self.gather_feats_per_crop(nor_feats, nor_fmgnt, avg=True)
        
        
        ## bag*nc,k,f ->  bag*nc, f (crop dim kept exposed)
        #sel_abn_feat, sel_nor_feats, idx_abn, idx_nor = super().sel_feats_ss(abn_feats, nor_feats, 
        #                                                                abn_fmgnt, nor_fmgnt,
        #                                                                per_crop=True, avg=True)
        
        log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        L0 = lossfx['mgnt'](sel_abn_feats, sel_nor_feats) ## video
        
        
        ####################
        ## experiment batch-level sel as bndfm
        #feats_uc = super().uncrop(feats, 'mean') ## bs, t, f 
        #abn_feats, nor_feats = super().unbag(feats_uc, ldata["label"]) ## bag, t, f
        #abn_fmgnt, nor_fmgnt = self._get_mtrcs_magn(feats, labels=ldata["label"]) ## bag,t
        #sel_abn_feats, sel_nor_feats = super().sel_feats_sbs(abn_feats, nor_feats, abn_fmgnt, nor_fmgnt, 0.1, 0.4)
        #
        #log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        #abn_feat = sel_abn_feats
        #nor_feat = sel_nor_feats
        ### 4 scores
        #idx_abn = super()._get_topk_idxs(abn_fmgnt, self.k, self.do)
        #idx_nor = super()._get_topk_idxs(nor_fmgnt, self.k, self.do)
        #L0 = lossfx['mgnt'](abn_feat, nor_feat) ## video
        ########################
        
        ## SCORE
        scors = super().uncrop( ndata['scors'], 'mean') ## bs*nc,t -> bs, t
        abn_scors, nor_scors = super().unbag(scors, ldata["label"]) ## bag, t
        vls_abn, vls_nor = super().sel_scors( abn_scors, nor_scors, idx_abn, idx_nor, avg=True) ## bag
        log.debug(f"{vls_abn.mean()=}  {vls_nor.mean()=}")
        
        ## !!!!! labels may be out of order for dl w/ bal_wgh != 0.5
        L1 = lossfx['bce'](torch.cat((vls_abn,vls_nor)), ldata['label']) ## vls
        #L1 = lossfx['bce'](vls_abn, torch.ones_like(vls_abn), 'abn') ## vls
        #L2 = lossfx['bce'](vls_nor, torch.zeros_like(vls_nor), 'nor') ## vls
        
        ## (bag*t)
        L2 = lossfx['smooth'](abn_scors.view(-1)) ## sls
        L3 = lossfx['spars'](abn_scors.view(-1), rtfm=True) ## sls
        
        return super().merge(L0, L1, L2, L3, ) #L4

        
    def infer(self, ndata):
        #log.debug(f"scors: {ndata['scors']=}")
        return ndata['scors']    
    


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