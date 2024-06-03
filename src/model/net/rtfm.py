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
        
        sls = self.sig( self.slcls(x_new) )
        #log.debug(f'RTFM/sls {sls.shape} '{data.ds.{data.ds.{data.ds.{data.ds.{data.ds.)
        #sls = torch.nan_to_num(sls, nan=0, posinf=0, neginf=0)
        
        return {
            'sls': sls,
            'feats': x_new
        }
        

class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)

    def train(self, ndata, ldata, lossfx):
        
        ## FEATS
        #super().rshp_out(ndata, 'feats', 'mean')
        ##super().rshp_out(ndata, '', 'crop0')
        #log.debug(f" pos_rshp: {ndata['feats'].shape}")
        t, f = ndata['feats'].shape[1:3]
        feats = ndata['feats'] ## (bs*ncrops, t, f)

        ## magnitudes
        feat_magn = torch.norm(feats, p=2, dim=2)  ## ((bs)*ncrops, t)
        feat_magn = super().rshp_out2(feat_magn, 'mean') ## (bs, t)
        abnr_fmagn = feat_magn[self.bs//2:]  ## (bs//2, t) 
        norm_fmagn = feat_magn[0:self.bs//2]  ## (bs//2, t) 
        
        ## feats
        abnr_feats, norm_feats = super().split_per_crop(feats,t,f)
        log.debug(f"{abnr_feats.shape=} {norm_feats.shape=}")
        
        
        ## scores 
        super().rshp_out(ndata, 'sls', 'mean')
        #super().rshp_out(ndata, '', 'crop0')
        log.debug(f" pos_rshp: {ndata['sls'].shape}")
        sls = ndata['sls'] ## (bs, t)

        abnr_sls = sls[self.bs//2:] ## (bs/2, t)
        norm_sls = sls[0:self.bs//2] ## (bs/2, t)        
        
        ########
        ## LOSS 
        ## levarage frist returned dict and update only
        L0 = lossfx['mgnt'](abnr_fmagn, norm_fmagn, abnr_feats, norm_feats, abnr_sls, norm_sls, ldata)

        tmp_abnr_sls = abnr_sls.view(-1) ## (bs/2*t)
        L1 = lossfx['smooth'](tmp_abnr_sls)
        L2 = lossfx['spars'](tmp_abnr_sls, rtfm=True)
        
        return super().merge(L0, L1, L2)


    def infer(self, ndata):
        ## output is excepted to be segment level 
        log.debug(f"")
        log.debug(f"sls: {ndata['sls']=}")
        
        return ndata['sls']    

    
    