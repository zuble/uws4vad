import torch
import torch.nn as nn

from src.model.net.layers import BasePstFwd
from src.model.net.layers import GlanceFocus

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__( self, dfeat, 
        mag_ratio = 0.1,
        dims = [64, 128, 1024],
        _cfg = None,
        _fm = None,
    ):
        super().__init__()
        
        self.dfeat = sum(dfeat)
        self.mag_ratio = mag_ratio

        #if _cfg.dims is not None:
        #    dims = _cfg.dims
        init_dim, *_, last_dim = dims
        
        if _fm is not None:
            self.fmodulator = instantiate(_fm, dfeat=self.dfeat)
        else: self.fmodulator = GlanceFocus(self.dfeat)
        
        ## assert dim and compowr save
        ## feature map dimension from 𝐶 in 𝐹𝐹𝐴𝑀 to 𝐶/32.
        self.to_tokens = nn.Conv1d(self.dfeat, init_dim, kernel_size=3, stride = 1, padding = 1)
        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)
        
        self.to_logits = nn.Sequential( nn.LayerNorm(last_dim) )
        self.fc = nn.Linear(last_dim, 1)
        self.sig = nn.Sigmoid()
        
    def fam(self, x):
        ## Feature Amplification Mechanism (FAM)
        x_m = torch.linalg.norm(x, ord=2, dim=1)[:,None,:] ## bs*nc, 1, t
        #x_m2 = torch.norm(x_f, p=2, dim=1)[:,None, :] ## bs*nc, 1, t
        #assert torch.all(x_m.eq(x_m2)) == True
        #log.debug(f"{x_m2.shape} {x_m2}")
        log.info(f"FAM {x.shape} {x_m.shape} {x_m}")
        return x_m
        
    def forward(self, x):
        x = x.permute(0, 2, 1) ## bs*nc, f, t
        
        x_f = self.to_tokens(x)
        x_m = self.fam(x)
        x_m = self.to_mag(x_m)
        ## fmagn enha
        x_fme = x_f + self.mag_ratio*x_m ## bs*nc, init_dim, t
        
        log.info(f"ENC {x_f.shape=} {x_m.shape=} {x_fme.shape=}")
        log.debug(f"ENC {x_f}\n{x_m}\n{x_fme}")
        
        x_fm = self.fmodulator(x_fme)
        x_fm = x_fm.permute(0, 2, 1) ## bs*nc, t, f 
        log.info(f"FM {x_fm.shape=}")
        
        x_out = self.to_logits(x_fm) 
        scors = self.sig( self.fc(x_out) ) ## bs*nc, t, 1
        return {
            'scors': scors,
            'feats': x_out
        }

class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)

    def train(self, ndata, ldata, lossfx):
        ## FEAT
        feats = ndata['feats'] ## (bs*nc, t, f)

        ## bag*nc,k,f ->  bag*nc, f (crop dim kept exposed)
        abn_feat, nor_feat, idx_abn, idx_nor = super().sel_feats_by_magn(feats, 
                                                                        labels=ldata["label"],
                                                                        per_crop=True, avg=True)
        log.debug(f"{abn_feat.shape=} {nor_feat.shape=}")
        
        L0 = lossfx['mc'](abn_feats, nor_feats)

        ## SCORE
        #scors = ndata['scors'].shape ## bs*nc,t
        scors = super().uncrop( ndata['scors'], 'mean') ## bs, t
        abn_scors, nor_scors = super().unbag(scors, ldata["label"]) ## bag, t
        vls_abn = super().sel_scors( abn_scors, idx_abn, avg=True) ## bag
        vls_nor = super().sel_scors( nor_scors, idx_nor, avg=True) ## bag
        
        log.debug(f"{abn_scors.shape=} {nor_scors.shape=}")
        log.debug(f"{vls_abn.shape=} {vls_nor.shape=}")
        
        L1 = lossfx['bce'](torch.cat((vls_nor, vls_abn)), ldata['label']) ## vls


        ## (bag*t)
        #L2 = lossfx['smooth'](abn_scors.view(-1)) ## sls
        #L3 = lossfx['spars'](abn_scors.view(-1), rtfm=True) ## sls
        
        return super().merge(L0, L1, L2)
