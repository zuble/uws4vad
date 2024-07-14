import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net.layers import BasePstFwd, SMlp

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig, _cls: DictConfig = None):
        super(Network, self).__init__()
        self.dfeat = sum(dfeat)
        
        if _cls is not None:
            self.cls = instantiate(_cls, dfeat=self.dfeat)            
        else: 
            self.cls = SMlp( self.dfeat, _cfg.rate, _cfg.do)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b, t, f = x.shape
        out = self.sig( self.cls(x) )
        log.warning(f"{x.shape} -> {out.shape}")
        return {
            'sls': out
        }

class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)
        
    def train(self, ndata, ldata, lossfx):
        super().logdat(ndata)
        
        scores = super().uncrop(ndata['sls'], 'mean')
        #log.info(f" pos_rshp: {ndata['sls'].shape}")
        
        L0 = lossfx['rnkg'](scores, ldata["label"])
        
        return L0
        
    def infer(self, ndata):        
        return ndata['sls']    