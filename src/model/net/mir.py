import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net.layers import SMlp
from src.model.pstfwd.utils import PstFwdUtils

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig, rgs = None):
        super().__init__()
        self.dfeat = sum(dfeat)
        self.rgs = rgs
        #self.rgs = SMlp( self.dfeat, _cfg.rate, _cfg.do) if rgs is None else rgs
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b, t, f = x.shape
        out = self.sig( self.rgs(x) )
        
        log.debug(f"{x.shape} -> {out.shape}")
        return {
            'scores': out
        }
    
class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata):  
        return ndata['scores'] 
