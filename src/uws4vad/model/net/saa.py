import torch
import torch.nn as nn
import torch.nn.functional as F

from uws4vad.model.pstfwd.utils import PstFwdUtils
from uws4vad.model.net.layers import Aggregate, Attention, SMlp

from omegaconf.dictconfig import DictConfig
from uws4vad.utils import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig, rgs):
        super().__init__()
        self.dfeat = sum(dfeat)
        
        ## idea
        #if _fm is not None:
        #    self.fmodulator = instantiate(_fm, dfeat=self.dfeat)
        #else: self.fmodulator = Aggregate(self.dfeat)
        
        self.aggregate = Aggregate(self.dfeat)
        self.do = nn.Dropout( _cfg.do )
        
        self.attn = Attention(self.dfeat)
        
        self.slrgs = rgs
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        b, t, f = x.shape
        
        rgbf = x[:, :, :self.dfeat]
        #audf = x[:, :, self.dfeat:]
        
        ## rgbf temporal refinement/ennahncment
        x_new = self.aggregate( rgbf.permute(0,2,1) ) ## (b, f, t)
        x_new = self.do(x_new)
        log.debug(f'SAA/aggregate {x_new.shape}')
        
        ########
        #xva_new = np.concatenate((xv_new, audf.transpose((0, 2, 1)) ), axis=1)
        ########
        
        ## for each segment feature -> 1 att value
        #attw = np.zeros((b,t), ctx=self.dvc[0])
        attw = self.attn(x_new).squeeze(dim=1) ## (b, 1, t) > (b, t)
        log.debug(f'MindSpore/attw {attw.shape}')
        
        
        scors = self.sig( self.slrgs(x_new.permute(0,2,1)) )
        
        return { ## standard output
            'scores': scors, 
            #'feats': x_new,
            'attw': attw
        }


class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata):
        self.pfu.logdat(ndata)
        assert ndata['scores'].shape[0] == 1 ## not dealing with crops at infer atm
        
        tmp = ndata['attw'] * ndata['scores'] ## SL_MindSpore: eq(16,4)
        
        log.debug(f'attw: {ndata["attw"]=}')
        log.debug(f"scores: {ndata['scores']=}")
        
        return tmp