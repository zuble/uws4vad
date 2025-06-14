import torch
import torch.nn as nn

from uws4vad.model.pstfwd.utils import PstFwdUtils
from uws4vad.model.net.layers import GlanceFocus

from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate as instantiate
from uws4vad.utils import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__( self,  
        dfeat: int, 
        _cfg: DictConfig, 
        rgs = None
    ): 
        super().__init__()
        log.error(_cfg)
        self.dfeat = sum(dfeat)
        self.mag_ratio = _cfg.mag_ratio
        
        init_dim, *_, last_dim = _cfg.fm.dims
        
        #if _cfg.fm.get("_target_"):
        #    self.fmodulator = instantiate(_cfg.fm, dfeat=self.dfeat)
        #else: 
        self.fmodulator = GlanceFocus(self.dfeat)
        
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
        log.debug(f"FAM x:{list(x.shape)} x_m{list(x_m.shape)} {x_m=}")
        return x_m
        
    def forward(self, x):
        x = x.permute(0, 2, 1) ## bs*nc, f, t
        
        x_f = self.to_tokens(x)
        x_m = self.fam(x)
        x_m = self.to_mag(x_m)
        ## fmagn enha
        x_fme = x_f + self.mag_ratio*x_m ## bs*nc, init_dim, t
        
        log.debug(f"ENC {x_f.shape=} {x_m.shape=} {x_fme.shape=}")
        log.debug(f"ENC {x_f=}\n{x_m=}\n{x_fme=}")
        
        x_fm = self.fmodulator(x_fme)
        x_fm = x_fm.permute(0, 2, 1) ## bs*nc, t, f 
        log.debug(f"FM {x_fm.shape=}")
        
        x_out = self.to_logits(x_fm) 
        scores = self.sig( self.fc(x_out) ) ## bs*nc, t, 1
        scores = scores.squeeze(-1)
        return {
            'scores': scores,
            'feats': x_out
        }

class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        self.sig = nn.Sigmoid()
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        #log.debug(f"{scores=}")
        #scores = self.sig(scores)
        #log.debug(f"{_scores=}")
        return scores

