import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net.layers import BasePstFwd

from omegaconf.dictconfig import DictConfig
from src.utils.logger import get_log
log = get_log(__name__)


class Network(nn.Module):
    def __init__(self, dfeat: int, _cfg: DictConfig):
        super().__init__()
        self.dfeat = dfeat
    
    def forward(self, x):
        return 

class NetPstFwd(BasePstFwd):
    def __init__(self, bs, ncrops):
        super().__init__(bs, ncrops)

    def train(self, ndata, ldata, lossfx):
        super().rshp_out(ndata, '', 'mean') ## crop0
        x = super().rshp_out2(x, 'mean')
        
        L0 = lossfx['']()
        L1 = lossfx['']()
        
        return L0.update(L1)

    def infer(self, ndata):

        
        return ndata['slscores']    