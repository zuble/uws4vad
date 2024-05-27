import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

from src.utils.logger import get_log
log = get_log(__name__)

#from ._layers import BasePstFwd


class Network(nn.Module):
    def __init__(self, feat_len: int, _cfg: DictConfig = None):
        super(Network, self).__init__()
        neurons = _cfg.neurons
        dropout = _cfg.do
        
        self.cls = nn.Sequential(
            nn.Linear(feat_len, neurons[0]),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(neurons[0], neurons[1]),
            #nn.ReLU(), ## bert-rtfm mentioned worse perform w/ relu
            nn.Dropout(dropout), 
            nn.Linear(neurons[1], 1),
            #nn.Sigmoid(),
            )
    def forward(self, x):
        b, t, f = x.shape
        return self.cls(x).view(b,t)


class NetPstFwd():
    def __init__(self, ): #, cfg: DictConfig
        super(NetPstFwd, self).__init__()
        #self.cfg = cfg
        
    def train(self, ndata, ldata, lossfx):
        log.debug(f"NETPSTFWD")
        
        
        super().rshp_out(ndata, '', 'mean') ## crop0
        log.debug(f" pos_rshp: {ndata[''].shape}")
        
        ## every lossfx returns a dict
        L0 = lossfx[''](ndata[''])
        L1 = lossfx[''](ndata[''])
        
        ## later indiv metered and summed to .backward 
        return L0.update(
            L1
            )
        


    def infer(self, ndata):
        pass
        ## output is excepted to be segment level 
        
        #log.debug(f"")
        #log.debug(f"slscores: {ndata['slscores']=}")
        #
        #return ndata['slscores']    