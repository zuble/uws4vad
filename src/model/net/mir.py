import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.net.layers import BasePstFwd

from omegaconf.dictconfig import DictConfig
from src.utils.logger import get_log
log = get_log(__name__)

#from ._layers import BasePstFwd


class Network(nn.Module):
    def __init__(self, feat_len: int, _cfg: DictConfig):
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
        out = self.cls(x).view(b,t)
        return {
            'slscores': out
        }


class NetPstFwd(BasePstFwd):
    def __init__(self, cfg_dl):
        super(NetPstFwd, self).__init__(cfg_dl)
        
    def train(self, ndata, ldata, lossfx):
        
        super().rshp_out(ndata, 'slscores', 'mean') ## crop0
        #log.info(f" pos_rshp: {ndata['slscores'].shape}")
        
        L0 = lossfx['rnkg'](ndata['slscores'])
        
        return L0
        


    def infer(self, ndata):
        ## output is excepted to be segment level 
        
        #log.debug(f"")
        #log.debug(f"slscores: {ndata['slscores']=}")
        
        return ndata['slscores']    