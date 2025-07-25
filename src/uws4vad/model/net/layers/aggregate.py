import torch
import torch.nn as nn
import torch.nn.functional as F

from uws4vad.model.net.layers.pdc import PiramidDilatedConv
from uws4vad.model.net.layers.nl_block import NONLocalBlock1D

from uws4vad.common.registry import registry
from uws4vad.utils import get_log
log = get_log(__name__)


#########################################
## PDC + NLnet (RTFM)
## every fm :? (can we generalize)
## excepts b, t, f
## return b, t, f
## must assign self.din and self.dout
@registry.register_fm("aggregate")
class Aggregate(nn.Module):
    def __init__(self, din, rate=4, do=0.):
        super().__init__()
        
        self.din = din ## orig in 2048
        self.dhid = din // rate ## 2048/4=512  1024/4=256
        self.dout = din
        
        self.pdc = PiramidDilatedConv(din, rate)
        
        self.conv_4 = nn.Sequential( 
            nn.Conv1d(in_channels=self.din, 
                    out_channels=self.dhid, 
                    kernel_size=1,
                    stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.non_local = NONLocalBlock1D(self.dhid, sub_sample=False, bn_layer=True)
        
        self.conv_5 = nn.Sequential(
                    nn.Conv1d(in_channels=self.din, 
                            out_channels=self.din, 
                            kernel_size=3,
                            stride=1, padding=1, bias=False), # should we keep the bias?
                    nn.ReLU(),
                    nn.BatchNorm1d(self.din),
                    # nn.dropout(0.7)
                )
        self.do = nn.Dropout(do)
    def forward(self, x):
        x = x.permute(0,2,1)
        residual = x ## (b, din, t)
        
        ## local
        out_cat = self.pdc(x)
        ## global
        out_enc = self.conv_4(x) ## (b, dhid, t)
        out_nl = self.non_local(out_enc) ## (b, dhid, t)
        ## aggregate 
        out_agg = torch.cat((out_cat, out_nl), dim=1) ## (b, din, t)
        out = self.conv_5(out_agg) ## (b, din, t)
        ## enhance
        xe = out + residual ## (b, din, t)
        
        xe = self.do(xe)
        xe = xe.permute(0,2,1)
        
        log.debug(f'Agg/x {x.shape}') 
        log.debug(f'Agg/pdc {out_cat.shape}')
        log.debug(f'Agg/conv_4 {out_enc.shape}')
        log.debug(f'Agg/non_local {out_nl.shape}')
        log.debug(f'Agg/agg {out_agg.shape}')
        log.debug(f'Agg/conv_5 {out.shape}')
        log.debug(f'Agg/enhanced {xe.shape}')
        return xe
    
    #def forward(self, x):
    #    residual = out = x ## (b, din, t)
    #    log.debug(f'PDCNL/x {out.shape}') 
    #
    #    ## local
    #    out1 = self.conv_1(out)
    #    out2 = self.conv_2(out)
    #    out3 = self.conv_3(out) ## (b, dout, t)
    #    log.debug(f'PDCNL/conv_123 {out1.shape, out2.shape, out3.shape}')
    #    out_cat = torch.cat((out1, out2, out3), dim = 1)
    #    log.debug(f'PDCNL/conv_fl123-cat {out_cat.shape}')
    #    out_cat = self.pdc(out)
    #    log.debug(f'PDCNL/conv_fl123-cat {out_cat.shape}')
    #    
    #    ## global
    #    out = self.conv_4(out) ## (b, dout, t)
    #    log.debug(f'PDCNL/conv_4 {out.shape}')
    #    
    #    out = self.non_local(out) ## (b, dout, t)
    #    log.debug(f'PDCNL/non_local {out.shape}')
    #    
    #    ## aggregate features
    #    out = torch.cat((out_cat, out), dim=1) ## (b, din, t)
    #    log.debug(f'PDCNL/cat {out.shape}')
    #    out = self.conv_5(out) ## (b, din, t)
    #    log.debug(f'PDCNL/conv_5 {out.shape}')
    #    
    #    ## enhanced feature
    #    xe = out + residual ## (b, din, t)
    #    log.debug(f'PDCNL/enhanced {xe.shape}')
    #    return xe