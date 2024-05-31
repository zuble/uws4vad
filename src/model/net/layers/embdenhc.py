import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)

## every Temporal ennhcent:
##  excepts (b,f,t)
##  outputs (b,f,t)

#torch.nn.Conv1d(in_channels, out_channels, kernel_size, 
#                stride=1, 
#                padding=0, 
#                dilation=1, 
#                groups=1, 
#                bias=True, padding_mode='zeros')

#######################
## BN https://github.com/cool-xuan/BN-WVAD/blob/main/models/model.py
## occurs a dimensional reduction
class Temporal(nn.Module):
    def __init__(self, dfeat, out_len):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=dfeat, 
                    out_channels=out_len, 
                    kernel_size=3,
                    stride=1, 
                    padding=1),
            nn.ReLU(),
        )
        self.dfeat = dfeat
    def forward(self, x):
        b, f, t = x.shape
        assert f == self.dfeat
        return self.conv_1(x)
    
############################
## TAD https://github.com/ktr-hubrt/WSAL/blob/master/models.py
## no dimension reduction
class Temporal2(nn.Module):
    def __init__(self, dfeat, ks=7):
        super(Temporal2, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=dfeat,
                            out_channels=dfeat,
                            kernel_size=ks)
        self.dfeat = dfeat
        self.ks = ks
            
    def forward(self, x):
        b, f, t = x.shape
        assert f == self.dfeat
        
        x = F.pad(x, (self.ks//2, self.ks//2), mode='replicate')
        x = self.conv(x)
        return x
    
