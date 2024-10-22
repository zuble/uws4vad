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
        super().__init__()
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
        b, t, f = x.shape
        assert f == self.dfeat
        
        return self.conv_1( x.permute(0, 2, 1) ).permute(0, 2, 1)
    
############################
## TAD https://github.com/ktr-hubrt/WSAL/blob/master/models.py
## no dimension reduction
class Temporal2(nn.Module):
    def __init__(self, dfeat, ks=7):
        super().__init__()
        
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
    
## snippet-level anomalous attention
class Attention(nn.Module):
    def __init__(self, dfeat):
        super().__init__()
        self.dfeat = dfeat
        ## 512/512/1
        self.hdim = dfeat // 2
        
        self.cmlp = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.hdim, 
                    kernel_size=3,
                    padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            
            nn.Conv1d(in_channels=self.hdim, 
                    out_channels=self.hdim, 
                    kernel_size=3,
                    padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            
            nn.Conv1d(in_channels=self.hdim, 
                    out_channels=1, 
                    kernel_size=1),
            nn.Dropout(0.7),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, f, t = x.shape
        assert f == self.dfeat
        #log.info(f'Attention/attention0_conv0_bias: {self.att[0].bias.data()}')
        x = self.cmlp(x)
        return x