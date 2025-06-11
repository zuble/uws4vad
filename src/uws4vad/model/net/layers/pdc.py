import torch
import torch.nn as nn
import torch.nn.functional as F

from uws4vad.utils.logger import get_log
log = get_log(__name__)


class PiramidDilatedConv(nn.Module):
    def __init__(self, dfeat, rate=4):
        super().__init__()
        
        self.dfeat = dfeat ## orig in 2048
        self.dout = dfeat // rate ## 2048/4=512
        
        bn = nn.BatchNorm1d
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.dout, 
                    kernel_size=3,
                    stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(self.dout)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.dout, 
                    kernel_size=3,
                    stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(self.dout)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.dout, 
                    kernel_size=3,
                    stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(self.dout)
            # nn.dropout(0.7),
        )

    def forward(self, x):
        ## (b, dfeat, t)
        log.debug(f'PDC/x {x.shape}') 

        out1 = self.conv_1(x)
        out2 = self.conv_2(x)
        out3 = self.conv_3(x) ## (b, dout, t)
        log.debug(f'PDC/conv_123 {out1.shape, out2.shape, out3.shape}')
        out_cat = torch.cat((out1, out2, out3), dim = 1)
        log.debug(f'PDC/conv_fl123-cat {out_cat.shape}')
        
        return out_cat