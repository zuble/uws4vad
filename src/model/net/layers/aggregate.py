import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)


#########################################
## PDC + NLnet (RTFM)
## excepts b, f, t
## return b, f, t
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, 
                        out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, 
                        out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                            out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, 
                            out_channels=self.inter_channels,
                            kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, 
                        out_channels=self.inter_channels,
                        kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                            inter_channels=inter_channels,
                                            dimension=1, sub_sample=sub_sample,
                                            bn_layer=bn_layer)


class Aggregate(nn.Module):
    def __init__(self, dfeat, rate=4):
        super(Aggregate, self).__init__()
        
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
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.dout, 
                    kernel_size=1,
                    stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=self.dfeat, 
                    out_channels=self.dfeat, 
                    kernel_size=3,
                    stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(self.dfeat),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(self.dout, sub_sample=False, bn_layer=True)


    def forward(self, x):
        residual = out = x ## (b, dfeat, t)
        log.debug(f'Aggregate/x {out.shape}') 

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out) ## (b, dout, t)
        log.debug(f'Aggregate/conv_123 {out1.shape, out2.shape, out3.shape}')
        out_cat = torch.cat((out1, out2, out3), dim = 1)
        log.debug(f'Aggregate/conv_fl123-cat {out_cat.shape}')
        
        ## global
        out = self.conv_4(out) ## (b, dout, t)
        log.debug(f'Aggregate/conv_4 {out.shape}')
        out = self.non_local(out) ## (b, dout, t)
        log.debug(f'Aggregate/non_local {out.shape}')
        
        ## aggregate features
        out = torch.cat((out_cat, out), dim=1) ## (b, dfeat, t)
        log.debug(f'Aggregate/cat {out.shape}')
        
        out = self.conv_5(out) ## (b, dfeat, t)
        log.debug(f'Aggregate/conv_5 {out.shape}')
        
        ## enhanced feature
        xe = out + residual ## (b, dfeat, t)
        log.debug(f'Aggregate/enhanced {xe.shape}')
        return xe