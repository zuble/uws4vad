import torch
import torch.nn as nn

log = None
def init(l):
    global log
    log = l
    
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
    def __init__(self, feat_len, out_len):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=feat_len, 
                    out_channels=out_len, 
                    kernel_size=3,
                    stride=1, 
                    padding=1),
            nn.ReLU(),
        )
        self.feat_len = feat_len
    def forward(self, x):
        b, f, t = x.shape
        assert f == self.feat_len
        return self.conv_1(x)
    
############################
## TAD https://github.com/ktr-hubrt/WSAL/blob/master/models.py
## no dimension reduction
class Temporal2(nn.Module):
    def __init__(self, feat_len, ks=7):
        super(Temporal2, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=feat_len,
                            out_channels=feat_len,
                            kernel_size=ks)
        self.feat_len = feat_len
        self.ks = ks
            
    def forward(self, x):
        b, f, t = x.shape
        assert f == self.feat_len
        
        x = F.pad(x, (self.ks//2, self.ks//2), mode='replicate')
        x = self.conv(x)
        return x
    
#########################################
## https://arxiv.org/pdf/2309.16309v1.pdf
## https://github.com/2023-MindSpore-4/Code4/blob/main/WS-VAD-mindspore-main/non_local.py
## taken from RTFM
## PDC + NLnet
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
    def __init__(self, feat_len):
        super(Aggregate, self).__init__()
        
        self.feat_len = feat_len ## orig in 2048
        self.out_len = feat_len // 4 ## 2048/4=512
        
        bn = nn.BatchNorm1d
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_len, 
                    out_channels=self.out_len, 
                    kernel_size=3,
                    stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(self.out_len)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_len, 
                    out_channels=self.out_len, 
                    kernel_size=3,
                    stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(self.out_len)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_len, 
                    out_channels=self.out_len, 
                    kernel_size=3,
                    stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(self.out_len)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_len, 
                    out_channels=self.out_len, 
                    kernel_size=1,
                    stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=self.feat_len, 
                    out_channels=self.feat_len, 
                    kernel_size=3,
                    stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(self.feat_len),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(self.out_len, sub_sample=False, bn_layer=True)


    def forward(self, x):
        residual = out = x ## (b, feat_len, t)
        log.debug(f'Aggregate/x {out.shape}') 

        out1 = self.conv_1(out)
        out2 = self.conv_2(out)
        out3 = self.conv_3(out) ## (b, out_len, t)
        log.debug(f'Aggregate/conv_123 {out1.shape, out2.shape, out3.shape}')
        out_cat = torch.cat((out1, out2, out3), dim = 1)
        log.debug(f'Aggregate/conv_fl123-cat {out_cat.shape}')
        
        ## global
        out = self.conv_4(out) ## (b, out_len, t)
        log.debug(f'Aggregate/conv_4 {out.shape}')
        out = self.non_local(out) ## (b, out_len, t)
        log.debug(f'Aggregate/non_local {out.shape}')
        
        ## aggregate features
        out = torch.cat((out_cat, out), dim=1) ## (b, feat_len, t)
        log.debug(f'Aggregate/cat {out.shape}')
        
        out = self.conv_5(out) ## (b, feat_len, t)
        log.debug(f'Aggregate/conv_5 {out.shape}')
        
        ## enhanced feature
        xe = out + residual ## (b, feat_len, t)
        log.debug(f'Aggregate/enhanced {xe.shape}')
        return xe