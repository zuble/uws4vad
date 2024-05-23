import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

logger = None
def init(l):
    global logger
    logger = l


#######################################
## https://arxiv.org/pdf/2209.06435.pdf
## https://github.com/sakurada-cnq/salient_feature_anomaly/blob/main/network/video_classifier.py

class Cls(nn.Module):
    def __init__(self, in_feats, neurons):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_feats, neurons[0]),
            nn.Linear(neurons[0], neurons[1]),
            nn.Sigmoid()
        )
    def forward(self, x):
        #x = x.reshape(-1, x.shape[-1])
        return self.net(x)

class SelfAnttention(nn.Module):
    def __init__(self, in_feats, cfg_net):
        super().__init__()
        self.r = cfg_net.R
        self.da = cfg_net.DA
        self.drop_rate = cfg_net.DROP_RATE
        
        self.net = nn.Sequential(
            nn.Linear(in_feats, self.da),
            nn.Tanh(),
            nn.Linear(self.da, self.r),
            
            nn.Softmax2d() if cfg_net.SPAT else nn.Softmax(dim=1),
            nn.Dropout(self.drop_rate)
        )
    def forward(self, x):
        return self.net(x)


## SelfAttentionClassfier
class SAVcls_lstm(nn.Module):
    def __init__(self, rgbnf, audnf, cfg_net):
        super().__init__()
        self.bilstm = nn.LSTM(rgbnf, cfg_net.LSTM_DIM,
                            batch_first=True, bidirectional=True, num_layers=2)
        
        self.selfanttention = SelfAnttention(rgbnf, cfg_net)
        self.cls = Cls(rgbnf*cfg_net.R, cfg_net.CLS_NEURONS)
        
        self.ret_att = cfg_net.RET_ATT
        
    def forward(self,x):
        b, t, f = x.shape
        logger.debug(f'SAVcls/x: {x.shape}') 
        
        out,_ = self.bilstm(x)
        logger.debug(f'SAVcls/bilstm: {out.shape}')
        
        att_wght = self.selfanttention(out) ## (b, t, r)
        logger.debug(f'SAVcls/att_wght: {att_wght.shape}')
                
        m1 = (out * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        mxyz = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        logger.debug(f'SAVcls/m: {mxyz.shape}')
        
        x_cls = self.cls(mxyz).view(b)
        logger.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "scores": x_cls, 
            "attw": att_wght.squeeze(axis=0) if self.ret_att else None
            }
        
## SelfAttentionClassfier_nolstm
## SelfAttentionClassfier_nolstm_spational
class SAVcls(nn.Module):
    def __init__(self, rgbnf, audnf, cfg_net):
        super().__init__()
        self.bilstm = nn.LSTM(rgbnf, cfg_net.LSTM_DIM,
                            batch_first=True, bidirectional=True, num_layers=2)
        
        self.selfanttention = SelfAnttention(rgbnf, cfg_net)
        self.cls = Cls(rgbnf*cfg_net.R, cfg_net.CLS_NEURONS)
        
        self.ret_att = cfg_net.RET_ATT
        
    def forward(self,x):
        b, t, f = x.shape
        logger.debug(f'SAVcls/x: {x.shape}') 
        
        att_wght = self.selfanttention(x) ## (b, t, r)
        logger.debug(f'SAVcls/att_wght: {att_wght.shape}')
                
        m1 = (x * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (x * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (x * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        mxyz = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        logger.debug(f'SAVcls/m: {mxyz.shape}')
        
        x_cls = self.cls(mxyz).view(b)
        logger.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "scores": x_cls, 
            "attw": att_wght.squeeze(axis=0) if self.ret_att else None
            }

class VCls(nn.Module):
    def __init__(self, rgbnf, audnf, cfg_net):
        super().__init__()
        
        self.selfanttention = SelfAnttention(rgbnf, cfg_net)
        self.cls = Cls(rgbnf*cfg_net.R, cfg_net.CLS_NEURONS)
        
        self.ret_att = cfg_net.RET_ATT
        
    def forward(self, x):
        b, t, f = x.shape
        logger.debug(f'VCls/x: {x.shape}') 

        ## creates r att weight maps trough temporal axis of x feats
        att_wght = self.selfanttention(x) ## (b, t, r)
        logger.debug(f'VCls/att_wght: {att_wght.shape}')
        
        ## each feature in (f, t) is weighted sum for each r att_wght map (t, r) across temporal dimension
        ## resulting in r new temporal att weighted aggregated f features
        ## 3 VL representations
        m = torch.bmm( x.permute(0,2,1), att_wght) ## (b, f, t)*(b, t, r)=(b, f, r)
        logger.debug(f'VCls/m: {m.shape}')
        
        ## (b, nfeats*r)->(b)
        x_cls = self.cls( m.view(b,-1) ).view(b)
        logger.debug(f'VCls/x_cls: {x_cls.shape}')

        return {
            "scores": x_cls, 
            "attw": att_wght.squeeze(axis=0) if self.ret_att else None
            }


class NetPstFwd(BasePstFwd):
    def __init__(self, bs, ncrops, dvc):
        super().__init__(bs, ncrops, dvc)

    def train(self, ndata, ldata, lossfx):
        log.debug(f"")
        
        super().rshp_out(ndata, '', 'mean')
        #super().rshp_out(ndata, '', 'crop0')
        log.debug(f" pos_rshp: {ndata[''].shape}")
        loss0 = lossfx[''](ndata[''])
        self.updt_lbat('', loss0.item())
        
        return loss0 


    def infer(self, ndata):
        ## output is excepted to be segment level 
        log.debug(f"")
        log.debug(f"slscores: {ndata['slscores']=}")
        
        return ndata['slscores']    


if __name__ == "__main__":
    # Set the device to GPU index 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    audnf = 0
    rgbnf = 1024
    
    f = torch.ones((2, 32, audnf + rgbnf), device=device)
    
    ms = [VCls(nfeats=audnf+rgbnf)] #, SAVCls_spat(), SAVCls(), SAVCls_lstm()
    
    for net in ms:
        # Move the network to GPU
        net.to(device)
        
        # Initialize the network parameters (weights and biases)
        net.apply(lambda m: nn.init.normal_(m.weight) if hasattr(m, 'weight') else None)
        
        # Optionally print the model summary and parameters
        print(net)
        for name, param in net.named_parameters():
            print(f"{name}: {param.size()}")
        
        # Perform a forward pass to get the output from the network
        z = net(f)
        break  # Remove this if you want to loop through all networks
