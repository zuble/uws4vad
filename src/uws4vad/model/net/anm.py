import torch
import torch.nn as nn
import torch.nn.functional as F

from uws4vad.model.net.layers import Temporal
from uws4vad.model.pstfwd.utils import PstFwdUtils

from hydra.utils import instantiate as instantiate
from uws4vad.utils import get_log
log = get_log(__name__)


#######################################
## https://arxiv.org/pdf/2209.06435.pdf
## https://github.com/sakurada-cnq/salient_feature_anomaly/blob/main/network/video_classifier.py

class Cls(nn.Module):
    def __init__(self, dfeat, cls_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dfeat, cls_dim),
            nn.Linear(cls_dim, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        #x = x.reshape(-1, x.shape[-1])
        return self.net(x)

class SelfAnttention(nn.Module):
    def __init__(self, dfeat, att_dim, r, do, spat=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dfeat, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, r),
            
            nn.Softmax2d() if spat else nn.Softmax(dim=1),
            nn.Dropout(do)
        )
    def forward(self, x):
        return self.net(x)


## SelfAttentionClassfier
class NetworkSAVCls_lstm(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)
        self.r = _cfg.r
        
        self.bilstm = nn.LSTM(self.dfeat, _cfg.lstm_dim,
                            batch_first=True, 
                            bidirectional=_cfg.lstm_bd, 
                            num_layers=2)
        lstm_out = _cfg.lstm_dim if not _cfg.lstm_bd else _cfg.lstm_dim*2

        self.santt = SelfAnttention( lstm_out,_cfg.att_dim,_cfg.r,_cfg.do,_cfg.spat)
        self.cls = Cls(lstm_out*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self,x):
        b, t, f = x.shape
        log.debug(f'SAVcls/x: {x.shape}')
        
        out,_ = self.bilstm(x)
        log.debug(f'SAVcls/bilstm: {out.shape}')
        
        att_wght = self.santt(out) ## (b, t, r)
        log.debug(f'SAVcls/att_wght: {att_wght.shape}')
        
        mxyz = [ (out * att_wght[:,:,0].unsqueeze(2)).sum(dim=1) for i in range(att_wght.shape[2])]
        mcat = torch.cat(mxyz, dim=1)  # (b, r*f)
        #m1 = (out * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        #m2 = (out * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        #m3 = (out * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        #mcat = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        log.debug(f'SAVcls/mcat: {mcat.shape}')
        
        x_cls = self.cls(mcat).view(b)
        log.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "vls": x_cls, 
            "vlf": mcat, #.permute(0,2,1), ## b, r, f
            "attw": att_wght.permute(0,2,1).squeeze(dim=0) #if self.ret_att else None
            }
        
## SelfAttentionClassfier_nolstm
## SelfAttentionClassfier_nolstm_spational
class NetworkSAVCls(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)
        #_cfg.spat = True
        
        if _cfg.embd_dim: 
            self.emb_dim = self.dfeat // _cfg.embd_dim
            self.embedding = Temporal(self.dfeat, self.emb_dim)
        else : 
            self.embedding = nn.Identity()
            self.emb_dim = self.dfeat 
        self.santt = SelfAnttention(self.emb_dim, _cfg.att_dim,_cfg.r,_cfg.do,_cfg.spat)
        self.cls = Cls(self.emb_dim*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self,x):
        b, t, f = x.shape
        log.debug(f'SAVcls/x: {x.shape}') 
        
        #x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        att_wght = self.santt(x) ## (b, t, r)
        log.debug(f'SAVcls/att_wght: {att_wght.shape}')
        
        mxyz = [ (x * att_wght[:,:,0].unsqueeze(2)).sum(dim=1) for i in range(att_wght.shape[2])]
        mcat = torch.cat(mxyz, dim=1)  # (b, r*f)
        #m1 = (x * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        #m2 = (x * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        #m3 = (x * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        #mcat = torch.cat([m1,m2,m3],dim=1) # b*nc,128*6
        log.debug(f'SAVcls/m: {mcat.shape}')
        
        x_cls = self.cls(mcat).view(b)
        log.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "vls": x_cls, 
            "vlf": mcat, #.permute(0,2,1), ## b, r, f
            "attw": att_wght.permute(0,2,1).squeeze(dim=0) #if self.ret_att else None
            }

## VCls
class NetworkVCls(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)
        
        ## !! have a main Network Constructer
        ## who assigns fm / rgs if provded
        ## else put sIdentity
        if _cfg.embd_dim: 
            self.emb_dim = self.dfeat // _cfg.embd_dim
            self.embedding = Temporal(self.dfeat, self.emb_dim)
        else : 
            self.embedding = nn.Identity()
            self.emb_dim = self.dfeat 
            
        self.santt = SelfAnttention(self.emb_dim, _cfg.att_dim,_cfg.r,_cfg.do,_cfg.spat)
        self.cls = Cls(self.emb_dim*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self, x):
        b, t, f = x.shape
        log.debug(f'VCls/x: {x.shape}') 

        ## change embd layers to accept either b,t,f or b,f,t
        ## by inspect if last dim match in_dim
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        ## creates r att weight maps trough temporal axis of x feats
        att_wght = self.santt(x) ## (b, t, r)
        log.debug(f'VCls/att_wght: {att_wght.shape}')
        
        ## r new temporal att weighted aggregated f features
        ## 3 VL representations
        m = torch.bmm( x.permute(0,2,1), att_wght) ## (b, f, t)*(b, t, r)=(b, f, r)
        log.debug(f'VCls/m: {m.shape}')
        
        ## (b, nfeats*r)->(b)
        x_cls = self.cls( m.view(b,-1) ).view(b)
        log.debug(f'VCls/x_cls: {x_cls.shape}')

        return {
            "vls": x_cls, ## b*nc
            "vlf": m.permute(0,2,1), ## b*nc, r, f
            "attw": att_wght.permute(0,2,1).squeeze(dim=0) ## b, r, t
            #"attw": att_wght.squeeze(dim=0) if self.ret_att else None
            }

class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata):
        ## nc, tchunck -> tchunck (norris)
        vls = ndata['vls'].mean(0)
        
        log.debug(f" {vls.shape=}")
        return vls   


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
