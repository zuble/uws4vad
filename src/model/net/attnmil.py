import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.net.layers import BasePstFwd
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance, BatchedDistance, CosineSimilarity

from hydra.utils import instantiate as instantiate
from src.utils.logger import get_log
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
    def __init__(self, dfeat, _cfg):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dfeat, _cfg.att_dim),
            nn.Tanh(),
            nn.Linear(_cfg.att_dim, _cfg.r),
            
            nn.Softmax2d() if _cfg.spat else nn.Softmax(dim=1),
            nn.Dropout(_cfg.do)
        )
    def forward(self, x):
        return self.net(x)


## SelfAttentionClassfier
class SAVCls_lstm(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)
        
        self.bilstm = nn.LSTM(self.dfeat, _cfg.lstm_dim,
                            batch_first=True, 
                            bidirectional=_cfg.lstm_bd, 
                            num_layers=2)
        lstm_out = _cfg.lstm_dim if not _cfg.lstm_bd else _cfg.lstm_dim*2
        
        self.santt = SelfAnttention( lstm_out, _cfg)
        self.cls = Cls(lstm_out*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self,x):
        b, t, f = x.shape
        log.debug(f'SAVcls/x: {x.shape}') 
        
        out,_ = self.bilstm(x)
        log.debug(f'SAVcls/bilstm: {out.shape}')
        
        att_wght = self.santt(out) ## (b, t, r)
        log.debug(f'SAVcls/att_wght: {att_wght.shape}')
                
        m1 = (out * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        mxyz = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        log.debug(f'SAVcls/mxyz: {mxyz.shape}')
        
        x_cls = self.cls(mxyz).view(b)
        log.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "vls": x_cls, 
            "attw": att_wght.squeeze(axis=0) if self.ret_att else None
            }
        
## SelfAttentionClassfier_nolstm
## SelfAttentionClassfier_nolstm_spational
class SAVCls(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)
        #_cfg.spat = True
        
        self.santt = SelfAnttention(self.dfeat, _cfg)
        self.cls = Cls(self.dfeat*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self,x):
        b, t, f = x.shape
        log.debug(f'SAVcls/x: {x.shape}') 
        
        att_wght = self.santt(x) ## (b, t, r)
        log.debug(f'SAVcls/att_wght: {att_wght.shape}')
                
        m1 = (x * att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (x * att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (x * att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        mxyz = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        log.debug(f'SAVcls/m: {mxyz.shape}')
        
        x_cls = self.cls(mxyz).view(b)
        log.debug(f'SAVcls/x_cls: {x_cls.shape}')
        return {
            "vls": x_cls, 
            "attw": att_wght.squeeze(axis=0) if self.ret_att else None
            }

## VCls
class VCls(nn.Module):
    def __init__(self, dfeat, _cfg):
        super().__init__()
        self.dfeat = sum(dfeat)

        self.santt = SelfAnttention(self.dfeat, _cfg)
        self.cls = Cls(self.dfeat*_cfg.r, _cfg.cls_dim)
        
        self.ret_att = _cfg.ret_att
        
    def forward(self, x):
        b, t, f = x.shape
        log.debug(f'VCls/x: {x.shape}') 

        ## creates r att weight maps trough temporal axis of x feats
        att_wght = self.santt(x) ## (b, t, r)
        log.debug(f'VCls/att_wght: {att_wght.shape}')
        
        ## each feature in (f, t) is weighted sum for each r att_wght map (t, r) across temporal dimension
        ## resulting in r new temporal att weighted aggregated f features
        ## 3 VL representations
        m = torch.bmm( x.permute(0,2,1), att_wght) ## (b, f, t)*(b, t, r)=(b, f, r)
        log.debug(f'VCls/m: {m.shape}')
        
        ## (b, nfeats*r)->(b)
        x_cls = self.cls( m.view(b,-1) ).view(b)
        log.debug(f'VCls/x_cls: {x_cls.shape}')

        return {
            "vls": x_cls, 
            "vlf": m.permute(0,2,1), ## b, r, f
            "attw": att_wght.permute(0,2,1).squeeze(dim=0)
            #"attw": att_wght.squeeze(dim=0) if self.ret_att else None
            }


class NetPstFwd(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)
        #self.triplet = losses.TripletMarginLoss(margin=0.5) # LpDistance(normalize_embeddings=True, p=2, power=1)
        ### squared L2 distance ###
        #self.triplet = losses.TripletMarginLoss(margin=0.2, distance=LpDistance(power=2))
        ### unnormalized L1 distance ###
        #self.triplet = losses.TripletMarginLoss(margin=0.2, distance=LpDistance(normalize_embeddings=False, p=1))
        self.triplet = losses.TripletMarginLoss(margin=0.2, distance=CosineSimilarity())
        
    def train(self, ndata, ldata, lossfx):
        #######
        ## VL SCORE
        super().uncrop(ndata, 'vls', 'mean')
        L0 = lossfx['bce'](ndata['vls'], ldata['label'])
        
        
        #######
        ## ATTW 
        super().uncrop(ndata, 'attw', 'mean')
        ndata["attw"] = ndata["attw"].mean(dim=1) ## b, t
        #log.debug(f"{ndata['attw'].shape}  ")
        
        abn_attw = ndata["attw"][ ldata['label'] != 0 ] ## bag, t
        nor_attw = ndata["attw"][ ldata['label'] == 0 ]
        
        ## normal
        L1 = { "guide0": torch.mean((nor_attw - 0) ** 2)}
        log.debug(f"{nor_attw.shape} {L1}")    
        
        ## abnormal
        #if self.cur_stp < self.M: lg1_tmp = SO_1
        #else: lg1_tmp = (SO_1 > 0.5).float()
        #L1 = torch.mean((abn_attw - lg1_tmp) ** 2)
        #log.debug(f'LG_1 {LG_1}')
        
        ## Loss L1-Norm , 2 polarize
        L2 = { "norm1": 0.8 * torch.sum(torch.abs(abn_attw)) } 
        #log.debug(f"{abn_attw.shape} {L2}") 
        
        
        #######
        ## VL FEAT
        super().uncrop(ndata, 'vlf', 'mean') ## b, r, f
        #vlf = ndata["vlf"].mean(dim=1) ## b, f
        vlf = ndata["vlf"].view(-1, vlf.shape[-1]) ## b*r, f 
        #log.info(f"{vlf.shape}") 
        
        #vlcosim = CosineSimilarity(sel_nor_vlf, sel_abn_vlf)
        
        L3 = {'triple': self.triplet(vlf, ldata['label'] )}
        log.info(f"{L3}") 
        
        
        nor_mask = ldata['label'] == 0
        abn_mask = ldata['label'] != 0
        log.warning(f"{nor_mask.shape}  {abn_mask.shape}")
        
        sel_nor_vlf = vlf[nor_mask]
        sel_abn_vlf = vlf[abn_mask]
        log.warning(f"{sel_nor_vlf.shape}  {sel_abn_vlf.shape}")
        
        
        '''
        def collect_fn(all_mat):
            def fn(mat, *_):
                all_mat.append(mat)
            return fn
        
        mat = []
        distance = BatchedDistance(CosineSimilarity(), collect_fn(mat), self.bs)
        
        #distance(embeddings, ref)
        distance(vlf)
        mat = torch.cat(mat, dim=0)

        log.info(f"{mat.shape} {mat}") 
        '''
        
        
        
        '''
        
        
        
        nor_euc_dist = torch.norm(vlf[nor_mask], p=2, dim=1)
        abn_euc_dist = torch.abs(100 - torch.norm(vlf[abn_mask], p=2, dim=1))
        log.debug(f"{nor_euc_dist.shape}  {abn_euc_dist.shape}")
        
        L1 = torch.mean( (nor_euc_dist + abn_euc_dist) ** 2)
        log.debug(f"{L1}")    
        
        
        L1 = None
        for bi in range( vlf.shape[0] ):
            
            if ldata['label'][bi]: 
                tmp = torch.abs(100 - torch.norm(vlf[bi], p=2, dim=1) )
                
            else:  
                tmp = torch.norm(vlf[bi], p=2, dim=1)
                
            #loss_rtfm = torch.mean((loss_abn + loss_norm) ** 2)
            log.debug(f"{tmp.shape} {tmp}")
            if L1 is None: L1 = tmp 
            else: L1 = torch.cat((L, tmp), dim=0)   
            
            
        log.debug(f"{L1}")    
        L1 = L1.mean( L1.sum() ** 2)
        log.debug(f"{L1}") 
        '''
        return super().merge(L0, L3) #, L1, L2


    def infer(self, ndata):
        #super().uncrop(ndata, '', 'mean')
        
        t, f = ndata['vls'].shape
        log.debug(f" {ndata['vls'].shape}")
        return ndata['vls']    


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
