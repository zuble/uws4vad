import torch
import torch.nn.functional as F, torch.nn as nn
from torch.nn import Module, Linear, Dropout, LayerNorm, ReLU, Softmax, Softmax2d, Sigmoid, Sequential, Tanh

from .layers import *

logger = None
def init(l):
    global logger
    logger = l


#######################################
## https://arxiv.org/pdf/2209.06435.pdf
## https://github.com/sakurada-cnq/salient_feature_anomaly/blob/main/network/video_classifier.py

## Reusable Layers
class Cls(Module):
    def __init__(self, in_dim, neurons, **kwargs):
        super().__init__(**kwargs)
        self.net = Sequential(
            Linear(in_features=in_dim, out_features=neurons[0]),
            Linear(in_features=neurons[0], out_features=neurons[1]),
            Sigmoid()
        )
    def forward(self, x):
        #x = x.reshape(-1, x.shape[-1])
        return self.net(x)

class SelfAnttention(Module):
    def __init__(self, nfeats, r, da, dout_rate, sftm_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.net = Sequential(
            Linear(in_features=nfeats, out_features=da),
            Tanh(),
            Linear(in_features=da, out_features=r),
            
            Softmax2d() if sftm_dim == 2 else Softmax(dim=1),
            Dropout(dout_rate)
        )
    def forward(self, x):
        return self.net(x)

'''


class LSTMCls(HybridBlock):
    def __init__(self, lstm_dim=256):
        super(LSTMCls,self).__init__()
        #self.bilstm = nn.LSTM(args.feature_size,256,batch_first=True,bidirectional=True,num_layers=2)
        self.bilstm = rnn.LSTM(lstm_dim, num_layers=2, layout='NTC', bidirectional=True)
        
        self.ffn = nn.HybridSequential()
        self.ffn.add ( nn.Dropout(0.5) )
        self.ffn.add( nn.Dense(units=1, activation='sigmoid', weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        
    def forward(self,x):
        b, t, f = x.shape
        #x = x.view(-1,t,f) #[batch,32,2048]
        
        out,(hidden,_) = self.bilstm(x)
        logger.debug(f'LSTMCls/bilstm: {out.shape} {hidden}')
        
        x = out[:,-1,:] #[batch,256*2]
        logger.debug(f'LSTMCls/out: {x.shape}')
        
        x = self.ffn(x) #[batch,1]
        logger.debug(f'LSTMCls/ffn: {x.shape}')
        
        x = x.reshape(b) 
        logger.debug(f'LSTMCls/x: {x.shape}')
        return x


class SAVCls_lstm(HybridBlock):
    def __init__(self, lstm_dim=256, r=3, da=64, cls_neurons=[32,1], ret_att=False):
        super(SAVCls_lstm,self).__init__()
        self.ret_att = ret_att
        
        #torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, 
        #    bidirectional=False, proj_size=0, device=None, dtype=None)
        #self.bilstm = nn.LSTM(args.feature_size, lstm_dim, batch_first=True, bidirectional=True, num_layers=2)
        
        #class LSTM(hidden_size, num_layers=1, layout='TNC', dropout=0, bidirectional=False , input_size=0, 
        #    i2h_weight_initializer=None, h2h_weight_initializer=None, 
        #    i2h_bias_initializer='zeros', h2h_bias_initializer='zeros', 
        #    projection_size=None, h2r_weight_initializer=None, 
        #    state_clip_min=None, state_clip_max=None, state_clip_nan=False, dtype='float32', **kwargs)
        self.bilstm = rnn.LSTM(lstm_dim, num_layers=2, layout='NTC', bidirectional=True)
        
        self.self_anttention = self_anttention(r, da)
        
        #self.ffn = nn.Linear(lstm_dim*6,32)->nn.Linear(32,1)->Sigmoid
        self.cls = Cls(cls_neurons)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        b, t, f = x.shape
        logger.debug(f'SAVCls_lstm/x: {x.shape}')
        
        out,_ = self.bilstm(x) ## (b, t, lstm_dim)
        logger.debug(f'SAVCls_lstm/bilstm: {out.shape}')
        
        #att_wght = self.dropout(F.softmax(self.self_anttention(out),dim=1))
        att_wght = self.dropout( npx.softmax( self.self_anttention(x), axis=1) ) ## (b, t, 3)
        logger.debug(f'SAVCls_lstm/att_wght: {att_wght.shape}')
        
        #m1 = (out*att_wght[:,:,0].unsqueeze(2)).sum(dim=1)
        #m2 = (out*att_wght[:,:,1].unsqueeze(2)).sum(dim=1)
        #m3 = (out*att_wght[:,:,2].unsqueeze(2)).sum(dim=1)
        m1 = np.sum( out * np.expand_dims(att_wght[:,:,0], axis=2), axis=1) ## (b, f)
        m2 = np.sum( out * np.expand_dims(att_wght[:,:,1], axis=2), axis=1)
        m3 = np.sum( out * np.expand_dims(att_wght[:,:,2], axis=2), axis=1)
        logger.debug(f'SAVCls_lstm/m1 m2 m3: {m1.shape} {m2.shape} {m3.shape}')
        
        #x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        mcat = np.concatenate( (m1,m2,m3), axis=1) ## (b, 3*f)
        logger.debug(f'SAVCls_lstm/mcat: {mcat.shape}')
        
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.sig(x)
        #x = x.view(bs) #[batch]
        x_cls = self.cls(mcat).reshape(b)
        logger.debug(f'SAVCls_lstm/x_cls: {x_cls.shape}')
        
        if self.ret_att: return {"scores": x_cls, "attw": att_wght.squeeze(axis=0)}
        else: return {"scores": x_cls}


class SAVCls_spat(HybridBlock):
    def __init__(self, r=3, da=64, cls_neurons=[32,1], ret_att=False, **kwargs):
        super(SAVCls_spat, self).__init__(**kwargs)
        self.ret_att = ret_att
        
        #self.self_anttention = nn.Sequential(nn.Linear(args.feature_size,64),nn.Tanh(),nn.Linear(64,3))
        self.self_anttention = self_anttention(r, da)
        
        #self.ffn = nn.Linear(args.feature_size*3,32)->nn.Linear(32,1)->Sigmoid
        self.cls = Cls(cls_neurons)
        
        self.dropout = nn.Dropout(0.3)
        #self.softmax = nn.Softmax2d() ## npx.softmax def axis = -1
        
    def forward(self,x):
        b, t, f = x.shape
        logger.debug(f'SAVCls_spat/x: {x.shape}')
        
        ## for each sequence/segment/snippet feature -> 3 prob att value summing to 1
        att_wght = self.dropout( npx.softmax( self.self_anttention(x) ) ) ## (b, t, 3)
        logger.debug(f'SAVCls_spat/att_wght: {att_wght.shape}') #{att_wght} 
        
        m1 = np.sum( x * np.expand_dims(att_wght[:,:,0], axis=2), axis=1) ## (b, f)
        m2 = np.sum( x * np.expand_dims(att_wght[:,:,1], axis=2), axis=1)
        m3 = np.sum( x * np.expand_dims(att_wght[:,:,2], axis=2), axis=1)
        logger.debug(f'SAVCls_spat/m1 m2 m3: {m1.shape} {m2.shape} {m3.shape}')
        
        #x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        mcat = np.concatenate( (m1,m2,m3), axis=1) ## (b, 3*f)
        logger.debug(f'SAVCls_spat/mcat: {mcat.shape}')
        
        x_cls = self.cls(mcat).reshape(b)
        logger.debug(f'SAVCls_spat/x_cls: {x_cls.shape}')
        
        if self.ret_att: return {"scores": x_cls, "attw": att_wght.squeeze(axis=0)}
        else: return {"scores": x_cls} 


class SAVCls(HybridBlock):
    def __init__(self, r=3, da=64, cls_neurons=[32,1], ret_att=False, **kwargs):
        super(SAVCls, self).__init__(**kwargs)
        self.ret_att = ret_att
        
        #self.self_anttention = nn.Sequential(nn.Linear(args.feature_size,64),nn.Tanh(),nn.Linear(64,3))
        self.self_anttention = self_anttention(r, da)
        
        #self.ffn = nn.Linear(args.feature_size*3,32)->nn.Linear(32,1)->Sigmoid
        self.cls = Cls(cls_neurons)
        
        self.dropout = nn.Dropout(0.3)        

    def forward(self, x):
        b, t, f = x.shape
        logger.debug(f'SAVCls/x: {x.shape}')
        
        att_wght = self.dropout( npx.softmax( self.self_anttention(x), axis=1) ) ## (b, t, 3)
        logger.debug(f'SAVCls/att_wght: {att_wght.shape}')
        
        m1 = np.sum( x * np.expand_dims(att_wght[:,:,0], axis=2), axis=1) ## (b, f)
        m2 = np.sum( x * np.expand_dims(att_wght[:,:,1], axis=2), axis=1)
        m3 = np.sum( x * np.expand_dims(att_wght[:,:,2], axis=2), axis=1)
        logger.debug(f'SAVCls/m1 m2 m3: {m1.shape} {m2.shape} {m3.shape}')
        
        #x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        mcat = np.concatenate( (m1,m2,m3), axis=1) ## (b, 3*f)
        logger.debug(f'SAVCls/mcat: {mcat.shape}')
        
        x_cls = self.cls(mcat).reshape(b)
        logger.debug(f'SAVCls/x_cls: {x_cls.shape}')
        
        if self.ret_att: return {"scores": x_cls, "attw": att_wght.squeeze(axis=0)}
        else: return {"scores": x_cls}
'''

class VCls(Module):
    def __init__(self, rgbnf, cfg_net, cfg_cls, **kwargs):
        super().__init__(**kwargs)
        r = cfg_net.R
        da = cfg_net.DA
        cls_neurons=[32,1]
        dout_rate=0.3
        sftm_dim=1
        ret_att=False
        
        self.selfanttention = SelfAnttention(rgbnf, r, da, dout_rate, sftm_dim)
        self.cls = Cls(in_dim=rgbnf*r, neurons=cls_neurons)
        self.ret_att = ret_att

    def forward(self, x):
        b, t, f = x.shape
        logger.debug(f'VCls/x: {x.shape}') 

        ## creates r att weight maps trough temporal axis of x feats
        att_wght = self.selfanttention(x) ## (b, t, r)
        logger.debug(f'VCls/att_wght: {att_wght.shape}')
        
        ## each feature in (f, t) is weighted sum for each r att_wght map (t, r) across temporal dimension
        ## resulting in r new temporal att aggregated f features
        m = torch.bmm( x.permute(0,2,1), att_wght) ## (b, f, t)*(b, t, r)=(b, f, r)
        logger.debug(f'VCls/m: {m.shape}')
        
        ## (b, nfeats*r)->(b)
        x_cls = self.cls( m.view(b,-1) ).view(b)
        logger.debug(f'VCls/x_cls: {x_cls.shape}')

        return {"scores": x_cls, "attw": att_wght.squeeze(axis=0) if self.ret_att else None}


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
