import torch
import torch.nn as nn
import torch.nn.functional as F

log = None
def init(l):
    global log
    log = l
    
    
#####################
## every classifier :
# excepts (b,t,f)
# outputs at SL, (b,t)
# without sigmoid 
    
class MLP(nn.Module):
    def __init__(self, feat_len, neurons=[512,512//4], activa='relu', dropout=0.7):
        super(MLP, self).__init__()
        ## 16: 512/32 1024/64 2048/128
        ## 8: 512/64 1024/128 2048/256
        ## 4: 512/128 1024/256 2048/512
        self.feat_len = feat_len
        self.cls = nn.Sequential(
            nn.Linear(self.feat_len, neurons[0]),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(neurons[0], neurons[1]),
            #nn.ReLU(), ## bert-rtfm mentioned worse perform w/ relu
            nn.Dropout(dropout), 
            nn.Linear(neurons[1], 1),
            #nn.Sigmoid(),
            )
    def forward(self, x):
        b, t, f = x.shape
        return self.cls(x).view(b,t)


class ConvCLS(nn.Module):
    def __init__(self, feat_len, ks):
        super(ConvCLS, self).__init__()
        self.cls = nn.Conv1d(feat_len, 
                            out_channels=1, 
                            kernel_size=ks)
        self.ks = ks
        
    def forward(self, x):
        b, t, f = x.shape
        x = x.permute(0, 2, 1) ## (b, f, t)
        
        ## add ks-1 yeros at start of last axis, so conv output sl scores
        x = F.pad(x, (self.ks-1, 0), "constant", 0) ## causal and no lookahead bias
        log.debug(f'ConvCLS/xpad {x.shape=}')

        x = self.cls(x).squeeze(dim=1) ## (b, 1, t) > (b, t)
        log.debug(f'ConvCLS/x {x.shape}')
        return x

        
class LSTMSCls(nn.Module):
    def __init__(self, lstm_dim=256 , lstm_bd=True):
        super(LSTMSCls,self).__init__()
        self.lstm_dim=lstm_dim

        self.lstm = rnn.LSTM(lstm_dim, num_layers=2, layout='NTC', bidirectional=lstm_bd)
        
        self.ffn = nn.HybridSequential()
        #self.ffn.add( nn.Linear(out_features=lstm_dim//2, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        self.ffn.add ( nn.Dropout(0.5) )
        self.ffn.add( nn.Linear(out_features=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        #self.ffn.add( nn.Linear(out_features=1, activation='sigmoid', weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        
    def forward(self,x):
        b, t, f = x.shape
        out = self.lstm(x)
        log.debug(f'LSTMSCls/lstm: {out.shape} ') ##  if lstm_bd (b, t, lstm_dim*2) else (b, t, lstm_dim)
        
        out = self.ffn( out.reshape(b*t,-1) ).reshape(b,t) ## (b*t,-1)>(b*t)>(b,t)
        log.debug(f'LSTMSCls/ffn: {out.shape} ')
        return out




## 4 attnomil and outputs at vl
## bert-rtfm used as substitue of bert 
class LSTMVCls(nn.Module):
    def __init__(self, lstm_dim=256, lstm_bd=True):
        super(LSTMVCls,self).__init__()
        self.lstm = rnn.LSTM(lstm_dim, num_layers=2, layout='NTC', bidirectional=lstm_bd)
        
        ## attnomil
        #self.ffn = nn.HybridSequential()
        #self.ffn.add( nn.Linear(out_features=32, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        #self.ffn.add ( nn.Dropout(0.5) )
        #self.ffn.add( nn.Linear(out_features=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        #self.ffn.add( nn.Linear(out_features=1, activation='sigmoid', weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
        
        self.ffn2 = MLP()
        
    def forward(self,x):
        b, t, f = x.shape
        #x = x.reshape(-1,b,t) #[batch,32,2048]
        
        #out, hidden = self.bilstm(x)
        out = self.lstm(x) ##  if lstm_bd (b, t, lstm_dim*2) else (b, t, lstm_dim)
        log.debug(f'LSTMVCls/lstm: {out.shape} ') ##{hidden.shape}
        
        x = out[:,-1,:] ## (b, lstm_dim*2)
        log.debug(f'LSTMVCls/out: {x.shape}')
        
        ## FFN2 
        #x = x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        #log.debug(f'LSTMVCls/xnorm: {x.shape}')
        x = self.ffn2( np.expand_dims(x, axis=1) ).reshape(b)
        
        ## FFN
        #x = self.ffn(x).reshape(b) ## (b, 1)>(b)
        
        log.debug(f'LSTMVCls/ffn: {x.shape}')
        return x