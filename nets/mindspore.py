import mxnet as mx
from mxnet.gluon import nn, rnn, HybridBlock
from mxnet import init as mxinit, np, npx

logger = None
def init(l):
    global logger
    logger = l
    
    
#########################################
## https://arxiv.org/pdf/2309.16309v1.pdf
## https://github.com/2023-MindSpore-4/Code4/blob/main/WS-VAD-mindspore-main/non_local.py

class _NonLocalBlockND(HybridBlock):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0: self.inter_channels = 1

        with self.name_scope():
            self.g = nn.HybridSequential()
            self.g.add( nn.Conv1D(self.inter_channels, kernel_size=1) )  #strides=1, padding=0, in_channels=self.in_channels
            self.theta = nn.Conv1D(self.inter_channels, kernel_size=1) #strides=1, padding=0, in_channels=self.in_channels
            self.phi = nn.HybridSequential()
            self.phi.add( nn.Conv1D(self.inter_channels, kernel_size=1) ) #strides=1, padding=0, in_channels=self.in_channels

            if bn_layer:
                self.W = nn.HybridSequential()
                self.W.add( nn.Conv1D(self.in_channels, kernel_size=1) ) #in_channels=self.inter_channels
                self.W.add( nn.BatchNorm() ) #in_channels=self.in_channels
            else:
                self.W = nn.Conv1D(self.in_channels, kernel_size=1 ) #in_channels=self.inter_channels

            if sub_sample:
                ## class MaxPool1D(pool_size=2, strides=None, padding=0, layout='NCW', ceil_mode=False, **kwargs)
                self.max_pool = nn.MaxPool1D()
                self.g.add(self.max_pool)
                self.phi.add(self.max_pool)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """  
        bs = x.shape[0]

        g_x = self.g(x).reshape(bs, self.inter_channels, -1)
        g_x = g_x.transpose((0, 2, 1))

        theta_x = self.theta(x).reshape(bs, self.inter_channels, -1)
        theta_x = theta_x.transpose((0, 2, 1))
        phi_x = self.phi(x).reshape(bs, self.inter_channels, -1)

        f = npx.batch_dot(theta_x, phi_x)
        N = f.shape[-1]
        f_div_C = f / N

        y = npx.batch_dot(f_div_C, g_x)
        y = y.transpose((0, 2, 1))#.contiguous()
        y = y.reshape(bs, self.inter_channels, *x.shape[2:])
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

class Aggregate(HybridBlock):
    def __init__(self, rgbnf):
        super(Aggregate, self).__init__()
        
        self.rgbnf = rgbnf
        self.out_len = rgbnf // 4

        with self.name_scope():
            ## As for the local branch, in order to acquire the different time scale local reliance, 
            ## 1-D convolution operation with dilation (1, 2, 4) is separately used: Fl1 = Fl2 = Fl3 = φ(F)
            ##      where φ denotes the dilated convolution and Fl[1:3]∈ R^(T ∗ (D/4))
            
            ##Conv1D(channels, kernel_size, strides=1, padding=0, dilation=1, groups=1, layout='NCW', activation=None, use_bias=True, weight_initializer=None, bias_initializer='zeros', in_channels=0, **kwargs)
            self.conv_fl1 = nn.HybridSequential()
            self.conv_fl1.add( nn.Conv1D(channels=self.out_len, kernel_size=3, padding=1,  dilation=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.conv_fl1.add( nn.Activation('relu') )
            self.conv_fl1.add( nn.BatchNorm() )  ## in_channels=self.out_len
            
            self.conv_fl2 = nn.HybridSequential()
            self.conv_fl2.add( nn.Conv1D(channels=self.out_len, kernel_size=3, padding=2, dilation=2, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.conv_fl2.add( nn.Activation('relu') )
            self.conv_fl2.add( nn.BatchNorm() )  ## in_channels=self.out_len
            
            self.conv_fl3 = nn.HybridSequential()
            self.conv_fl3.add( nn.Conv1D(channels=self.out_len, kernel_size=3, padding=4, dilation=4, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.conv_fl3.add( nn.Activation('relu') )
            self.conv_fl3.add( nn.BatchNorm() )  ## in_channels=self.out_len
            

            self.conv_4 = nn.HybridSequential()
            self.conv_4.add( nn.Conv1D(channels=self.out_len, kernel_size=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.conv_4.add( nn.Activation('relu') )
            
            self.conv_5 = nn.HybridSequential()
            self.conv_5.add( nn.Conv1D(channels=self.rgbnf, kernel_size=3, padding=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.conv_5.add( nn.Activation('relu') )
            self.conv_5.add( nn.BatchNorm() ) ## in_channels=self.rgbnf

            ## Given the feature F ∈ R^(T∗D) , in the global branch, we
            ## simply introduce the non-local block proposed in [19]:
            ## Fg = ψ(F ),
            ##     where ψ denotes the 1-D non-local operation and Fg ∈ R^(T∗D/4)
            self.non_local = NONLocalBlock1D(self.out_len, sub_sample=False, bn_layer=True)

    def forward(self, x):
        residual = out = x ## (b, rgnf, seq.len)
        logger.debug(f'Aggregate/x {out.shape}') 
        
        ## local
        out1 = self.conv_fl1(out) ## (b, rgbnf//4, seq.len)
        out2 = self.conv_fl2(out) ## (b, rgbnf//4, seq.len)
        out3 = self.conv_fl3(out) ## (b, rgbnf//4, seq.len)
        logger.debug(f'Aggregate/conv_fl123 {out1.shape, out2.shape, out3.shape}')
        out_cat = np.concatenate((out1, out2, out3), axis=1) ## (b, 3*(rgbnf//4), seq.len)
        logger.debug(f'Aggregate/conv_fl123-cat {out_cat.shape}')
        
        ## global
        out = self.conv_4(out) ## (b, rgbnf//4, seq.len)
        logger.debug(f'Aggregate/conv_4 {out.shape}')
        out = self.non_local(out) ## (b, rgbnf//4, seq.len)
        logger.debug(f'Aggregate/non_local {out.shape}')
        
        ## aggregate features
        out = np.concatenate((out_cat, out), axis=1) ## (b, rgbnf, seq.len)
        logger.debug(f'Aggregate/cat {out.shape}')
        
        out = self.conv_5(out) ## (b, rgbnf, seq.len)
        logger.debug(f'Aggregate/conv_5 {out.shape}')
        
        ## enhanced feature
        xe = out + residual ## (b, rgbnf, seq.len)
        logger.debug(f'Aggregate/enhanced {xe.shape}')
        return xe


class Attention(HybridBlock):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        with self.name_scope():
            ## 512/512/1
            self.att = nn.HybridSequential()
            self.att.add( nn.Conv1D(512, kernel_size=3, padding=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )  ## in_channels=n_feature
            self.att.add( nn.LeakyReLU(0.2) )
            self.att.add( nn.Dropout(0.7) )
            self.att.add( nn.Conv1D(128, kernel_size=3, padding=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') ) ## in_channels=512
            self.att.add( nn.LeakyReLU(0.2) )
            self.att.add( nn.Conv1D(1, kernel_size=1, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') ) ## in_channels=512
            self.att.add( nn.Dropout(0.7) )
            self.att.add( nn.Activation('sigmoid') )

    def forward(self, feat):
        ## snippet-level anomalous attention
        #logger.info(f'Attention/attention0_conv0_bias: {self.att[0].bias.data()}')
        x = self.att(feat)
        return x

class MLP(HybridBlock):
    def __init__(self, neurons=[512,512//4], activa='relu', dropout=0.7, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            ## 16: 512/32 1024/64 2048/128
            ## 8: 512/64 1024/128 2048/256
            ## 4: 512/128 1024/256 2048/512
            self.mlp = nn.HybridSequential()
            self.mlp.add( nn.Dense(units=neurons[0], activation=activa, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.mlp.add( nn.Dropout(dropout) )
            self.mlp.add( nn.Dense(units= neurons[1], activation=activa, weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )
            self.mlp.add( nn.Dropout(dropout) )
            self.mlp.add( nn.Dense(units=1, activation='sigmoid', weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )        

    def forward(self, x):
        b, t, f = x.shape
        x = np.reshape(x, (-1, f)) 
        #logger.debug(f'MLP {x.shape}')
        x = self.mlp(x)
        #logger.debug(f'MLP {x.shape}')
        return x

class MindSpore(HybridBlock):
    def __init__(self, rbgnf, devices, **kwargs):
        super(MindSpore, self).__init__(**kwargs)

        self.rgbnf = rbgnf
        self.devices = devices
        
        #with self.name_scope():
        self.aggregate = Aggregate(self.rgbnf)
        self.classifier = MLP()
        self.attn = Attention()

    def forward(self, x):
        b, t, f = x.shape
        logger.debug(f'MindSpore/x {x.shape}')
        
        rgbf = x[:, :, :self.rgbnf]
        #audf = x[:, :, self.rgbnf:]
        
        ## rgbf temporal refinement/ennahncment
        xv_new = self.aggregate( rgbf.transpose((0, 2, 1)) ) ## (b, rgbnf, seq.len)
        logger.debug(f'MindSpore/xv_new {xv_new.shape}')
        
        ########
        #xva_new = np.concatenate((xv_new, audf.transpose((0, 2, 1)) ), axis=1)
        ########
        
        ## for each sequence/segment/snippet feature -> 1 att value
        xv_att = self.attn(xv_new).squeeze(axis=1) ## (b, seq.len)
        logger.debug(f'MindSpore/xv_att {xv_att.shape}')
        
        ## for each sequence/segment/snippet feature -> 1 cls value
        xv_cls = self.classifier( xv_new.transpose((0, 2, 1)) ).reshape((b,t)) ## (b, seq.len)
        logger.debug(f'MindSpore/xv_cls {xv_cls.shape}')
        
        #return {'scores': xv_cls }
        return  {'scores': xv_cls, 'scores_att': xv_att} #.transpose((-1, -2))
        
        #if 'test': x_cls = xv_att.transpose((0, 2, 1)) * x_cls
        #    return np.mean(x_cls.reshape(bs, n_crops, -1), axis=1).expand_dims(2)