import mxnet as mx
from mxnet.gluon import nn, rnn, HybridBlock
from mxnet import init as mxinit, np, npx
#from scipy.spatial.distance import pdist, squareform

logger = None
def init(l):
    global logger
    logger = l

    
###############################################
## https://ieeexplore.ieee.org/document/9712793
## https://github.com/yujiangpu20/cma_xdVioDet/tree/main
'''
def get_adjmtx(seq_len):
    arith = np.arange(seq_len).reshape(-1, 1)
    adj_mtx = np.abs(arith - arith.transpose())
    return np.power(adj_mtx, 2)
    adj_mtx2 = np.array( squareform( pdist(arith.asnumpy(), metric='cityblock').astype(np.float32) ) )
    #logger.debug(f'1: {adj_mtx} \n 2: {adj_mtx2}')
    
class DistanceAdjTorch(nn.Module):
    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.bias = nn.Parameter(torch.FloatTensor(1))

    def hybrid_forward(self, bs, max_seqlen):
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).cuda()
        self.dist = torch.exp(- torch.abs(self.w * (self.dist**2) + self.bias))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(bs, 1, 1).cuda()

        return self.dist
'''    
class DistanceAdj(HybridBlock):
    def __init__(self, device):
        super(DistanceAdj, self).__init__()
        self.device=device
        with self.name_scope():
            self.w = self.params.get('weight', shape=(1,), init=mxinit.Constant(value=0.1) )
            self.bias = self.params.get('bias', shape=(1,), init=mxinit.Constant(value=0.1) )
            #self.w = self.params.get_constant( 'weight', value=0.1 )
            #self.bias = self.params.get_constant( 'bias', value=0.1 )
        #self.w = Parameter('weight',shape=(1,), allow_deferred_init=True)
        #self.bias = Parameter('bias', shape=(1,), allow_deferred_init=True)
        #logger.debug(f'{self.w} {self.bias}')
        
    def forward(self, bs, seqlen):
        arith = np.arange(seqlen).reshape(-1, 1)
        adj_mtx = np.power( np.abs(arith - arith.transpose()) , 2 ).copyto(self.device)
        #self.dist = torch.exp(- torch.abs(self.w * (self.dist**2) + self.bias))
        #self.dist = np.exp(- np.abs(self.w.data() * (self.dist**2) + self.bias.data()))
        self.dist = np.exp(- np.abs(self.w.data() * adj_mtx + self.bias.data()))
        #self.dist = np.exp(- np.abs( np.multiply(self.w.data(), (self.dist*self.dist) ) + self.bias.data()))
        
        self.dist = np.expand_dims(self.dist, 0)
        if bs != 1: self.dist = self.dist.repeat(bs, axis=0)

        return self.dist


class CrossAttention(HybridBlock):
    ''' 
        Takes the audio and RGB features, computes query (Q), key (K), and value (V) representations of these features. 
        Then it computes an attention map by taking the dot product of the queries and keys, 
        adding the adjacency matrix, 
        and applying softmax. 
        The attention map is then used to compute a weighted sum of the values.
    '''
    def __init__(self, audnf, rgbnf, dim_k, n_heads=1):
        super(CrossAttention, self).__init__()
        self.audnf = audnf
        self.rgbnf = rgbnf
        self.dim_k = dim_k
        self.n_heads = n_heads
        #self.q = nn.Linear(audnf, dim_k)
        #self.k = nn.Linear(rgbnf, dim_k)
        #self.v = nn.Linear(rgbnf, dim_k)
        #self.o = nn.Linear(dim_k, rgbnf)

        with self.name_scope():
            ## dim_k = audnf ? because its a linear projection ? 
            
            ## in_units=audnf
            self.q = nn.Dense(units= dim_k, flatten=False, weight_initializer=mxinit.Xavier(),bias_initializer='zeros')
            ## in_units=rgbnf
            self.k = nn.Dense(units= dim_k, flatten=False, weight_initializer=mxinit.Xavier(),bias_initializer='zeros')
            ## in_units=rgbnf
            self.v = nn.Dense(units= dim_k, flatten=False, weight_initializer=mxinit.Xavier(),bias_initializer='zeros')
            ## in_units=dim_k
            self.o = nn.Dense(units= rgbnf, flatten=False, weight_initializer=mxinit.Xavier(),bias_initializer='zeros')
            
            #self.norm_fact = 1 / math.sqrt(dim_k)
            self.norm_fact = self.params.get_constant( 'norm_fact', value=(1 / np.sqrt(np.array([dim_k]))) )
            
            #self.act = nn.Softmax(dim=-1)
    
    def forward(self, xa, xv, adj):
        ## The audio features are transformed into a set of query vectors Q using a linear transformation (self.q). 
        ##  Each query vector represents a high-dimensional "question" about the content of the video features at a certain point in the sequence.
        Q = self.q(xa).reshape(-1, xa.shape[0], xa.shape[1], self.dim_k // self.n_heads) ## Xaud Wq (1, b, seq.len, dim_k//1)
        ## The video features y are transformed into a set of key vectors K and value vectors V using separate linear transformations (self.k and self.v).
        ##  Each key-value pair represents a piece of content in the video features: 
        ##  the key is used to match with a query, 
        ##  and the value is used to generate the output if the key is matched.
        K = self.k(xv).reshape(-1, xv.shape[0], xv.shape[1], self.dim_k // self.n_heads) ## Xvis Wk (1, b, seq.len, dim_k//1)
        V = self.v(xv).reshape(-1, xv.shape[0], xv.shape[1], self.dim_k // self.n_heads) ## Xvis Wv (1, b, seq.len, dim_k//1)
        logger.debug(f'CrossAttention/Q {Q.shape} , K {K.shape} , V {V.shape}')
        
        
        ## The scaled dot products between the queries and keys are computed 
        ##  to obtain a measure of match or similarity between the audio features and video features at each point in the sequence. 
        ##  The scaling factor self.norm_fact helps keep the dot products from growing too large, which can lead to numerical instability.
        attention_map = np.matmul(Q, K.transpose((0, 1, 3, 2))) * self.norm_fact.data() ## (1, b, seq.len, seq.len)
        logger.debug(f'CrossAttention/att_map {attention_map.shape}')
        ## The adjacency matrix adj is added to the scaled dot products to incorporate the relationships between different points in the sequence. 
        ##  The softmax function is then applied to convert these scores into attention weights,
        ##  which sum to 1 and can be interpreted as the model's "attention" or focus on the video features based on the audio features.
        attention_map = npx.softmax(attention_map + adj) ## eq(1) (1, b, seq.len, seq.len)
        logger.debug(f'CrossAttention/att_map {attention_map.shape}')
        
        ## The attention weights are used to compute a weighted sum of the value vectors, 
        ##  which represents the output of the attention mechanism for each point in the sequence. 
        ##  This output is a combination of the video features, weighted based on their match with the audio features and their relationships as determined by the adjacency matrix.
        temp = np.matmul(attention_map, V).reshape(xv.shape[0], xv.shape[1], -1) ## (b, seq.len, dimk//1)
        logger.debug(f'CrossAttention/temp {temp.shape}')
        
        ## rgbnf shape match
        output = self.o(temp).reshape(-1, xv.shape[1], xv.shape[2]) ## (b, seq.len, rgbnf)
        logger.debug(f'CrossAttention/output {output.shape}')
        return output


class CMA_LA(HybridBlock):
    '''
        applies a cross attention mechanism to the RGB and audio features, 
        using the adjacency matrix to guide the attention. 
        its then added to the original input (residual connection), normalized, 
        and then passed through a MLP network
    '''
    def __init__(self, rgbnf, audnf, hid_dim=128, d_ff=512, dropout_rate=0.1):
        super(CMA_LA, self).__init__()

        if hid_dim != audnf: 
            logger.info(f"Setting hod_dim of CrossAttention block: 128 -> {audnf}")
            #hid_dim = audnf
            
        with self.name_scope():
            self.cross_attention = CrossAttention(audnf, rgbnf, hid_dim)

            ## defaults: (axis=-1, epsilon=1e-05, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', in_channels=0)
            self.norm = nn.LayerNorm() ## in_channels=rgbnf (inferred)
            #self.norm = npx.layer_norm()
            
            #self.ffn = nn.HybridSequential()
            #self.ffn.add( nn.Conv1D( d_ff, kernel_size=1, weight_initializer=mxinit.Xavier(),bias_initializer='zeros') )
            #self.ffn.add( nn.LeakyReLU(0.2))
            #self.ffn.add( nn.Dropout(dropout_rate) )
            ## this conv layer should match the audnf ??
            ## take into account that for rgbnf = 2048 , the arousal will be greater
            ## so maybe increse the nfilters relation from this 2 convs
            #self.ffn.add( nn.Conv1D(128, kernel_size=1, weight_initializer=mxinit.Xavier(),bias_initializer='zeros') )
            #self.ffn.add( nn.Dropout(dropout_rate) )
            ####
            ## 2 act as cls
            #self.ffn.add( nn.Dense(units=1, activation='sigmoid', weight_initializer=mxinit.Xavier(), bias_initializer='zeros') )        

    def forward(self, rgbf, audf, dist):
        new_x = rgbf + self.cross_attention(audf, rgbf, dist) ## (b, seq.len, rgbnf)
        new_x = self.norm(new_x)
        #new_x = new_x.transpose((0, 2, 1)) ## (b, rgbnf, seq.len)
        #new_x = self.ffn(new_x) ## (b, 128, seq.len)
        return new_x


class CMA(HybridBlock):
    def __init__(self, rgbnf, audnf, devices, **kwargs):
        super(CMA, self).__init__(**kwargs)
        
        #self.devices = kwargs.get('devices', None)
        #self.device = devices[0]
        self.rgbnf = rgbnf
        self.audnf = audnf

        self.distanceadj = DistanceAdj(devices[0])
        
        self.cma_la = CMA_LA( self.rgbnf, self.audnf)
        
        #self.classifier = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=7, stride=1, padding=0)
        #self.classifier = nn.Conv1D(channels=1, kernel_size=7, padding=0, ##in_channels = 128
        #                        weight_initializer=mxinit.Xavier(),bias_initializer='zeros')
        self.classifier2 = MLP()
        
    def forward(self, x): #adj_mtx   
        rgbf = x[:, :, :self.rgbnf]
        audf = x[:, :, self.rgbnf:]
        #logger.debug(f'CMA/rgbf:{rgbf.shape}, CMA/audf:{audf.shape} {type(audf)} ')
        
        bs, seqlen = rgbf.shape[0:2]
        #logger.debug(f'CMA/bs:{bs}, CMA/seqlen:{seqlen} ')
        
        ## constant matrix controlled by seqlen
        dist = self.distanceadj( bs, seqlen ) ## (bs, seq.len, seq.len)
        logger.debug(f'CMA/dost: {dist.shape}')

        ## rgbf enhanced by audf -> ffn to macth audnf
        newf = self.cma_la(rgbf, audf, dist) ## (bs, audnf, seq.len)
        logger.debug(f'CMA/cma_la: {newf.shape}')

        ## add 6 yeros at start of last axis, so conv output seqlen scores
        #newf = F.pad(newf, mode='constant', constant_value=0, pad_width=(0, 0, 6, 0))
        #newf = np.pad(newf, ((0, 0), (0, 0), (6, 0)), mode='constant', constant_values=0) ## (bs, audnf, seq.len+6)
        #logger.debug(f'CMA/pad: {newf.shape}')
        #scores = self.classifier(newf).squeeze(axis=1) ## (bs, seq.len)
        #scores = npx.sigmoid(scores) ## (bs, seq.len)
        #logger.debug(f'CMA/sig: {scores.shape}')
        
        
        ## for each sequence/segment/snippet feature -> 1 cls value
        scores = self.classifier2( newf ).reshape((bs,seqlen)) ## (b, seq.len)
        logger.debug(f'CMA/scores {scores.shape}')
                
        return {"scores": scores}


