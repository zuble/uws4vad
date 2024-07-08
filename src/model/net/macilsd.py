import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *
import copy


############
class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(SelfAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size
    def forward(self, feature):
        feature_sa = self.layer(feature, feature, feature)
        return feature_sa

class CrossAttentionBlock(nn.Module):
    def __init__(self, attention_layer):
        super(CrossAttentionBlock, self).__init__()
        self.layer = attention_layer
        self.size = attention_layer.size
    def forward(self, video, audio):
        video_cma = self.layer(video, audio, audio)
        audio_cma = self.layer(audio, video, video)
        return video_cma, audio_cma

class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, q, k, v):
        q = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])
        return self.sublayer[1](q, self.feed_forward)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def attention(query, key, value, masksize, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if masksize != 1:
        masksize = int(masksize / 2)
        mask = torch.ones(scores.size()).cuda()
        for i in range(mask.shape[2]):
            if i - masksize > 0:
                mask[:, :, i, :i - masksize] = 0
            if i + masksize + 1 < mask.shape[3]:
                mask[:, :, i, masksize + i + 1:] = 0
        # print(mask[0][0])
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, masksize=1, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.masksize = masksize
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                            zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.masksize, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return out, self.attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return output
#############



###############
class Att_MMIL(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Att_MMIL, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    ## topkmean
    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0)#.cuda()  # tensor([])
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, a_out, v_out, seq_len):
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2) ## b, t, 2, 128
        frame_prob = self.fc(x) ## b, t, 2, 1
        
        a_sls = torch.sigmoid(frame_prob[:, :, 0, :]) ## b, t, 1
        v_sls = torch.sigmoid(frame_prob[:, :, 1, :]) ## b, t, 1
        av_sls = frame_prob.sum(dim=2) ## b, t, 1
        mil_vls = self.clas(av_sls, seq_len) ## b
        return mil_vls, a_sls, v_sls, av_sls


class AVCE_Model(nn.Module):
    def __init__(self):
        super(AVCE_Model, self).__init__()
        c = copy.deepcopy
        dropout = 0.1 #args.dropout
        nhead = 4 #args.nhead
        hid_dim = 128 #args.hid_dim
        ffn_dim = 128 #args.ffn_dim
        num_classes = 1
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(1024, hid_dim)
        self.fc_a = nn.Linear(128, hid_dim)
        self.cma = CrossAttentionBlock(
            TransformerLayer(hid_dim, 
                            MultiHeadAttention(nhead, hid_dim), 
                            c(self.feedforward), 
                            dropout))
        self.att_mmil = Att_MMIL(hid_dim, num_classes)

    def forward(self, f_a, f_v, seq_len):
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a) ## b, t, 128
        v_out, a_out = self.cma(f_v, f_a) ## b, t, 128
        #mil_vls, a_sls, v_sls, av_sls = 
        return self.att_mmil(a_out, v_out, seq_len), v_out, a_out



class Single_Model(nn.Module):
    def __init__(self, ):
        super(Single_Model, self).__init__()
        c = copy.deepcopy
        dropout = 0.1 #args.dropout
        nhead = 4 #args.nhead
        hid_dim = 128 #args.hid_dim
        ffn_dim = 128 #args.ffn_dim
        num_classes = 1
        n_dim = 1024#args.v_feature_size
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        self.fc_v = nn.Linear(n_dim, hid_dim)
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        self.fc = nn.Linear(hid_dim, num_classes)

    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0).cuda()  # tensor([])
        for i in range(logits.shape[0]):
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def forward(self, f, seq_len):
        f = self.fc_v(f)  ## b, t, 128
        sa = self.cma(f)  ## b, t, 128  
        out = self.fc(sa) ## b, t, 1  
        if seq_len is not None:
            ## topkmean
            out = self.clas(out, seq_len) ## b VL
        return out

net_av = AVCE_Model()
net_v = Single_Model()

f_v = torch.ones((2,200,1024))
f_a = torch.ones((2,200,128))
#f_v = torch.cat((f_v, torch.zeros(2,20,1024)), dim=1)
#print(f_v.shape)
seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
#print(seq_len)
#print(f_v[:, :torch.max(seq_len), :].shape)

#mil_vls, a_sls, v_sls, _, audio_rep, visual_rep = net_av(f_a, f_v, seq_len)


for param_av in net_av.named_parameters():
    if 'sa_a' in param_av[0] or 'fc_a' in param_av[0]:
        continue
    for param_v in net_v.named_parameters():
        if param_av[0] == param_v[0]:
            print("1",param_av[0], param_v[0])
            #param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
            break
        elif param_av[0] == 'att_mmil.fc.weight' and param_v[0] == 'fc.weight':
            print("2",param_av[0], param_v[0])
            #param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
            break
        elif param_av[0] == 'att_mmil.fc.bias' and param_v[0] == 'fc.bias':
            print("3",param_av[0], param_v[0])
            #param_av[1].data.mul_(m).add_((1 - m) * param_v[1].detach().data)
            break
###############



###############
## LOSS
class CMA_MIL(nn.Module):
    def __init__(self):
        super().__init__()

        self.crit = InfoNCE(negative_mode='unpaired')

    def get_new_rep(self, sls, seqlen, rep, lrgst=False, mean=False):
        ## sls: t
        ## seqlen: 1
        ## rep: t, 128
        k_idxs = torch.topk(sls[:seqlen], k=int(seqlen // 16 + 1), largest=lrgst)[1]
        rep_new = rep[k_idxs] 
        if mean:
            print(f"{rep_new.size()=}")
            rep_new = torch.mean(rep_new, 0, keepdim=True).expand( rep_new.size() )
        print(f"{rep_new.shape=}\n")    
        return rep_new
    
    def gather_semi_bags(self, mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
        a_abn = torch.zeros(0)#.cuda()  # tensor([])
        v_abn = torch.zeros(0)#.cuda()  # tensor([])
        a_bgd = torch.zeros(0)#.cuda()  # tensor([])
        v_bgd = torch.zeros(0)#.cuda()  # tensor([])
        a_norm = torch.zeros(0)#.cuda()
        v_norm = torch.zeros(0)#.cuda()
        
        for i in range(a_sls.shape[0]): ## b
            
            ########################
            ## VISUAL BOTTOM/BCKGRND
            v_rep_down = self.get_new_rep(v_sls[i], seqlen[i], v_rep[i], lrgst=False, mean=False) ## k, 128
            v_bgd = torch.cat((v_bgd, v_rep_down), 0) 
            
            #######################
            ## AUDIO BOTTOM/BCKGRND            
            a_rep_down = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False, mean=False)
            a_bgd = torch.cat((a_bgd, a_rep_down), 0)
            
            ########################
            ## UPPER/FRGND
            if mil_vls[i] > 0.5:
                ## In contrast, we conduct average pooling to embeddings 
                ## of all violence instances in each bag and form a semi-bag-level representation
                ## By doing so, the aud and vis repre both express event-level semantics, 
                ## thereby alleviating the noise issue. 
                ## To this end, we construct semi-bag-level positive pairs, 
                ## which are assembled by audio and visual violent semi-bag representations B𝑣𝑖𝑜 𝑎 , B𝑣𝑖𝑜
                
                ## VISUAL 
                v_rep_top = self.get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=True)
                v_abn = torch.cat((v_abn, v_rep_top), 0)
                
                ## AUDIO UPPER/FRGND
                a_rep_top = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=True, mean=True)
                a_abn = torch.cat((a_abn, a_rep_top), 0)

            else:
                ## VISUAL 
                v_rep_top = self.get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=False) ## k, 128
                v_norm = torch.cat((v_norm, v_rep_top), 0) 
                
                ## AUDIO 
                a_rep_top = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=True, mean=False)
                a_norm = torch.cat((a_norm, a_rep_top), 0)
        
        return a_abn, v_abn, a_norm, v_norm, a_bgd, v_bgd                    
    
    
    def forward(self, mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
        a_abn, v_abn, \
        a_norm, v_norm, \
        a_bgd, v_bgd = self.gather_semi_bags(mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep)
        
            
        self.crit = InfoNCE(negative_mode='unpaired')
        if a_norm.size(0) == 0 or a_abn.size(0) == 0:
            return {
                "loss_a2v_abn2bgd": 0.0, 
                "loss_a2v_abn2nor": 0.0, 
                "loss_v2a_abn2bgd": 0.0, 
                "loss_v2a_abn2nor": 0.0
            }
        else:
            ## query, positive_key, negative_keys
            loss_a2v_abn2bgd = self.crit(a_abn, v_abn, v_bgd)
            loss_a2v_abn2nor = self.crit(a_abn, v_abn, v_norm)
            loss_v2a_abn2bgd = self.crit(v_abn, a_abn, a_bgd)
            loss_v2a_abn2nor = self.crit(v_abn, a_abn, a_norm)
            return {
                "loss_a2v_abn2bgd": loss_a2v_abn2bgd, 
                "loss_a2v_abn2nor": loss_a2v_abn2nor, 
                "loss_v2a_abn2bgd": loss_v2a_abn2bgd, 
                "loss_v2a_abn2nor": loss_v2a_abn2nor
            }

b = 2
t = 200
hdim = 128
mil_vls = torch.tensor([0.6, 0.2])
a_sls = torch.randn((b,t))
v_sls = torch.randn((b,t))
seqlen = [32, 32]
a_rep = torch.randn((b,t,hdim))
v_rep = torch.randn((b,t,hdim))

lossfx = CMA_MIL()
loss = lossfx(mil_vls, a_sls, v_sls, seqlen, a_rep, v_rep)
for k, v in loss.items():
    print(k, v)


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x): return x.transpose(-2, -1)

def normalize(*xs): return [None if x is None else F.normalize(x, dim=-1) for x in xs]

'''
def CMAL(mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
    a_abn = torch.zeros(0)#.cuda()  # tensor([])
    v_abn = torch.zeros(0)#.cuda()  # tensor([])
    a_bgd = torch.zeros(0)#.cuda()  # tensor([])
    v_bgd = torch.zeros(0)#.cuda()  # tensor([])
    a_norm = torch.zeros(0)#.cuda()
    v_norm = torch.zeros(0)#.cuda()
    
    for i in range(a_sls.shape[0]): ## b
        if mil_vls[i] > 0.5:
            
            ########################
            ## VISUAL BOTTOM/BCKGRND
            v_sls_down_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
            v_rep_down = v_rep[i][v_sls_down_idxs] 
            print(f"{v_rep_down.shape=}")
            
            v_rep_down2 = get_new_rep(v_sls[i], seqlen[i], v_rep[i], lrgst=False, mean=False)
            assert v_rep_down.shape == v_rep_down2.shape
            
            v_bgd = torch.cat((v_bgd, v_rep_down), 0)

            ## VISUAL UPPER/FRGND
            v_sls_top_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
            v_rep_top = v_rep[i][v_sls_top_idxs]
            v_rep_top = torch.mean(v_rep_top, 0, keepdim=True).expand( v_rep_top.size() )
            
            v_rep_top2 = get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=True)
            assert v_rep_top.shape == v_rep_top2.shape
            
            v_abn = torch.cat((v_abn, v_rep_top), 0)
            
            
            #######################
            ## AUDIO BOTTOM/BCKGRND            
            a_sls_down_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
            a_rep_down = a_rep[i][a_sls_down_idxs]
            #a_rep_down = torch.mean(a_rep_down, 0, keepdim=True).expand( a_rep_down.size() )
            
            a_rep_down2 = get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False)
            assert v_rep_top.shape == v_rep_top2.shape
            
            a_bgd = torch.cat((a_bgd, a_rep_down), 0)

            ## AUDIO UPPER/FRGND
            a_sls_top_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
            a_rep_top = a_rep[i][a_sls_top_idxs]
            a_rep_top = torch.mean(a_rep_top, 0, keepdim=True).expand( a_rep_top.size() )
            
            a_rep_top2 = get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False, mean=True)
            assert a_rep_top.shape == a_rep_top2.shape
            
            a_abn = torch.cat((a_abn, a_rep_top), 0)
            
            
        else:
            ########################
            ## VISUAL BOTTOM/BCKGRND
            v_sls_down_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
            v_rep_down = v_rep[i][v_sls_down_idxs]
            # v_rep_down = torch.mean(v_rep_down, 0, keepdim=True).expand( v_rep_down.size() )
            v_bgd = torch.cat((v_bgd, v_rep_down), 0)

            ## VISUAL UPPER/FRGND
            v_sls_top_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
            v_rep_top = v_rep[i][v_sls_top_idxs]
            # v_rep_top = torch.mean(v_rep_top, 0, keepdim=True).expand( v_rep_top.size() )
            v_norm = torch.cat((v_norm, v_rep_top), 0)
            
            #######################
            ## AUDIO BOTTOM/BCKGRND  
            a_sls_down_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
            a_rep_down = a_rep[i][a_sls_down_idxs]
            # a_rep_down = torch.mean(a_rep_down, 0, keepdim=True).expand( a_rep_down.size() )
            a_bgd = torch.cat((a_bgd, a_rep_down), 0)
            
            ## AUDIO UPPER/FRGND
            a_sls_top_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
            a_rep_top = a_rep[i][a_sls_top_idxs]
            # a_rep_top = torch.mean(a_rep_top, 0, keepdim=True).expand( a_rep_top.size() )
            a_norm = torch.cat((a_norm, a_rep_top), 0)  
'''   
###########


##############
def avce_train(dataloader, model_av, model_v, optimizer_av, optimizer_v, criterion, lamda_a2b, lamda_a2n, logger):
    with torch.set_grad_enabled(True):
        model_av.train()
        model_v.train()
        for i, (f_v, f_a, label) in enumerate(dataloader):
            ## f_v f_a : b, maxseqlen, f
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            ## by doing so t goes back to original shape before pad
            ## or simply reduce amount of pad by consider the max seqlen as TH
            f_v = f_v[:, :torch.max(seq_len), :]
            f_a = f_a[:, :torch.max(seq_len), :]
            f_v, f_a, label = f_v.float().cuda(), f_a.float().cuda(), label.float().cuda()
            mil_vls, a_sls, v_sls, _, audio_rep, visual_rep = model_av(f_a, f_v, seq_len)
            a_sls = a_sls.squeeze()
            v_sls = v_sls.squeeze()
            mil_vls = mil_vls.squeeze()
            clsloss = criterion(mil_vls, label) ## bce
            cmaloss_a2v_a2b, cmaloss_a2v_a2n, cmaloss_v2a_a2b, cmaloss_v2a_a2n = CMAL(mil_vls, a_sls,
                                                                                    v_sls, seq_len, audio_rep,
                                                                                    visual_rep)
            total_loss = clsloss + lamda_a2b * cmaloss_a2v_a2b + lamda_a2b * cmaloss_v2a_a2b + lamda_a2n * cmaloss_a2v_a2n + lamda_a2n * cmaloss_v2a_a2n
            unit = dataloader.__len__() // 2
            if i % unit == 0:
                logger.info(f"Current Lambda_a2b: {lamda_a2b:.2f}, Current Lambda_a2n: {lamda_a2n:.2f}")
                logger.info(
                    f"{int(i // unit)}/{2} MIL Loss: {clsloss:.4f}, CMA Loss A2V_A2B: {cmaloss_a2v_a2b:.4f}, CMA Loss A2V_A2N: {cmaloss_a2v_a2n:.4f},"
                    f"CMA Loss V2A_A2B: {cmaloss_v2a_a2b:.4f},  CMA Loss V2A_A2N: {cmaloss_v2a_a2n:.4f}")

            v_logits = model_v(f_v, seq_len) 
            loss_v = criterion(v_logits, label) ## bce

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = True
            model_v.requires_grad = False
            total_loss.backward()
            optimizer_av.step()

            optimizer_av.zero_grad()
            optimizer_v.zero_grad()
            model_av.requires_grad = False
            model_v.requires_grad = True
            loss_v.backward()
            optimizer_v.step()

        return total_loss, loss_v
