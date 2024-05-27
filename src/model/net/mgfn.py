import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

## POST PROCESSOR / FEAT GATHER
def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k):
    #magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[0:batch_size * 10]  # [b/2*ten,32,1024]
    normal_scores = scores[0:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size * 10:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes)#.cuda()
    select_idx = drop_out(select_idx)


    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0)
    for abnormal_feature in abnormal_features:
        feat_select_abn = torch.gather(abnormal_feature, 1,
                                    idx_abn_feat)
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)


    select_idx_normal = torch.ones_like(nfea_magnitudes)#.cuda()
    select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0)
    for nor_fea in normal_features:
        feat_select_normal = torch.gather(nor_fea, 1,
                                        idx_normal_feat)
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


def FeedForward(dim, repe = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv1d(dim, dim * repe, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(dim * repe, dim, 1)
    )

# MHRAs (multi-head relation aggregators)
class FOCUS(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads ## stage2: 64*2h, stage3: 64*16h
        self.norm = nn.BatchNorm1d(dim)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, bias = False)
        self.rel_pos = nn.Conv1d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x) #(b*crop,c,t)
        b, c, *_, h = *x.shape, self.heads
        v = self.to_v(x) #(b*crop,c,t)
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h) #(b*ten*64,c/64,32)
        out = self.rel_pos(v)
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

class GLANCE(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads ## stage1: 64*1h
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)
        self.attn =0

    def forward(self, x):
        ## b*nc, dim, t
        x = self.norm(x)
        shape, h = x.shape, self.heads
        x = rearrange(x, 'b c ... -> b c (...)')
        
        q, k, v = self.to_qkv(x).chunk(3, dim = 1) ## b*nc, dim, t
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v)) ## b*nc, 1, t, dim  
        q = q * self.scale
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) ## att scores
        self.attn = sim.softmax(dim = -1) ## b*nc, 1, t, t prob
        out = einsum('b h i j, b h j d -> b h i d', self.attn, v) ## b*nc, 1, t, dim  weighted sum of values
        ## Concatenation of Head Outputs
        out = rearrange(out, 'b h n d -> b (h d) n', h = h) ## b*nc, dim, t
                
        out = self.to_out(out)
        return out.view(*shape)


class Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim, ## 64, 128, 1024
        depth, ## 3, 3, 2
        heads, ## 1, 2, 16
        mgfn_type = 'gb', ## 'gb', 'fb', 'fb'
        kernel = 5,
        dim_headnumber = 64,
        ff_repe = 4,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads = heads, dim_head = dim_headnumber, local_aggr_kernel = kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else: raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding = 1),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
            ]))

    def forward(self, x):
        for i, (scc, attention, ffn) in enumerate(self.layers):
            print(f"BACKBONE @ depth {i}")
            x = scc(x) + x  ## encode + residual
            print(f"scc [{i}] {x.shape}")
            x = attention(x) + x ## GLANCE -> FOCUS -> FOCUS
            print(f"att [{i}] {x.shape}")
            x = ffn(x) + x
            print(f"ffn [{i}] {x.shape}")
        return x


class mgfn(nn.Module):
    def __init__(
        self,
        *,
        classes=0,
        dims = (64, 128, 1024),
        depths = (3, 3, 2),
        mgfn_types = ( 'gb', 'fb', 'fb'),
        lokernel = 5,
        channels = 2048,
        ff_repe = 4,
        dim_head = 64,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))
        
        self.stages = nn.ModuleList([])
        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind] ## 64, 128, 1024
            heads = stage_dim // dim_head ## 1, 2, 16

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mgfn_type = mgfn_types,
                    ff_repe = ff_repe,
                    dropout = dropout,
                    attention_dropout = attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride = 1),
                ) if not is_last else None
            ]))
        print(self.stages,"\n\n\n\n")
        
        self.mag_ratio = 0.1
        self.batch_size =  16 
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(0.7)
        
        ## assert dim and compowr save
        ## feature map dimension from 𝐶 in 𝐹𝐹𝐴𝑀 to 𝐶/32.
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride = 1, padding = 1)
        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)
        
        self.to_logits = nn.Sequential( nn.LayerNorm(last_dim) )
        
    def forward(self, video):
        k = 3
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
        
        ## Feature Amplification Mechanism (FAM)
        x_f = x[:,:2048,:] ## bs*nc, 2048, 32
        x_m = x[:,2048:,:] ## bs*nc, 1, 32

        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m) 
        x_f = x_f + self.mag_ratio*x_m ## bs*nc, 64, 32

        
        ##############
        for i, (backbone, conv) in enumerate(self.stages):
            print(f"MGFN @ STAGE {i} ")
            x_f = backbone(x_f)
            print(f"MGFN / after backbone {x_f.shape}")
            if exists(conv):
                x_f = conv(x_f)
                print(f"MGFN / conv dim prep {x_f.shape}")
        #############

        x_f = x_f.permute(0, 2, 1) ## bs*nc, 32, 1024
        x =  self.to_logits(x_f)
        scores = self.sigmoid(self.fc(x))  ## bs*nc, 32, 1
        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores  = MSNSD(x,scores,bs,self.batch_size,self.drop_out,ncrops,k)
        print(f"{score_abnormal.shape=} {score_normal.shape=} {abn_feamagnitude.shape=} {nor_feamagnitude.shape=} {scores.shape=}")
        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True) ## bag*nc, 1
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class mgfn_loss(torch.nn.Module):
    def __init__(self, alpha):
        super(mgfn_loss, self).__init__()
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss()
        
    def forward(self, score_normal, score_abnormal, nor_feamagnitude, abn_feamagnitude):
        #label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()
        #label = label.cuda()
        seperate = len(abn_feamagnitude) / 2

        #loss_cls = self.criterion(score, label)
        
        ## different from rtfm, no mean over topk sel feats
        ## thus abn/nor_feamagnitude = bag*nc, k, f 
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), ## b*nc, k
                                    torch.norm(nor_feamagnitude, p=1, dim=2), ## b*nc, k
                                    1)  # try tp separate normal and abnormal
        ## bag*nc, k, f 
        loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2), ## (b*nc)/2, k
                                    torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),  ## (b*nc)/2, k
                                    0)  # try to cluster the same class 
        ## bag*nc, k, f 
        loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2), ## (b*nc)/2, k
                                    torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2), ## (b*nc)/2, k
                                    0)
        
        #loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n )
        return 0.0



f = torch.randn((32, 10, 32, 2049))
net = mgfn()

score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = net(f)

scores = scores.view(16 * 32 * 2, -1)
scores = scores.squeeze()

#nlabel = nlabel[0:batch_size]
#alabel = alabel[0:batch_size]

loss_criterion = mgfn_loss(0.0001)

cost = loss_criterion(score_normal, score_abnormal, nor_feamagnitude, abn_feamagnitude)