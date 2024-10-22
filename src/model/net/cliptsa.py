import torch
import torch.nn as nn
import torch.nn.functional as F
#import itertools
#from scipy.linalg import pascal

from src.model.pstfwd.utils import PstFwdUtils
from src.model.net.layers import Aggregate, SMlp, Temporal

from src.utils import get_log
log = get_log(__name__)



#################
class PerturbedTopK(nn.Module):
    def __init__(self, k: float, num_samples: int=1000, sigma: float=0.05):
        super(PerturbedTopK, self).__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x, train_mode):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma, train_mode)

class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: float, num_samples:int=1000, sigma: float=0.05, train_mode: bool=True): # k = top-k
        b, t = x.shape  ## b*nc, t
        
        k = int(t * k) ## t*0.95, if t=32 -> k=30
        log.debug(k)
        
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, t)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma # b, n_samples , t


        if k > perturbed_x.shape[-1]:
            k = perturbed_x.shape[-1]
        elif k == 0:
            k = 1

        # k = max(3, k)
        if not train_mode:
            k = min(1000, k)

        #topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False) 
        #indices = topk_results.indices # b, n_samples , k
        #valus = torch.sort(indices, dim=-1).values 
        topk_values = torch.topk(perturbed_x, k=k, dim=-1, sorted=True)[1]  ## b, n_samples , k
        
        perturbed_output = torch.nn.functional.one_hot(topk_values, num_classes=t).float()
        # b, n_samples, k, t
        indicators = perturbed_output.mean(dim=1) # b, k, t

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([ None ] * 5)

        grad_expected = torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, ctx.noise)
        grad_expected /= (ctx.num_samples * ctx.sigma)
        grad_input = torch.einsum("bkde,bke->bd", grad_expected, grad_output)
        return (grad_input,) + tuple([ None ] * 5)

class HardAttention(nn.Module):
    def __init__(self, k, num_samples, in_dim=512):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),       
            nn.Sigmoid()
        )
        self.hard_att = PerturbedTopK(k=k, num_samples=num_samples)

    def forward(self, x):
        ## b*nc, t, 512  ->  b*nc, t, 1
        scores = self.scorer(x)
        b, t, _ = scores.shape 
        #if b > 1: train_mode = True
        #else: train_mode = False
        
        ## b*nc*t,1  ->  b*nc, (k*t), t
        topk = self.hard_att(scores.squeeze(-1), b > 1) 
        
        ## (b*nc, K, t, 1)  *  (b*nc, 1, t, 512) -> (b*nc, K, t, 512)
        out1 = topk.unsqueeze(-1) * x.unsqueeze(1)      
        out2 = torch.sum(out1, dim=1) ## b*nc, t, 512
        
        log.debug(f"HA {x.shape=}")
        log.debug(f"HA {scores.shape=}")
        log.debug(f"HA {topk.shape=}")
        log.debug(f"HA {out1.shape=}")
        log.debug(f"HA {out2.shape=}")
        return out2



class Network(nn.Module):
    def __init__(self, dfeat, _cfg, rgs=None, k=0.7, num_samples=100):
        super().__init__()
        
        self.dfeat = sum(dfeat)
        
        ## always present if in_dim != 512
        if self.dfeat != 512:
            #self.embedding = nn.Sequential(
            #    #nn.Linear(self.dfeat, 1024),
            #    #nn.ReLU(),
            #    nn.Linear(self.dfeat, 512),
            #    nn.Sigmoid()
            #)
            self.embedding = Temporal(self.dfeat, 512 )
        else: self.embedding = nn.Identity()
        
        self.hard_attention = HardAttention(k=k, num_samples=num_samples, in_dim=512)
        
        self.aggregate = Aggregate(512)
        self.do = nn.Dropout( _cfg.do )
        
        #self.rgs =  nn.Sequential(
        #    nn.Linear(512, 128),
        #    nn.ReLU(),
        #    nn.Dropout(_cfg.do), 
        #    nn.Linear(128, 32),
        #    nn.ReLU(),
        #    nn.Dropout(_cfg.do), 
        #    nn.Linear(32, 1),
        #    nn.Sigmoid(),
        #    )
        #self.rgs = rgs
        self.rgs = SMlp(dfeat=[512], hdim_ratio=[4,4], do=0.7)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        log.debug(f"{x.shape}")
        
        #x = self.embedding(x) ## b*nc, t, 512
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        log.debug(f"{x.shape=}")
        
        ## != clip  and  bs > 1 ()
        #if self.visual != "vit" and x.shape[0] > 1 and x.shape[1] != 32:
        #    concat = []
        #    for i in x:
        #        concat.append(self.hard_attention(i.unsqueeze(0)))
        #    x_ha = torch.cat(concat, dim=0)
        #else:
        x_ha = self.hard_attention(x)
        log.debug(f"{x_ha.shape=}")
        
        x_new = self.aggregate( x_ha.permute(0,2,1) ).permute(0,2,1) ## (b, t, f)
        x_new = self.do(x_new)
        log.debug(f"{x_new.shape=}")
        
        
        #scores = self.rgs(x_new).view(b,t)
        scores = self.sig( self.rgs(x_new) )
        log.debug(f"{scores.shape=}")
        
        return { 
            'scores': scores, 
            'feats': x_new 
        } 
        

class Infer():
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self._cfg = _cfg
        self.pfu = pfu
        
    def __call__(self, ndata): 
        scores = self.pfu.uncrop(ndata['scores'], 'mean')
        return scores


'''        
class Model(nn.Module):
    def __init__(self, n_features, batch_size, k=0.95, num_samples=10, apply_HA=True, args=None):
        super(Model, self).__init__()
        args = {}
        visual = 'I3D'
        gpu = [0]
        
        OG_feat = n_features
        if visual.upper() in ["I3D", "C3D"]: # and args.enable_HA:            
            n_features = 512

        self.hard_attention = HardAttention(k=k, num_samples=num_samples, in_dim=n_features)
        self.apply_HA = apply_HA
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = self.num_segments // 10
        self.k_nor = self.num_segments // 10

        
        self.mlp = MLP(in_dim=OG_feat)
        
        ## n_features = 2048 -> 1 -> agg: 2048 rgs: 2048 -> 512 -> 128 -> 1
        ## n_features = 1024 -> 2 -> agg: 1024 -> rgs: 1024 -> 512 -> 128 -> 1
        
        self.division = 2048 // n_features 
        #self.division = 2048 // 512 ## 4
        
        self.Aggregate = Aggregate(len_feature=2048 // self.division)
        #self.Aggregate = Aggregate(len_feature=512)
        
        self.fc1 = nn.Linear(n_features, 512 // self.division)
        self.fc2 = nn.Linear(512 // self.division, 128 // self.division)
        self.fc3 = nn.Linear(128 // self.division, 1)
        #self.fc1 = nn.Linear(512,128)
        #self.fc2 = nn.Linear(128, 32)
        #self.fc3 = nn.Linear(32, 1)
        
        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

        self.parallel = 0.5 if "," in gpu else 1
        self.visual = visual

    def forward(self, inputs):
        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs # ^torch.Size([64, 10, 32, 2048]), *[64,1,32,512]
        bs, ncrops, t, f = out.size() # => torch.Size([1, 1, 89, 512]), ^torch.Size([64, 10, 32, 2048])

        out = out.view(-1, t, f) # => ^[640, 32, 2048] bs*nc, t, f 

        if f > 512:
            out = self.mlp(out) ## down_res
            f = 512

        ## b*nc, t, 512
        
        if self.apply_HA:
            if self.visual != "vit" and out.shape[0] > 1 and out.shape[1] != 32:
                concat = []
                for i in out:
                    concat.append(self.hard_attention(i.unsqueeze(0)))
                out = torch.cat(concat, dim=0)
            else:
                out = self.hard_attention(out)
        return 
    
        out = self.Aggregate(out) # ^[640, 32, 2048]
        out = self.drop_out(out) # => torch.Size([1, 89, 512]), ^[640, 32, 2048], *[64,32,512]

        features = out # torch.Size([1, 89, 512]), *[64,32,512]
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        scores = scores.view(bs, ncrops, -1).mean(1) # => ^torch.Size([64, 32])
        scores = scores.unsqueeze(dim=2)    # => torch.Size([1, 89, 1]), ^[64,32,1], *[64, 32, 1]

        ####################
        
        # Place self.hard_attention here and remove MLP inside hard attention
        adjusted_scoremag_batch_size = int(self.batch_size * self.parallel)
        adjusted_feat_batch_size = int(self.batch_size*ncrops * self.parallel)

        normal_features = features[0:adjusted_feat_batch_size] # torch.Size([1, 89, 512]), ^[320, 32, 2048], *[64, 32, 512]
        normal_scores = scores[0:adjusted_scoremag_batch_size]

        abnormal_features = features[adjusted_feat_batch_size:] # torch.Size([0, 89, 512]), ^[320, 32, 2048], *[0, 32, 512], +[32, 32, 512]
        abnormal_scores = scores[adjusted_scoremag_batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1) # [1, 89], ^[64, 32], *+[64,32]
        nfea_magnitudes = feat_magnitudes[0:adjusted_scoremag_batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[adjusted_scoremag_batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0] # 1, ^32, +32

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features # == torch.Size([1, 89, 512])

        #######  process abnormal videos -> select top3 feature magnitude  #######
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        afea_magnitudes_drop = afea_magnitudes * select_idx

        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]

        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]]) # => torch.Size([1, 3, 512]), ^[32, 3, 2048], +[32, 3, 512]
        abnormal_features = abnormal_features.view(n_size, ncrops, t, f) # => torch.Size([1, 1, 197, 512]), ^[32, 10, 32, 2048], +[1,32,32,512]
        abnormal_features = abnormal_features.permute(1, 0, 2, 3) # => ^[10, 32, 32, 2048], +[1, 32, 32, 512]

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude

        ####### process normal videos -> select top3 feature magnitude #######
        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]

        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes

'''