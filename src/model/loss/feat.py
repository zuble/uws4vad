import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
from pytorch_metric_learning import reducers
from pytorch_metric_learning import regularizers


from functools import partial

from src.utils.logger import get_log
log = get_log(__name__)
    
    
class MPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_triplet = [5, 20]
        self.w_mpp = 1.
        
    def forward(self, anchors, variances, abn_sel, nor_sel):
        def mahalanobis_distance(mu, x, var):
            return torch.sqrt(torch.sum((x - mu)**2 / var, dim=-1))
        
        # nor_sel / abn_sel are result of DFM (topk malh_dist embd:anchor  ) 
        
        L = []
        for anchor, var, pos, neg, wt in zip(anchors, variances, nor_sel, abn_sel, self.w_triplet):
            triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1, distance_function=partial(mahalanobis_distance, var=var))
            
            B, C, k = pos.shape
            pos = pos.permute(0, 2, 1).reshape(B*k, -1)
            neg = neg.permute(0, 2, 1).reshape(B*k, -1)
            loss_triplet = triplet_loss(anchor[None, ...].repeat(B*k, 1), pos, neg)
            L.append(loss_triplet * wt)
            
        L = torch.stack(L, dim=0).sum()
        log.debug(f'{L=}')
        return {
            'mpp': self.w_mpp * L,
            #'mpp2': self.w_mpp * l[1]
        } 
        
class Rtfm(nn.Module):
    def __init__(self, _cfg):
        super(Rtfm, self).__init__()
        self.alpha = _cfg.alpha
        self.margin = _cfg.margin
        
    def forward(self, abn_feat, nor_feat):
        ## bag*nc,f

        ## needs mil
        l2n_abn = torch.norm(abn_feat, p=2, dim=1) ## (bag*nc)
        l_fabn = torch.abs(self.margin - l2n_abn)
        l_fnor = torch.norm(nor_feat, p=2, dim=1) ## (bag*nc)
        l = torch.mean((l_fabn + l_fnor) ** 2)
        
        log.debug(f"{l_fabn.shape=} {l_fnor.shape=} ")
        
        return  {
            'rtfm': self.alpha * l,
        }
        
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True) ## bag*nc, 1
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class MagnCont(nn.Module):
    def __init__(self, _cfg):
        super(MagnCont, self).__init__()
        self.alpha = _cfg.alpha
        self.margin = _cfg.margin
        
        self.crit = ContrastiveLoss(_cfg.margin)

    def forward(self, abn_feat, nor_feat):
        
        ## different from rtfm, no mean over topk sel feats
        ## thus abn/nor_feamagnitude = bag*nc, k, f 
        loss_con = self.crit(torch.norm(abn_feamagnitude, p=1, dim=2), ## bag*nc, k
                                    torch.norm(nor_feamagnitude, p=1, dim=2), ## bag*nc, k
                                    1)  # try tp separate normal and abnormal
        ## bag*nc, k, f 
        loss_con_n = self.crit(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2), ## (bag*nc)/2, k
                                    torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),  ## (bag*nc)/2, k
                                    0)  # try to cluster the same class 
        ## bag*nc, k, f 
        loss_con_a = self.crit(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2), ## (bag*nc)/2, k
                                    torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2), ## (bag*nc)/2, k
                                    0)
        
        #loss_total = loss_cls + 0.001 * (0.001 * loss_con + loss_con_a + loss_con_n )
        
        return {
            'mc': 0
        }





class Dev(nn.Module):
    def __init__(self, _cfg):
        super(Dev, self).__init__()
        
        #self.loss_feat = losses.TripletMarginLoss(
        #    margin=0.2,# _cfg.margin,
        #    swap=False,
        #    smooth_loss=False,
        #    triplets_per_anchor="all",
        #    #distance=distances.CosineSimilarity,
        #    #reducer = reducers.ThresholdReducer(high=0.3)
        #    )
        
        #   Specifically, given an image xi, the learning objective is to push its negative points
        #   farther than a boundary α and pull its positive ones closer than
        #   another boundary α − m. Thus m becomes the margin between two boundaries
        self.loss_feat = losses.RankedListLoss(
            margin=10,  # margin betwen neg & pos
            Tn=0.5,       # Temperature for neg samples
            imbalance=0.8, # Give more weight to positive ranking
            alpha=1,    # min dist betwen anc and neg samples
            #distance=distances.CosineSimilarity() 
            #embedding_regularizer = regularizers.LpRegularizer()
        )
        
    def forward(self, abn_embds, nor_embds):
        embds = torch.cat((abn_feat,nor_feat), dim=0)
        labels = torch.cat((torch.ones(abn_feat.shape[0]), torch.zeros(nor_feat.shape[0])))

        log.warning(f"{abn_feat.shape=} {nor_feat.shape=} {embds.shape=} {labels.shape=}")
        
        #abn_norm = torch.norm(abn_feat, p=2, dim=1)
        #nor_norm = torch.norm(nor_feat, p=2, dim=1)
        #log.error(f"{abn_norm=} {nor_norm=}")
        
        
        l = self.loss_feat(embds,labels)
        log.debug(f"{l=}")
        return  {
            'dev': l,
        }
        