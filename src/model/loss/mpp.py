import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_normal = 1.
        
    def forward(self, normal_scores):
        ## bag_normal, t -> bag_normal
        l = torch.norm(normal_scores, dim=1, p=2).mean()
        return {
            'norm': self.w_normal * l
            }
    
class MPPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_triplet = [5, 20]
        self.w_mpp = 1.

    def forward(self, anchors, variances, nor_sel, abn_sel):
        losses_triplet = []

        def mahalanobis_distance(mu, x, var):
            return torch.sqrt(torch.sum((x - mu)**2 / var, dim=-1))
        
        # nor_sel / abn_sel are result of DFM (topk malh_dist embd:anchor  ) 
        
        
        for anchor, var, pos, neg, wt in zip(anchors, variances, nor_sel, abn_sel, self.w_triplet):
            triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1, distance_function=partial(mahalanobis_distance, var=var))
            
            B, C, k = pos.shape
            pos = pos.permute(0, 2, 1).reshape(B*k, -1)
            neg = neg.permute(0, 2, 1).reshape(B*k, -1)
            loss_triplet = triplet_loss(anchor[None, ...].repeat(B*k, 1), pos, neg)
            losses_triplet.append(loss_triplet * wt)
        
        l = sum(losses_triplet)
        return {
            'mpp': self.w_mpp * l
        } 
