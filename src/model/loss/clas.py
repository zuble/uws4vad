import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)


class Loss(nn.Module):
    def __init__(self, _cfg):
        super(Loss, self).__init__()

        self.crit = nn.BCELoss()        
        self.forward = {
            'topk': self.fwd_topk,
            'full': self.fwd_full
        }.get(_cfg.fx)
        
        self.k = _cfg.k
    
    def get_k(self, x):
        if self.k == -1: return int(x//16+1)
        else: return min(x, self.k)
                
    def fwd_topk(self, scores, ldata):
        label = ldata['label']
        seqlen = ldata['seqlen']
        
        #log.debug(f"BCE/{scores.shape} {scores.context} {label.shape} {label.context}")
        scores = scores.squeeze()
        instance_scores = torch.zeros(0).to(scores.device)  # tensor([])
        for i in range(scores.shape[0]):
            tmp, _ = torch.topk(scores[i][:seqlen[i]], k=self.get_k(seqlen[i]), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_scores = torch.cat((instance_scores, tmp))

        instance_scores = torch.sigmoid(instance_scores)
        l = self.crit(instance_scores, label)
        return {
            'clas': l
            }
            
    def fwd_full(self, scores, ldata):
        seqlen = ldata['seqlen']
        
        vl_scores = torch.zeros(0).to(scores.device)
        for i in range(scores.shape[0]): #.self.bs
            sl = int(seqlen[i])
            tmp3 = np.mean(scores[i, :sl])
            vl_scores.append( np.expand_dims(tmp3,axis=0) ) 
        vl_scores = np.concatenate(vl_scores, axis=0)
        
        l = self.crit(vl_scores, label)
        return {
            'clas': l
            }