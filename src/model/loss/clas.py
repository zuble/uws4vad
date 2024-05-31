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
        self.topk = _cfg.topk
            
    def fwd_topk(self, scores, label, seqlen):
        #log.debug(f"BCE/{scores.shape} {scores.context} {label.shape} {label.context}")
        scores = scores.squeeze()
        instance_scores = torch.zeros(0).to(scores.device)  # tensor([])
        for i in range(scores.shape[0]):
            tmp, _ = torch.topk(scores[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
            tmp = torch.mean(tmp).view(1)
            instance_scores = torch.cat((instance_scores, tmp))

        instance_scores = torch.sigmoid(instance_scores)

        l = self.crit(instance_scores, label)
        return {
            'loss_clas': l
            }
            
    def fwd_full(self, scores, seqlen):
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