import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)

class Bce(nn.Module):
    def __init__(self):
        super(Bce, self).__init__()
        self.crit = nn.BCELoss()
        
    def forward(self, scores, label):
        #log.debug(f"Bce/{scores.shape} {scores.context} {label.shape} {label.context}")
        l = self.crit(scores, label)
        return {
            'loss_bce': l
            }

def smooth(arr, lambd = 8e-4):
    '''
    slides arr one index in negative direction
    and copys (1 to last) to last
    '''
    arr2 = torch.cat([arr[1:], arr[-1:]], dim=0)
    loss = torch.sum( (arr2-arr) ** 2 ) 
    return {
        "smooth": lambd * loss
    }
    
def sparsity(arr, lambd = 8e-3, rtfm=False):
    if rtfm: 
        loss = torch.mean(torch.norm(arr, dim=0))
    else: 
        loss =  torch.sum(arr)
    return {
        "spars": lambd * loss
    }