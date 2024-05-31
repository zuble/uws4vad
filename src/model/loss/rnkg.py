import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)


class Loss(nn.Module):
    def __init__(self, _cfg):
        super(Loss, self).__init__()
        log.info(_cfg)
        ## when instanteate partial cant do _cfg.bs
        self.bs = _cfg.get("bs")
        self.seglen = _cfg.get("seglen")
        self.lambda1 = _cfg.get("lambda12")[0]
        self.lambda2 = _cfg["lambda12"][1] 

    ## MotionAware found it harmfull
    def smooth(self, arr):
        '''
        slides arr one index in negative direction
        and copys (1 to last) to last
        '''
        arr2 = torch.concatenate([arr[1:], arr[-1:]], dim=0)
        loss = torch.sum( (arr2-arr) ** 2 ) 
        return self.lambda1 * loss
    
    def sparsity(self, arr, rtfm=False):
        if rtfm: 
            loss = torch.mean(torch.norm(arr, dim=0))
        else: 
            loss =  torch.sum(arr)
        return self.lambda2 * loss

    def forward(self, slscores):
        ## https://github.com/Roc-Ng/DeepMIL
        
        if slscores.ndim == 2:
            slscores = slscores.view(-1)
            log.debug(f"{slscores.shape}")
            
        L = []
        for i in range(self.bs//2):
            ## norm
            startn = i * self.seglen
            endn = (i + 1) * self.seglen
            maxn = torch.max( slscores[ startn : endn ] ) 
            #maxn = torch.mean( torch.topk( slscores[ startn : endn ], k=self.seglen//4) )
            
            ## anom
            starta = (i * self.seglen + (self.bs//2) * self.seglen)
            enda = (i + 1) * self.seglen + (self.bs//2) * self.seglen
            maxa = torch.max( slscores[ starta : enda ] ) ##that
            #maxa = torch.mean( torch.topk( slscores[ starta : enda ], k=self.seglen//4) )
            
            tmp = F.relu(1.0 - maxa + maxn)
            loss = tmp + self.sparsity(slscores[ starta : enda ]) ## + self.smooth(slscores[ starta : enda ])
            
            ## TCN-IBL inner bag loss
            #mina = np.min( slscores[ starta : enda ] )
            #minn = np.min( slscores[ startn : endn ] )
            #loss_ibl = npx.relu(1.0 - maxa + mina)
            #loss_gap = np.abs(maxn - minn)
            #loss = loss + loss_ibl + loss_gap
            L.append(loss)
        L = torch.stack(L, dim=0)

        loss_mil = torch.mean(L)
        log.debug(f'RNKG/{loss_mil=} {loss_mil.shape=}')
        
        return {
            'rnkg': loss_mil
        }
    