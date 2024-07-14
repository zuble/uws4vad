import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.net.layers import BasePstFwd

from src.utils.logger import get_log
log = get_log(__name__)


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
    
    
class Bce(nn.Module):
    def __init__(self, ):
        super(Bce, self).__init__()
        self.crit = nn.BCELoss()
        
    def forward(self, scores, label, tag=None):
        log.debug(f"Bce/{scores.shape} {scores.device} {label.shape} {label.device}")
        #log.debug(f"Bce/{scores=} {label=}")
        return {
            f'bce-{tag}' if tag else 'bce': self.crit(scores, label)
            }

class Clas(nn.Module):
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
        
        
class Ranking(nn.Module, BasePstFwd):
    def __init__(self, _cfg):
        super(Ranking, self).__init__()
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
        arr2 = torch.cat([arr[1:], arr[-1:]], dim=0)
        loss = torch.sum( (arr2-arr) ** 2 ) 
        return self.lambda1 * loss
    
    def sparsity(self, arr, rtfm=False):
        if rtfm: 
            loss = torch.mean(torch.norm(arr, dim=0))
        else: 
            loss =  torch.sum(arr)
        return self.lambda2 * loss

    def forward(self, scores, labels, use_tcn=False):
        log.debug(f" {scores.shape=} {labels.shape=}")

        loss = torch.tensor(0., requires_grad=True, device=scores.device)
        assert self.bs == scores.shape[0]


        #if self.use_tcn:
        L_tcn = loss.clone()
        #scores = super().uncrop(scores, 'mean')
        #log.info(f" {scores.shape=} ")
        #abn_scors, nor_scors = super().unbag(scores,labels)
        #log.info(f" {abn_scors.shape=} {nor_scors.shape=}")
        
        #scores_abn = scores[labels == 1]#.view(-1, self.seglen) 
        #scores_nor = scores[labels == 0]#.view(-1, self.seglen) 
        #log.info(f" {scores_abn.shape=} {scores_nor.shape=}")
        #super().is_equal(nor_scors, scores_nor)
        #super().is_equal(abn_scors, scores_abn)
        
        #max_scores_nor = torch.max(scores_nor, dim=1).values
        #max_scores_abn = torch.max(scores_abn, dim=1).values
        #min_scores_nor = torch.min(scores_nor, dim=1).values
        #min_scores_abn = torch.min(scores_abn, dim=1).values
        #
        #loss_rnkg = torch.sum(F.relu(
        #    1 - max_scores_abn[:, None] + max_scores_nor[None, :]
        #))
        #
        #loss_ibl = torch.sum(F.relu(1 - max_scores_abn + min_scores_abn))
        #loss_gap = torch.sum(torch.abs(max_scores_nor - min_scores_nor))
        #
        #log.warning(f"L tcn: RNKG {loss_rnkg.item()} | IBL: {loss_ibl.item()} | GAP: {loss_gap.item()}")
        #
        #L_tcn += loss_ibl + loss_gap + loss_rnkg
        #L_tcn += torch.sum(self.smooth(scores_abn)) 
        #L_tcn += torch.sum(self.sparsity(scores_abn)) 
        #
        #mask = labels == 1
        #L_tcn /= mask.sum()
        #log.warning(f"L tcn final: {L_tcn.item()}")
            
        
        L_sult = loss.clone()
        max_scores = torch.max(scores, dim=1)[1]
        for i in range(self.bs):
            if labels[i] == 1:
                L_sult += self.smooth(scores[i])
                L_sult += self.sparsity(scores[i])
                mask = labels == 0
                L_sult += torch.sum(F.relu(1 - max_scores[i] + max_scores[mask]))
        L_sult /= self.bs
        log.warning(f"L sult {L_sult.item()}")

        return {
            'rnkg_sult': L_sult,
            #'rnkg_tcn': L_tcn
            } 

    
    #def forward(self, scores):
    #    ## https://github.com/Roc-Ng/DeepMIL
    #    
    #    if scores.ndim == 2:
    #        scores = scores.view(-1)
    #        log.debug(f"{scores.shape}")
    #        
    #    L = []
    #    for i in range(self.bs//2):
    #        ## norm
    #        startn = i * self.seglen
    #        endn = (i + 1) * self.seglen
    #        maxn = torch.max( scores[ startn : endn ] ) 
    #        #maxn = torch.mean( torch.topk( scores[ startn : endn ], k=self.seglen//4) )
    #        
    #        ## anom
    #        starta = (i * self.seglen + (self.bs//2) * self.seglen)
    #        enda = (i + 1) * self.seglen + (self.bs//2) * self.seglen
    #        maxa = torch.max( scores[ starta : enda ] ) ##that
    #        #maxa = torch.mean( torch.topk( scores[ starta : enda ], k=self.seglen//4) )
    #        
    #        tmp = F.relu(1.0 - maxa + maxn)
    #        loss = tmp + self.sparsity(scores[ starta : enda ]) ## + self.smooth(scores[ starta : enda ])
    #        
    #        ## TCN-IBL inner bag loss
    #        #mina = np.min( scores[ starta : enda ] )
    #        #minn = np.min( scores[ startn : endn ] )
    #        #loss_ibl = npx.relu(1.0 - maxa + mina)
    #        #loss_gap = np.abs(maxn - minn)
    #        #loss = loss + loss_ibl + loss_gap
    #        L.append(loss)
    #    L = torch.stack(L, dim=0)
    #    loss_mil = torch.mean(L)
    #    log.debug(f'RNKG/{loss_mil=} {loss_mil.shape=}')
    #    
    #    return {
    #        'rnkg': loss_mil
    #    }
    
            
class Normal(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_normal = 1.
        
    def forward(self, normal_scores):
        ## bag_normal, t -> bag_normal
        #l = torch.norm(normal_scores, dim=1, p=2).mean()
        l = torch.linalg.norm(normal_scores, ord=2, dim=1).mean()
        return {
            'norm': self.w_normal * l
            }


class MultiBranchSupervision(nn.Module):
    def __init__(self, _cfg):
        super(MultiBranchSupervision, self).__init__()
        self.device = devices
        self.eps = 0.2
        self.alpha = 0.8
        self.gamma = 0.8
        self.µ = 0.01
        self.cur_stp = 1
        self.M = 400
        self.bce = nn.BCEWithLogitsLoss()
        self.vis = vis

    def forward(self, mdata, ldata):
        log.debug(f"mbattsup [{self.cur_stp}]")
        if self.cur_stp == self.M:
            log.warning("LG_1 (L pos guide) calc changing")

        x_cls, x_att = mdata["slscores"], mdata["attw"]

        bs = x_cls.shape[0] // 2
        slen = x_cls.shape[1]
        dbg_stp = 100
        L = None

        for bi in range(bs): ## assumes mil
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f"mbattsup[{bi}/{bs}]")

            # NORMAL
            SO_0 = x_cls[bi]
            Ai_0 = x_att[bi]
            SA_0 = Ai_0 * SO_0
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/Scores: SO_0 {SO_0.shape}, Ai_0 {Ai_0.shape}, SA_0 {SA_0.shape}')

            thetai_0 = ((torch.max(Ai_0) - torch.min(Ai_0)) * self.eps) + torch.min(Ai_0)
            SSO_0 = SO_0 * (Ai_0 < thetai_0)
            SSA_0 = SA_0 * (Ai_0 < thetai_0)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/SupressedScores: SSO_0 {SSO_0.shape}, SSA_0 {SSA_0.shape}')

            # ABNORMAL
            SO_1 = x_cls[bs + bi]
            Ai_1 = x_att[bs + bi]
            SA_1 = Ai_1 * SO_1
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/Scores: SO_1 {SO_1.shape}, Ai_1 {Ai_1.shape}, SA_1 {SA_1.shape}')

            thetai_1 = ((torch.max(Ai_1) - torch.min(Ai_1)) * self.eps) + torch.min(Ai_1)
            SSO_1 = SO_1 * (Ai_1 < thetai_1)
            SSA_1 = SA_1 * (Ai_1 < thetai_1)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/SupressedScores: SSO_1 {SSO_1.shape}, SSA_1 {SSA_1.shape}')

            ## Guide , eq(9,10)
            LG_0 = torch.mean((Ai_0 - 0) ** 2) ## MSE((Aneg, {0 · · · 0}))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossGuide: LG_0 {LG_0}')

            if self.cur_stp < self.M: lg1_tmp = SO_1
            else: lg1_tmp = (SO_1 > 0.5).float()
            LG_1 = torch.mean((Ai_1 - lg1_tmp) ** 2)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossGuide: LG_1 {LG_1}')
                
            ## Loss L1-Norm , eq(11)
            LN_1 = torch.sum(torch.abs(Ai_1))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossNorm: LN_1 {LN_1}')

            ## Loss SL scores
            z = torch.zeros((1, slen), device=self.device)
            o = torch.ones((1, slen), device=self.device)
            ## Raw Org/Att , eq(12)
            LC_SO = self.bce(SO_0.unsqueeze(0), z) + self.bce(SO_1.unsqueeze(0), o)
            LC_SA = self.bce(SA_0.unsqueeze(0), z) + self.bce(SA_1.unsqueeze(0), o)
            ## Supressed Org/Att , eq(12)
            LC_SSO = self.bce(SSO_0.unsqueeze(0), z) + self.bce(SSO_1.unsqueeze(0), o)
            LC_SSA = self.bce(SSA_0.unsqueeze(0), z) + self.bce(SSA_1.unsqueeze(0), o)

            LC = self.alpha * (LC_SO + LC_SA) + (1 - self.alpha) * (LC_SSO + LC_SSA) ## eq(13)
            if (self.cur_stp % dbg_stp) == 0: 
                log.debug(f'mbattsup[{bi}/{bs}]/LossCls: LC_SO {LC_SO}, LC_SA {LC_SA}, LC_SSO {LC_SSO}, LC_SSA {LC_SSA}, LC_ALL {LC}')
            
            ## Loss Smoothness , eq(14)
            LSmt_SOrg = torch.sum((SO_0[:-1] - SO_0[1:]) ** 2) + torch.sum((SO_1[:-1] - SO_1[1:]) ** 2)
            LSmt_SAtt = torch.sum((SA_0[:-1] - SA_0[1:]) ** 2) + torch.sum((SA_1[:-1] - SA_1[1:]) ** 2)
            LSmt = LSmt_SOrg + LSmt_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossSmooth: SO {LSmt_SOrg}, SA {LSmt_SAtt}, SUM {LSmt}')
            
            ## Loss Sparsity , eq(14)
            LSpr_SOrg = torch.sum(SO_0) + torch.sum(SO_1)
            LSpr_SAtt = torch.sum(SA_0) + torch.sum(SA_1)
            LSpr = LSpr_SOrg + LSpr_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbattsup[{bi}/{bs}]/LossSparse: SO {LSpr_SOrg}, SA {LSpr_SAtt}, SUM {LSpr}')

            tmp = LC + self.gamma * LN_1 + LG_0 + LG_1 + self.µ * LSmt + LSpr  ## eq(15)

            self.plot_loss(LC, self.gamma * LN_1, LG_0, LG_1, self.µ * LSmt, LSpr, tmp)

            if L is None:
                L = tmp
            else:
                L = torch.cat((L, tmp), dim=0)
                
        self.cur_stp += 1
        return L

    def plot_loss(self, LC , LN_1, LG_0, LG_1, LSmt, LSpr, L):
        self.vis.plot_lines('LCls (LC)', LC.detach().cpu().numpy())
        self.vis.plot_lines('LNorm (LN_1)', LN_1.detach().cpu().numpy())
        self.vis.plot_lines('LGuia_0 (LG_0)', LG_0.detach().cpu().numpy())
        self.vis.plot_lines('LGuia_1 (LG_1)', LG_1.detach().cpu().numpy())
        self.vis.plot_lines('LSmth (LSmt)', LSmt.detach().cpu().numpy())
        self.vis.plot_lines('LSpars (LSpr)', LSpr.detach().cpu().numpy())
        self.vis.plot_lines('L (tmp)', L.detach().cpu().numpy())