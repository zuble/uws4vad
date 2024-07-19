import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.pstfwd.utils import PstFwdUtils

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
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.crit = nn.BCELoss()
        
    def forward(self, scores, label, tag=None):
        log.debug(f"Bce/{scores.shape} {scores.device} {label.shape} {label.device}")
        #log.debug(f"Bce/{scores=} {label=}")
        return {
            f'bce-{tag}' if tag else 'bce': self.crit(scores, label)
            }

class Clas(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        self.crit = nn.BCELoss()    
        self._fx = {
            'topk': self.fwd_topk,
            'full': self.fwd_full
        }.get(_cfg.fx)
        self.k = _cfg.k
        self.per_crop = _cfg.per_crop
    
    def get_k(self, x, label):
        #if label == 0: 
        #    return 1 ##pel4vad
        if self.k == -1: 
            return int(torch.div(x, 16, rounding_mode='trunc')+1)
        else: 
            return min(x, self.k)
                
    def fwd_topk(self, scores, label, seqlen):
        #scores = scores.squeeze()
        vl_scores = torch.zeros(0).to(scores.device)  # tensor([])
        for i in range(scores.shape[0]): ## bs(*nc)
            tmp, _ = torch.topk(scores[i][:seqlen[i]], k=self.get_k(seqlen[i],label[i]), largest=True)
            tmp = torch.mean(tmp).view(1)
            vl_scores = torch.cat((vl_scores, tmp))
        return vl_scores
        
    def fwd_full(self, scores, label, seqlen):
        vl_scores = torch.zeros(0).to(scores.device)
        for i in range(scores.shape[0]): #.self.pfu.bs
            sl = int(seqlen[i])
            tmp = np.mean(scores[i, :sl])
            vl_scores = torch.cat((vl_scores, tmp))
        return vl_scores
    
    def forward(self, ndata, ldata):
        label = ldata['label'] ## bs
        seqlen = ldata['seqlen'] ## bs
        scores = ndata['scores'] 
        
        if self.per_crop : #and self.pfu.ncrops > 1
            seqlen = seqlen.repeat_interleave(self.pfu.ncrops)
            label = label.repeat_interleave(self.pfu.ncrops)
            log.warning(f"{seqlen.shape}  {label.shape}")
        else:
            scores = self.pfu.uncrop(scores, 'mean')   
        
        vl_scores = self._fx(scores, label, seqlen)
        #log.warning(vl_scores)
        #vl_scores = torch.sigmoid(vl_scores)
        #log.warning(vl_scores)
        l = self.crit(vl_scores, label)
        return {
            'clas': l
            }
        
        
class Ranking(nn.Module): 
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs//2
        
        self.lambda1 = _cfg.get("lambda12")[0]
        self.lambda2 = _cfg["lambda12"][1] 
        self.use_tcn = False
        
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

    def forward(self, ndata, ldata):
        labels = ldata['label']
        seqlen = ldata['seqlen']
        scores = ndata['scores']
        scores = self.pfu.uncrop(scores, 'mean')
        
        loss = torch.tensor(0., requires_grad=True, device=scores.device)
        
        abn_scores, nor_scores = self.pfu.unbag(scores, labels)
        for i in range(self.pfu.bat_div): ## bs//2
            maxn = torch.max( nor_scores[i][:seqlen[i]] ) 
            maxa = torch.max( abn_scores[i][:seqlen[i]] )
            #maxa = torch.mean( torch.topk( abn_scores[i][:seqlen[i]], k=self.pfu.seg_len//4)[0] )
            loss = loss + F.relu(1.0 - maxa + maxn)
            loss = loss + self.sparsity( abn_scores[i][:seqlen[i]] ) ## 
            loss = loss + self.smooth( abn_scores[i][:seqlen[i]] )
            
            ## TCN-IBL inner bag loss
            #mina = torch.min( abn_scores[i][:seqlen[i]] )
            #minn = torch.min( nor_scores[i][:seqlen[i]] )
            #loss = loss +  F.relu(1.0 - maxa + mina) ## loss_ibl
            #loss = loss + torch.abs(maxn - minn) ## loss_gap
            #loss = loss + loss_ibl + loss_gap
            
        #scores = scores.view(-1)
        #for i in range(self.pfu.bs//2):
        #    startn = (i * self.pfu.seg_len + (self.pfu.bs//2) * self.pfu.seg_len)
        #    endn = (i + 1) * self.pfu.seg_len + (self.pfu.bs//2) * self.pfu.seg_len
        #    maxn = torch.max( scores[ startn : endn ] ) 
        #    #maxn = torch.mean( torch.topk( scores[ startn : endn ], k=self.pfu.seg_len//4) )
        #
        #    starta = i * self.pfu.seg_len
        #    enda = (i + 1) * self.pfu.seg_len
        #    maxa = torch.max( scores[ starta : enda ] ) ##that
        #    #maxa = torch.mean( torch.topk( scores[ starta : enda ], k=self.pfu.seg_len//4) )
        #    
        #    tmp = F.relu(1.0 - maxa + maxn) 
        #    loss = loss + tmp
        #    loss = loss + self.sparsity(scores[ starta : enda ])
        #    loss = loss + self.smooth(scores[ starta : enda ])
        
        return {
            'rnkg_sult': loss / self.pfu.bat_div,
            #'rnkg_tcn': L_tcn
            }
    
            
class Normal(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.w_normal = _cfg.w #1.
        #self.w_normal = nn.Para
        
    def forward(self, ndata, ldata): ## supos batch-normed scores (normal_scores)

        scores = self.pfu.uncrop(ndata['norm_scors'], 'mean')
        scores_abn, scores_nor = self.pfu.unbag(scores)#ldata['labels']
        log.debug(f"{scores_nor.shape=}")
        
        ## bag_normal, t -> bag_normal
        #l = torch.norm(scores, dim=1, p=2).mean()
        l = torch.linalg.norm(scores_nor, ord=2, dim=1).mean()
        return {
            'norm': self.w_normal * l
            }


class MultiBranchSupervision(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs // 2
        self.per_crop = _cfg.per_crop
        
        self.eps = _cfg.eps
        self.alpha = _cfg.alpha
        self.gamma = _cfg.gamma
        self.mu = _cfg.mu
        self.M = _cfg.M
        
        self.bce = nn.BCELoss()
        self.cur_stp = 1
    
    def preproc(self, ndata, ldata):
        
        label = ldata['label'] 
        seqlen = ldata['seqlen']
        scores = ndata['scores']
        scores_att = ndata['attw']
        
        scores = self.pfu.uncrop(scores, 'mean')  
        scores_att = self.pfu.uncrop(scores_att, 'mean') 
        scores_abn, scores_nor = self.pfu.unbag(scores, label)
        attw_abn, attw_nor = self.pfu.unbag(scores_att, label)
        slen_abn, slen_nor = self.pfu.unbag(seqlen, label)
        
        #L = None
        LG_0 = None
        LG_1 = None
        LN_1 = None
        Lslclas = None
        LSmt = None
        LSpr = None
        
        bdiv = self.pfu.bat_div ##self.pfu.bs//2
        for bi in range( bdiv ): ## assumes bag=0.5
            log.debug(f"{'-'*10} mbs[{bi}/{bdiv}] {'-'*10}")

            # NORMAL
            SO_0 = scores_nor[bi][:slen_nor[bi]]
            Ai_0 = attw_nor[bi][:slen_nor[bi]]
            SA_0 = Ai_0 * SO_0
            log.debug(f'mbs[{bi}/{bdiv}]/NOR Scores : SO_0 {list(SO_0.shape)=}, Ai_0 {list(Ai_0.shape)}, SA_0 {list(SA_0.shape)}')

            thetai_0 = ((torch.max(Ai_0) - torch.min(Ai_0)) * self.eps) + torch.min(Ai_0)
            SSO_0 = SO_0 * (Ai_0 < thetai_0)
            SSA_0 = SA_0 * (Ai_0 < thetai_0)
            log.debug(f'mbs[{bi}/{bdiv}]/NOR Supressed: SSO_0 {SSO_0.shape}, SSA_0 {SSA_0.shape}')


            # ABNORMAL
            SO_1 = scores_abn[bi][:slen_abn[bi]]
            Ai_1 = attw_abn[bi][:slen_abn[bi]]
            SA_1 = Ai_1 * SO_1
            log.debug(f'mbs[{bi}/{bdiv}]/ABN Scores : SO_1 {list(SO_1.shape)=}, Ai_1 {list(Ai_1.shape)}, SA_1 {list(SA_1.shape)}')

            thetai_1 = ((torch.max(Ai_1) - torch.min(Ai_1)) * self.eps) + torch.min(Ai_1)
            SSO_1 = SO_1 * (Ai_1 < thetai_1)
            SSA_1 = SA_1 * (Ai_1 < thetai_1)
            log.debug(f'mbs[{bi}/{bdiv}]/ABN Supressed: SSO_1 {list(SSO_1.shape)}, SSA_1 {list(SSA_1.shape)}')


            ## Guide , eq(9,10)
            LG_0 = torch.mean((Ai_0 - 0) ** 2) ## MSE((Aneg, {0 · · · 0}))
            log.debug(f'mbs[{bi}/{bdiv}]/LossGuide: LG_0 {LG_0}')

            if self.cur_stp < self.M: lg1_tmp = SO_1
            else: lg1_tmp = (SO_1 > 0.5).float()
            LG_1 = torch.mean((Ai_1 - lg1_tmp) ** 2)
            log.debug(f'mbs[{bi}/{bdiv}]/LossGuide: LG_1 {LG_1}')
                
                
            ## Loss L1-Norm , eq(11)
            tmp_ln1 = torch.sum(torch.abs(Ai_1))
            log.debug(f'mbs[{bi}/{bdiv}]/LossNorm: LN_1 {tmp_ln1}')
            if LN_1 is None: LN_1 = tmp_ln1.unsqueeze(0)
            else: LN_1 = torch.cat((LN_1, tmp_ln1.unsqueeze(0)), dim=0)


            ## Loss SL scores
            zeros = torch.zeros((1, slen_nor[bi]), device=self.pfu.dvc)
            ones = torch.ones((1, slen_abn[bi]), device=self.pfu.dvc)
            ## Raw Org/Att , eq(12)
            LC_SO = self.bce(SO_0.unsqueeze(0), zeros) + self.bce(SO_1.unsqueeze(0), ones)
            LC_SA = self.bce(SA_0.unsqueeze(0), zeros) + self.bce(SA_1.unsqueeze(0), ones)
            ## Supressed Org/Att , eq(12)
            LC_SSO = self.bce(SSO_0.unsqueeze(0), zeros) + self.bce(SSO_1.unsqueeze(0), ones)
            LC_SSA = self.bce(SSA_0.unsqueeze(0), zeros) + self.bce(SSA_1.unsqueeze(0), ones)

            tmp_slclas = self.alpha * (LC_SO + LC_SA) + (1 - self.alpha) * (LC_SSO + LC_SSA) ## eq(13)
            log.debug(f'mbs[{bi}/{bdiv}]/LossCls: LC_SO {LC_SO}, LC_SA {LC_SA}, LC_SSO {LC_SSO}, LC_SSA {LC_SSA}, LC_ALL {tmp_slclas}')
            
            if Lslclas is None: Lslclas = tmp_slclas.unsqueeze(0)
            else: Lslclas = torch.cat((Lslclas, tmp_slclas.unsqueeze(0)), dim=0)
            
            
            ## Loss Smoothness , eq(14)
            LSmt_SOrg = torch.sum((SO_0[:-1] - SO_0[1:]) ** 2) + torch.sum((SO_1[:-1] - SO_1[1:]) ** 2)
            LSmt_SAtt = torch.sum((SA_0[:-1] - SA_0[1:]) ** 2) + torch.sum((SA_1[:-1] - SA_1[1:]) ** 2)
            tmp_smt = LSmt_SOrg + LSmt_SAtt
            log.debug(f'mbs[{bi}/{bdiv}]/LossSmooth: SO {LSmt_SOrg}, SA {LSmt_SAtt}, SUM {tmp_smt}')
            
            if LSmt is None: LSmt = tmp_smt.unsqueeze(0)
            else: LSmt = torch.cat((LSmt, tmp_smt.unsqueeze(0)), dim=0)
            
            
            ## Loss Sparsity , eq(14)
            LSpr_SOrg = torch.sum(SO_0) + torch.sum(SO_1)
            LSpr_SAtt = torch.sum(SA_0) + torch.sum(SA_1)
            tmp_spr = LSpr_SOrg + LSpr_SAtt
            log.debug(f'mbs[{bi}/{bdiv}]/LossSparse: SO {LSpr_SOrg}, SA {LSpr_SAtt}, SUM {tmp_spr}')

            if LSpr is None: LSpr = tmp_spr.unsqueeze(0)
            else: LSpr = torch.cat((LSpr, tmp_spr.unsqueeze(0)), dim=0)
            
            
        #L = LC + self.gamma * LN_1 + LG_0 + LG_1 + self.mu * LSmt + LSpr  ## eq(15)
        #if (self.cur_stp % dbg_stp) == 0:
        #log.debug(f'mbs[{bi}/{bdiv}]/LLLL: {L}')

        self.cur_stp += 1
        return {
            'slclas': Lslclas.mean(),
            'norm_1': (self.gamma * LN_1).mean(),
            'guide_0': LG_0.mean(),
            'guide_1': LG_1.mean(),
            'smooth': (self.mu * LSmt).mean(), ## Weight before averaging
            'sparse': LSpr.mean()
            #'mbs': L.mean()
            }
        
    def forward(self, ndata, ldata):
        return self.preproc(ndata,ldata)
        
        '''log.debug(f"mbs [{self.cur_stp}]")
        if self.cur_stp == self.M:
            log.warning("LG_1 (L pos guide) calc changing")

        label = ldata['label'] 
        seqlen = ldata['seqlen']
        scores = ndata['scores']
        scores_att = ndata['attw']
        
        # Unbag scores and attention weights BEFORE uncropping
        
        
        if self.per_crop : #and self.pfu.ncrops > 1
            raise NotImplementedError
            seqlen = seqlen.repeat_interleave(self.pfu.ncrops)
            label = label.repeat_interleave(self.pfu.ncrops)
            log.warning(f"{seqlen.shape}  {label.shape}")
        else: 
            scores = self.pfu.uncrop(scores, 'mean')  
            scores_att = self.pfu.uncrop(scores_att, 'mean') 
            scores_abn, scores_nor = self.pfu.unbag(scores, label)
            attw_abn, attw_nor = self.pfu.unbag(scores_att, label)
        
        
        #num_abn = scores_abn.shape[0]
        #num_nor = scores_nor.shape[0]
        
        
        dbg_stp = 1
        L = None
        bdiv = self.pfu.bat_div ##self.pfu.bs//2
        for bi in range( bdiv ): ## assumes bag=0.5
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f"mbs[{bi}/{bdiv}]")

            # NORMAL
            slen_nor = seqlen[bdiv + bi]
            SO_0 = scores[bdiv + bi][:slen_nor]
            Ai_0 = scores_att[bdiv + bi][:slen_nor]
            
            SA_0 = Ai_0 * SO_0
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/Scores: SO_0 {SO_0.shape}, Ai_0 {Ai_0.shape}, SA_0 {SA_0.shape}')

            thetai_0 = ((torch.max(Ai_0) - torch.min(Ai_0)) * self.eps) + torch.min(Ai_0)
            SSO_0 = SO_0 * (Ai_0 < thetai_0)
            SSA_0 = SA_0 * (Ai_0 < thetai_0)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/SupressedScores: SSO_0 {SSO_0.shape}, SSA_0 {SSA_0.shape}')


            # ABNORMAL
            slen_abn = seqlen[bdiv]
            SO_1 = scores[bdiv][:slen_abn]
            Ai_1 = scores_att[bdiv][:slen_abn]
            SA_1 = Ai_1 * SO_1
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/Scores: SO_1 {SO_1.shape}, Ai_1 {Ai_1.shape}, SA_1 {SA_1.shape}')

            thetai_1 = ((torch.max(Ai_1) - torch.min(Ai_1)) * self.eps) + torch.min(Ai_1)
            SSO_1 = SO_1 * (Ai_1 < thetai_1)
            SSA_1 = SA_1 * (Ai_1 < thetai_1)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/SupressedScores: SSO_1 {SSO_1.shape}, SSA_1 {SSA_1.shape}')

            ## Guide , eq(9,10)
            LG_0 = torch.mean((Ai_0 - 0) ** 2) ## MSE((Aneg, {0 · · · 0}))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LossGuide: LG_0 {LG_0}')

            if self.cur_stp < self.M: lg1_tmp = SO_1
            else: lg1_tmp = (SO_1 > 0.5).float()
            LG_1 = torch.mean((Ai_1 - lg1_tmp) ** 2)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LossGuide: LG_1 {LG_1}')
                
            ## Loss L1-Norm , eq(11)
            LN_1 = torch.sum(torch.abs(Ai_1))
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LossNorm: LN_1 {LN_1}')

            ## Loss SL scores
            zeros = torch.zeros((1, slen_nor), device=self.pfu.dvc)
            ones = torch.ones((1, slen_abn), device=self.pfu.dvc)
            ## Raw Org/Att , eq(12)
            LC_SO = self.bce(SO_0.unsqueeze(0), zeros) + self.bce(SO_1.unsqueeze(0), ones)
            LC_SA = self.bce(SA_0.unsqueeze(0), zeros) + self.bce(SA_1.unsqueeze(0), ones)
            ## Supressed Org/Att , eq(12)
            LC_SSO = self.bce(SSO_0.unsqueeze(0), zeros) + self.bce(SSO_1.unsqueeze(0), ones)
            LC_SSA = self.bce(SSA_0.unsqueeze(0), zeros) + self.bce(SSA_1.unsqueeze(0), ones)

            LC = self.alpha * (LC_SO + LC_SA) + (1 - self.alpha) * (LC_SSO + LC_SSA) ## eq(13)
            if (self.cur_stp % dbg_stp) == 0: 
                log.debug(f'mbs[{bi}/{bdiv}]/LossCls: LC_SO {LC_SO}, LC_SA {LC_SA}, LC_SSO {LC_SSO}, LC_SSA {LC_SSA}, LC_ALL {LC}')
            
            ## Loss Smoothness , eq(14)
            LSmt_SOrg = torch.sum((SO_0[:-1] - SO_0[1:]) ** 2) + torch.sum((SO_1[:-1] - SO_1[1:]) ** 2)
            LSmt_SAtt = torch.sum((SA_0[:-1] - SA_0[1:]) ** 2) + torch.sum((SA_1[:-1] - SA_1[1:]) ** 2)
            LSmt = LSmt_SOrg + LSmt_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LossSmooth: SO {LSmt_SOrg}, SA {LSmt_SAtt}, SUM {LSmt}')
            
            ## Loss Sparsity , eq(14)
            LSpr_SOrg = torch.sum(SO_0) + torch.sum(SO_1)
            LSpr_SAtt = torch.sum(SA_0) + torch.sum(SA_1)
            LSpr = LSpr_SOrg + LSpr_SAtt
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LossSparse: SO {LSpr_SOrg}, SA {LSpr_SAtt}, SUM {LSpr}')

            tmp = LC + self.gamma * LN_1 + LG_0 + LG_1 + self.mu * LSmt + LSpr  ## eq(15)

            if L is None: L = tmp.unsqueeze(0)
            else: L = torch.cat((L, tmp.unsqueeze(0)), dim=0)
            if (self.cur_stp % dbg_stp) == 0:
                log.debug(f'mbs[{bi}/{bdiv}]/LLLL: {L}')

            
        self.cur_stp += 1
        return {
            'mbs': L.mean()
            }
'''