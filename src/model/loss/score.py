import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance, BatchedDistance, CosineSimilarity

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
        self.pfu = pfu
        self.crit = nn.BCELoss()
        
    def forward(self, ndata, ldata):
        #log.debug(f"Bce/{scores.shape} {scores.device} {label.shape} {label.device}")
        return {
            'bce': self.crit(ndata['vlscores'],ldata["label"])
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
        if self.per_crop: assert self.pfu.ncrops
    
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
        for i in range(scores.shape[0]): ## bs,t  in original is cropasvideo ? 
            tmp, _ = torch.topk(scores[i][:seqlen[i]], k=self.get_k(seqlen[i],label[i]), largest=True)
            tmp = torch.mean(tmp).view(1)
            vl_scores = torch.cat((vl_scores, tmp))
            log.debug(f"{i} LBL:{label[i]} SEL:{tmp[0]} {tmp.shape}")
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
        
        #seqlen = seqlen.repeat_interleave(self.pfu.ncrops)
        #label = label.repeat_interleave(self.pfu.ncrops)
        #log.warning(f"{seqlen.shape}  {label.shape} {scores.shape}")
        
        scores = self.pfu.uncrop(scores, 'mean')
        if self.pfu.cropasvideo: 
            assert scores.shape[0] == self.pfu.bs
        #log.error(scores.shape)
        
        vl_scores = self._fx(scores, label, seqlen)
        #log.warning(vl_scores)
        #vl_scores = torch.sigmoid(vl_scores) # <- inside fwd
        #log.warning(vl_scores)
        L = self.crit(vl_scores, label)
        
        return {
            'clas': L
            ## retrieve additional info with certain key
            ## and if in debug send to vis, eg:
            ## '_sel_scores': [vl_scores,label]
            }
        
        
class Ranking(nn.Module): 
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs//2
        
        self.lambda1 = _cfg.get("lambda12")[0]
        self.lambda2 = _cfg["lambda12"][1] 
        self.use_tcn = False
        
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
        for i in range(self.pfu.bat_div): 
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
            
        return {
            'rnkg_sult': loss / self.pfu.bat_div,
            #'rnkg_tcn': L_tcn
            }
    
            
class Normal(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs//2
        self.w_normal = _cfg.w
        #self.w_normal = nn.Para
        self.crit = nn.BCELoss()
        self.k=-1
        
    def get_k(self, x, label):
        if label == 0: return 1 ##pel4vad
        if self.k == -1: return int(torch.div(x, 16, rounding_mode='trunc')+1)
        else: return min(x, self.k)
    def fwd_topk(self, scores, label, seqlen):
        #scores = scores.squeeze()
        vl_scores = torch.zeros(0).to(scores.device)  # tensor([])
        for i in range(scores.shape[0]): ## bs,t  in original is cropasvideo ? 
            tmp, _ = torch.topk(scores[i][:seqlen[i]], k=self.get_k(seqlen[i],label[i]), largest=True)
            tmp = torch.mean(tmp).view(1)
            vl_scores = torch.cat((vl_scores, tmp))
            log.debug(f"{i} LBL:{label[i]} SEL:{tmp[0]} {tmp.shape}")
        return vl_scores
        
    def forward(self, ndata, ldata): ## supos batch-normed scores (normal_scores)
        label = ldata['label'] ## bs
        seqlen = ldata['seqlen'] ## bs
        scores = ndata['norm_scors'] 
        
        scores = self.pfu.uncrop(scores, 'mean')
        scores_abn, scores_nor = self.pfu.unbag(scores)#ldata['labels']
        log.debug(f"{scores_nor.shape=}")
        
        ## -------------
        #label_abn, label_nor = self.pfu.unbag(label)
        #seqlen_abn, seqlen_nor = self.pfu.unbag(seqlen)
        #self.pfu.is_equal(seqlen[self.pfu.bat_div:],seqlen_nor)
        #self.pfu.is_equal(label[self.pfu.bat_div:],label_nor)
        #vls = self.fwd_topk(scores,label,seqlen)
        #L = self.crit(vls,label)
        #vls = self.fwd_topk(scores_nor,label_nor,seqlen_nor)
        #L = self.crit(vls,label_nor)
        #vls = self.fwd_topk(scores_abn,label_abn,seqlen_abn)
        #L = self.crit(vls,label_abn)
        ##-------------
        
        ## bag_normal, t -> bag_normal
        #L = torch.norm(scores, dim=1, p=2).mean() <- oldone
        L = torch.linalg.norm(scores_nor, ord=2, dim=1).mean()
        log.debug(f'{L=}')
        return {
            'norm': L * self.w_normal 
            }


class Salient(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        self.bce = nn.BCELoss()
        self.per_crop = _cfg.per_crop
        self.margin = _cfg.trip_margin
        
        #distance = distances.CosineSimilarity()
        distance= distances.LpDistance(normalize_embeddings=False, p=1)
        #distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        #distance= distances.LpDistance(power=2)
        
        #reducer = reducers.ThresholdReducer(low=0)
        self.mining_func = miners.TripletMarginMiner(
            margin=self.margin, distance=distance, type_of_triplets="semihard"
        )
        self.triplet = losses.TripletMarginLoss(margin=self.margin, distance=distance, ) #reducer=reducer
        
    def forward(self, ndata, ldata):
        #########
        ## VL SCORE
        if self.per_crop:
            vls = ndata['vls'] ## b*nc
            label = ldata['label'].repeat_interleave(self.pfu.ncrops)
        else:
            vls = self.pfu.uncrop(ndata['vls'], 'mean') #
            label = ldata["label"]
        #log.debug(f"{label.shape}")
        loss_vid = self.bce(vls, label) 
        return {'salient': loss_vid}
        '''
        #######
        ## ATTW 
        bsnc, r, t = ndata['attw'].shape
        assert bsnc == self.pfu.ncrops * self.pfu.bs
        log.error(f"{bsnc} {r} {t}")
        
        #label = ldata['label'].repeat_interleave(self.pfu.ncrops*r)
        #attw = ndata["attw"].reshape(-1, t) ## bsnc*r, t 
        #log.error(f"{attw.shape=} {label.shape=}")
        
        attw = self.pfu.uncrop(ndata['attw'], '') ## b, nc, r, t
        attw = attw.reshape(self.pfu.bs, -1, t) ## b, nc*r, t
        #attw = self.pfu.uncrop(ndata['attw'], 'mean') ## b, r, t
        #attw = attw.mean(1) ## b, t
        log.debug(f"{attw.shape}  ")
        
        ## normal
        nor_attw = attw[ ldata['label'] == 0 ]
        loss_guid0 = torch.mean((nor_attw - 0) ** 2)
        log.debug(f"{nor_attw.shape} {loss_guid0}")    
        
        ## abnormal
        #if self.cur_stp < self.M: lg1_tmp = SO_1
        #else: lg1_tmp = (SO_1 > 0.5).float()
        #L1 = torch.mean((abn_attw - lg1_tmp) ** 2)
        #log.debug(f'LG_1 {LG_1}')
        
        ## Loss L1-Norm , 2 polarize
        abn_attw = attw[ ldata['label'] != 0 ] ## bag, ..
        loss_norm1 = torch.sum(torch.abs(abn_attw))  
        log.debug(f"{abn_attw.shape} {loss_norm1}") 
        
        
        #######
        ## VL FEAT
        bsnc, r, f = ndata['vlf'].shape
        assert bsnc == self.pfu.ncrops * self.pfu.bs
        log.error(f"{bsnc} {r} {f}")
        
        label = ldata['label'].repeat_interleave(self.pfu.ncrops*r)
        vlf = ndata["vlf"].reshape(-1, f) ## bsnc*r, f 
        log.error(f"{vlf.shape=} {label.shape=}")
        
        ## it gets stuck at either margin value or half of it
        indices_tuple = self.mining_func(vlf, label)
        loss_trip = self.triplet(vlf, label, indices_tuple) #
        log.info(f"l3 {loss_trip}  {len(indices_tuple)=}") 
        
        ## not testes
        #vlf = self.pfu.uncrop(ndata['vlf'], '') ## b, nc, r, f
        #vlf = ndata["vlf"].mean(dim=1) ## b, f
        #vlf = ndata["vlf"].reshape(self.pfu.bs, -1) ## b, r*f 
        #log.info(f"vlf {vlf.shape}") 
        
        #abn_vlf, nor_vlf = self.pfu.unbag(vlf, ldata['label'])
        #log.info(f"vlf {abn_vlf.shape}{nor_vlf.shape}")
        
        #nor_mask = ldata['label'] == 0
        #abn_mask = ldata['label'] != 0
        #log.warning(f"{nor_mask.shape}  {abn_mask.shape}")
        #sel_nor_vlf = vlf[ nor_mask.repeat_interleave(self.pfu.ncrops) ]
        #sel_abn_vlf = vlf[ abn_mask.repeat_interleave(self.pfu.ncrops) ]
        #log.warning(f"{sel_nor_vlf.shape}  {sel_abn_vlf.shape}")
        
        return {
            'video': loss_vid,
            'guide0': loss_guid0,
            'norm1': loss_norm1 *0.1 , #
            'triple': loss_trip
            }
        '''
        
        '''
        def collect_fn(all_mat):
            def fn(mat, *_):
                all_mat.append(mat)
            return fn
        
        mat = []
        distance = BatchedDistance(CosineSimilarity(), collect_fn(mat), self.bs)
        
        #distance(embeddings, ref)
        distance(vlf)
        mat = torch.cat(mat, dim=0)

        log.info(f"{mat.shape} {mat}") 
        '''

        '''
        nor_euc_dist = torch.norm(vlf[nor_mask], p=2, dim=1)
        abn_euc_dist = torch.abs(100 - torch.norm(vlf[abn_mask], p=2, dim=1))
        log.debug(f"{nor_euc_dist.shape}  {abn_euc_dist.shape}")
        
        L1 = torch.mean( (nor_euc_dist + abn_euc_dist) ** 2)
        log.debug(f"{L1}")    
        
        
        L1 = None
        for bi in range( vlf.shape[0] ):
            
            if ldata['label'][bi]: 
                tmp = torch.abs(100 - torch.norm(vlf[bi], p=2, dim=1) )
                
            else:  
                tmp = torch.norm(vlf[bi], p=2, dim=1)
                
            #loss_rtfm = torch.mean((loss_abn + loss_norm) ** 2)
            log.debug(f"{tmp.shape} {tmp}")
            if L1 is None: L1 = tmp 
            else: L1 = torch.cat((L, tmp), dim=0)   
            
        log.debug(f"{L1}")    
        L1 = L1.mean( L1.sum() ** 2)
        log.debug(f"{L1}") 
        
        return super().merge(L0, L3) #, L1, L2
        '''



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
        LC_SO = None
        LC_SA = None
        LC_SSO = None
        LC_SSA = None
        LSmt = None
        LSpr = None
        
        bdiv = self.pfu.bat_div ##self.pfu.bs//2
        for bi in range( bdiv ): ## assumes bag=0.5
            log.debug(f"{'-'*10} mbs[{bi}/{bdiv}] {'-'*10}")
            
            ## preproc
            ## norm
            SO_0 = scores_nor[bi][:slen_nor[bi]]
            Ai_0 = attw_nor[bi][:slen_nor[bi]]
            SA_0 = Ai_0 * SO_0
            log.debug(f'mbs[{bi}/{bdiv}]/NOR Scores : SO_0 {list(SO_0.shape)=}, Ai_0 {list(Ai_0.shape)}, SA_0 {list(SA_0.shape)}')
            
            thetai_0 = ((torch.max(Ai_0) - torch.min(Ai_0)) * self.eps) + torch.min(Ai_0)
            SSO_0 = SO_0 * (Ai_0 < thetai_0)
            SSA_0 = SA_0 * (Ai_0 < thetai_0)
            log.debug(f'mbs[{bi}/{bdiv}]/NOR Supressed: SSO_0 {SSO_0.shape}, SSA_0 {SSA_0.shape}')
            
            ## abnormal
            SO_1 = scores_abn[bi][:slen_abn[bi]]
            Ai_1 = attw_abn[bi][:slen_abn[bi]]
            SA_1 = Ai_1 * SO_1
            log.debug(f'mbs[{bi}/{bdiv}]/ABN Scores : SO_1 {list(SO_1.shape)=}, Ai_1 {list(Ai_1.shape)}, SA_1 {list(SA_1.shape)}')
            
            thetai_1 = ((torch.max(Ai_1) - torch.min(Ai_1)) * self.eps) + torch.min(Ai_1)
            SSO_1 = SO_1 * (Ai_1 < thetai_1)
            SSA_1 = SA_1 * (Ai_1 < thetai_1)
            log.debug(f'mbs[{bi}/{bdiv}]/ABN Supressed: SSO_1 {list(SSO_1.shape)}, SSA_1 {list(SSA_1.shape)}')
            
            
            ###################
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
            
            
            ################
            ## Loss SL scores
            zeros = torch.zeros((1, slen_nor[bi]), device=self.pfu.dvc)
            ones = torch.ones((1, slen_abn[bi]), device=self.pfu.dvc)
            
            ## Raw Org/Att , eq(12)
            tmp_lcso = self.bce(SO_0.unsqueeze(0), zeros) + self.bce(SO_1.unsqueeze(0), ones)
            if LC_SO is None: LC_SO = tmp_lcso.unsqueeze(0)
            else: LC_SO = torch.cat((LC_SO, tmp_lcso.unsqueeze(0)), dim=0)
            
            tmp_lcsa = self.bce(SA_0.unsqueeze(0), zeros) + self.bce(SA_1.unsqueeze(0), ones)
            if LC_SA is None: LC_SA = tmp_lcsa.unsqueeze(0)
            else: LC_SA = torch.cat((LC_SA, tmp_lcsa.unsqueeze(0)), dim=0)
            
            ## Supressed Org/Att , eq(12)
            tmp_lcsso = self.bce(SSO_0.unsqueeze(0), zeros) + self.bce(SSO_1.unsqueeze(0), ones)
            if LC_SSO is None: LC_SSO = tmp_lcsso.unsqueeze(0)
            else: LC_SSO = torch.cat((LC_SSO, tmp_lcsso.unsqueeze(0)), dim=0)
            
            tmp_lcssa = self.bce(SSA_0.unsqueeze(0), zeros) + self.bce(SSA_1.unsqueeze(0), ones)
            if LC_SSA is None: LC_SSA = tmp_lcssa.unsqueeze(0)
            else: LC_SSA = torch.cat((LC_SSA, tmp_lcssa.unsqueeze(0)), dim=0)
            
            #tmp_slclas = self.alpha * (LC_SO + LC_SA) + (1 - self.alpha) * (LC_SSO + LC_SSA) ## eq(13)
            #log.debug(f'mbs[{bi}/{bdiv}]/LossCls: LC_SO {LC_SO}, LC_SA {LC_SA}, LC_SSO {LC_SSO}, LC_SSA {LC_SSA}, LC_ALL {tmp_slclas}')
            #
            #if Lslclas is None: Lslclas = tmp_slclas.unsqueeze(0)
            #else: Lslclas = torch.cat((Lslclas, tmp_slclas.unsqueeze(0)), dim=0)
            
            
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
            #'slclas': Lslclas.mean(),
            'slclas_og': (self.alpha * LC_SO).mean(),
            'slclas_att': (self.alpha * LC_SA).mean(),
            'slclas_supog': ( (1-self.alpha) * LC_SSO).mean(),
            'slclas_supatt': ( (1-self.alpha) * LC_SSA).mean(),    
            'norm_1': (self.gamma * LN_1).mean(),
            'guide_0': LG_0.mean(),
            'guide_1': LG_1.mean(),
            'smooth': (self.mu * LSmt).mean(), ## Weight before averaging
            'sparse': LSpr.mean()
            #'mbs': L.mean()
            }
        
    def forward(self, ndata, ldata):
        return self.preproc(ndata,ldata)
        
        '''
        log.debug(f"mbs [{self.cur_stp}]")
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