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
        #self.crit = nn.BCEWithLogitsLoss()
        
    def forward(self, ndata, ldata):
        #log.debug(f"Bce/{scores.shape} {scores.device} {label.shape} {label.device}")
        return {
            'bce': self.crit(ndata['vlscores'],ldata["label"])
        }

class Clas(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        self._fx = {
            'topk': self.fwd_topk,
            'full': self.fwd_full
        }.get(_cfg.fx)
        self.k = _cfg.k
        self.per_crop = _cfg.per_crop
        if self.per_crop: assert self.pfu.ncrops

        #self.crit = nn.BCELoss()
        self.crit = nn.BCEWithLogitsLoss()
    
    def get_k(self, x, label):
        if label == 0:  return 1 ##pel4vad
        elif self.k == -1: return int(torch.div(x, 16, rounding_mode='trunc')+1)
        else: return min(x, self.k)
                
    def fwd_topk(self, scores, label, seqlen):
        #scores = scores.squeeze()
        vl_scores = torch.zeros(0).to(scores.device)#self.pfu.dvc 
        for i in range(scores.shape[0]): ## bs,t 
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
        vl_scores = self._fx(scores, label, seqlen)
        #vl_scores2 = torch.sigmoid(vl_scores)
        #log.warning(f"{vl_scores}  {vl_scores2}")
        
        L = self.crit(vl_scores, label)
        
        return {
            'clas': L
            # TODO: define standard keys for normal and debug
        }
        
        
class Ranking(nn.Module): 
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs//2
        
        self.lambda1 = _cfg.get("lambda12")[0]
        self.lambda2 = _cfg["lambda12"][1] 
        self.use_tcn = False
        
        self.sig = nn.Sigmoid()
        
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
        
        scores = self.sig(scores)
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
        
        self.crit = nn.BCELoss()
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
        loss_vid = self.crit(vls, label) 
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
        
        self.crit = nn.BCELoss()
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
            tmp_lcso = self.crit(SO_0.unsqueeze(0), zeros) + self.crit(SO_1.unsqueeze(0), ones)
            if LC_SO is None: LC_SO = tmp_lcso.unsqueeze(0)
            else: LC_SO = torch.cat((LC_SO, tmp_lcso.unsqueeze(0)), dim=0)
            
            tmp_lcsa = self.crit(SA_0.unsqueeze(0), zeros) + self.crit(SA_1.unsqueeze(0), ones)
            if LC_SA is None: LC_SA = tmp_lcsa.unsqueeze(0)
            else: LC_SA = torch.cat((LC_SA, tmp_lcsa.unsqueeze(0)), dim=0)
            
            ## Supressed Org/Att , eq(12)
            tmp_lcsso = self.crit(SSO_0.unsqueeze(0), zeros) + self.crit(SSO_1.unsqueeze(0), ones)
            if LC_SSO is None: LC_SSO = tmp_lcsso.unsqueeze(0)
            else: LC_SSO = torch.cat((LC_SSO, tmp_lcsso.unsqueeze(0)), dim=0)
            
            tmp_lcssa = self.crit(SSA_0.unsqueeze(0), zeros) + self.crit(SSA_1.unsqueeze(0), ones)
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
            LC_SO = self.crit(SO_0.unsqueeze(0), zeros) + self.crit(SO_1.unsqueeze(0), ones)
            LC_SA = self.crit(SA_0.unsqueeze(0), zeros) + self.crit(SA_1.unsqueeze(0), ones)
            ## Supressed Org/Att , eq(12)
            LC_SSO = self.crit(SSO_0.unsqueeze(0), zeros) + self.crit(SSO_1.unsqueeze(0), ones)
            LC_SSA = self.crit(SSA_0.unsqueeze(0), zeros) + self.crit(SSA_1.unsqueeze(0), ones)
            
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


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x): return x.transpose(-2, -1)

def normalize(*xs): return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class CIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = InfoNCE(negative_mode='unpaired')

    def get_new_rep(self, sls, seqlen, rep, lrgst=False, mean=False):
        ## sls: t
        ## seqlen: 1
        ## rep: t, 128
        k_idxs = torch.topk(sls[:seqlen], k=int(seqlen // 16 + 1), largest=lrgst)[1]
        rep_new = rep[k_idxs] 
        if mean:
            print(f"{rep_new.size()=}")
            rep_new = torch.mean(rep_new, 0, keepdim=True).expand( rep_new.size() )
        print(f"{rep_new.shape=}\n")    
        return rep_new
    
    def gather_semi_bags(self, mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
        a_abn = torch.zeros(0)#.cuda()  # tensor([])
        v_abn = torch.zeros(0)#.cuda()  # tensor([])
        a_bgd = torch.zeros(0)#.cuda()  # tensor([])
        v_bgd = torch.zeros(0)#.cuda()  # tensor([])
        a_norm = torch.zeros(0)#.cuda()
        v_norm = torch.zeros(0)#.cuda()
        
        for i in range(a_sls.shape[0]): ## b
            
            ########################
            ## VISUAL BOTTOM/BCKGRND
            v_rep_down = self.get_new_rep(v_sls[i], seqlen[i], v_rep[i], lrgst=False, mean=False) ## k, 128
            v_bgd = torch.cat((v_bgd, v_rep_down), 0) 
            
            #######################
            ## AUDIO BOTTOM/BCKGRND            
            a_rep_down = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False, mean=False)
            a_bgd = torch.cat((a_bgd, a_rep_down), 0)
            
            ########################
            ## UPPER/FRGND
            if mil_vls[i] > 0.5:
                ## In contrast, we conduct average pooling to embeddings 
                ## of all violence instances in each bag and form a semi-bag-level representation
                ## By doing so, the aud and vis repre both express event-level semantics, 
                ## thereby alleviating the noise issue. 
                ## To this end, we construct semi-bag-level positive pairs, 
                ## which are assembled by audio and visual violent semi-bag representations B𝑣𝑖𝑜 𝑎 , B𝑣𝑖𝑜
                
                ## VISUAL 
                v_rep_top = self.get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=True)
                v_abn = torch.cat((v_abn, v_rep_top), 0)
                
                ## AUDIO UPPER/FRGND
                a_rep_top = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=True, mean=True)
                a_abn = torch.cat((a_abn, a_rep_top), 0)

            else:
                ## VISUAL 
                v_rep_top = self.get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=False) ## k, 128
                v_norm = torch.cat((v_norm, v_rep_top), 0) 
                
                ## AUDIO 
                a_rep_top = self.get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=True, mean=False)
                a_norm = torch.cat((a_norm, a_rep_top), 0)
        
        return a_abn, v_abn, a_norm, v_norm, a_bgd, v_bgd                    
    
    
    def forward(self, mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
        a_abn, v_abn, \
        a_norm, v_norm, \
        a_bgd, v_bgd = self.gather_semi_bags(mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep)
        
            
        #self.crit = InfoNCE(negative_mode='unpaired')
        if a_norm.size(0) == 0 or a_abn.size(0) == 0:
            return {
                "loss_a2v_abn2bgd": 0.0, 
                "loss_a2v_abn2nor": 0.0, 
                "loss_v2a_abn2bgd": 0.0, 
                "loss_v2a_abn2nor": 0.0
            }
        else:
            ## query, positive_key, negative_keys
            loss_a2v_abn2bgd = self.crit(a_abn, v_abn, v_bgd)
            loss_a2v_abn2nor = self.crit(a_abn, v_abn, v_norm)
            loss_v2a_abn2bgd = self.crit(v_abn, a_abn, a_bgd)
            loss_v2a_abn2nor = self.crit(v_abn, a_abn, a_norm)
            return {
                "loss_a2v_abn2bgd": loss_a2v_abn2bgd, 
                "loss_a2v_abn2nor": loss_a2v_abn2nor, 
                "loss_v2a_abn2bgd": loss_v2a_abn2bgd, 
                "loss_v2a_abn2nor": loss_v2a_abn2nor
            }
'''            
    b = 2
    t = 200
    hdim = 128
    mil_vls = torch.tensor([0.6, 0.2])
    a_sls = torch.randn((b,t))
    v_sls = torch.randn((b,t))
    seqlen = [32, 32]
    a_rep = torch.randn((b,t,hdim))
    v_rep = torch.randn((b,t,hdim))

    lossfx = CMA_MIL()
    loss = lossfx(mil_vls, a_sls, v_sls, seqlen, a_rep, v_rep)
    for k, v in loss.items():
        print(k, v)
        

    def CMAL(mil_vls, a_sls, v_sls, seq_len, a_rep, v_rep):
        a_abn = torch.zeros(0)#.cuda()  # tensor([])
        v_abn = torch.zeros(0)#.cuda()  # tensor([])
        a_bgd = torch.zeros(0)#.cuda()  # tensor([])
        v_bgd = torch.zeros(0)#.cuda()  # tensor([])
        a_norm = torch.zeros(0)#.cuda()
        v_norm = torch.zeros(0)#.cuda()
        
        for i in range(a_sls.shape[0]): ## b
            if mil_vls[i] > 0.5:
                
                ########################
                ## VISUAL BOTTOM/BCKGRND
                v_sls_down_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
                v_rep_down = v_rep[i][v_sls_down_idxs] 
                print(f"{v_rep_down.shape=}")
                
                v_rep_down2 = get_new_rep(v_sls[i], seqlen[i], v_rep[i], lrgst=False, mean=False)
                assert v_rep_down.shape == v_rep_down2.shape
                
                v_bgd = torch.cat((v_bgd, v_rep_down), 0)

                ## VISUAL UPPER/FRGND
                v_sls_top_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
                v_rep_top = v_rep[i][v_sls_top_idxs]
                v_rep_top = torch.mean(v_rep_top, 0, keepdim=True).expand( v_rep_top.size() )
                
                v_rep_top2 = get_new_rep(v_sls[i], seq_len[i], v_rep[i], lrgst=True, mean=True)
                assert v_rep_top.shape == v_rep_top2.shape
                
                v_abn = torch.cat((v_abn, v_rep_top), 0)
                
                
                #######################
                ## AUDIO BOTTOM/BCKGRND            
                a_sls_down_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
                a_rep_down = a_rep[i][a_sls_down_idxs]
                #a_rep_down = torch.mean(a_rep_down, 0, keepdim=True).expand( a_rep_down.size() )
                
                a_rep_down2 = get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False)
                assert v_rep_top.shape == v_rep_top2.shape
                
                a_bgd = torch.cat((a_bgd, a_rep_down), 0)

                ## AUDIO UPPER/FRGND
                a_sls_top_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
                a_rep_top = a_rep[i][a_sls_top_idxs]
                a_rep_top = torch.mean(a_rep_top, 0, keepdim=True).expand( a_rep_top.size() )
                
                a_rep_top2 = get_new_rep(a_sls[i], seq_len[i], a_rep[i], lrgst=False, mean=True)
                assert a_rep_top.shape == a_rep_top2.shape
                
                a_abn = torch.cat((a_abn, a_rep_top), 0)
                
                
            else:
                ########################
                ## VISUAL BOTTOM/BCKGRND
                v_sls_down_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
                v_rep_down = v_rep[i][v_sls_down_idxs]
                # v_rep_down = torch.mean(v_rep_down, 0, keepdim=True).expand( v_rep_down.size() )
                v_bgd = torch.cat((v_bgd, v_rep_down), 0)

                ## VISUAL UPPER/FRGND
                v_sls_top_idxs = torch.topk(v_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
                v_rep_top = v_rep[i][v_sls_top_idxs]
                # v_rep_top = torch.mean(v_rep_top, 0, keepdim=True).expand( v_rep_top.size() )
                v_norm = torch.cat((v_norm, v_rep_top), 0)
                
                #######################
                ## AUDIO BOTTOM/BCKGRND  
                a_sls_down_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)[1]
                a_rep_down = a_rep[i][a_sls_down_idxs]
                # a_rep_down = torch.mean(a_rep_down, 0, keepdim=True).expand( a_rep_down.size() )
                a_bgd = torch.cat((a_bgd, a_rep_down), 0)
                
                ## AUDIO UPPER/FRGND
                a_sls_top_idxs = torch.topk(a_sls[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)[1]
                a_rep_top = a_rep[i][a_sls_top_idxs]
                # a_rep_top = torch.mean(a_rep_top, 0, keepdim=True).expand( a_rep_top.size() )
                a_norm = torch.cat((a_norm, a_rep_top), 0)  
'''   
###########


class Glance(nn.Module):
    # https://github.com/pipixin321/GlanceVAD/blob/master/engine.py#L20
    # https://github.com/pipixin321/GlanceVAD/pull/5
    
    def __init__(self, _cfg, pfu: PstFwdUtils): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs//2 , f"dataload.balance.bag != 0.5"
        
        self.abn_ratio = _cfg.alpha
        #self.sigma = torch.nn.Parameter(torch.tensor(0.1))  
        self.sigma = torch.tensor(_cfg.sigma) #requires_grad=True, device=self.pfu.dvc
        self.min_mining_step = _cfg.min_mining_step

        self.sig = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction='none')  # Using reduction='none' for flexibility
        
    def gaussian_kernel_mining(self, score, point_label, seqlen=None):
        abn_snippet = point_label.clone().detach()
        bs, max_len = point_label.shape
        
        for b in range(bs):
            valid_len = max_len
            if seqlen is not None:
                valid_len = min(seqlen[b].item(), max_len)
                if valid_len != max_len: log.error(f"gkm {b} {valid_len=} != {max_len=}")
                
            #point_label = point_label[b, :valid_len]
            abn_idx = torch.nonzero(point_label[b]).squeeze(1)
            if len(abn_idx.shape) == 0 and abn_idx.numel() > 0:  # Handle single index case
                abn_idx = abn_idx.unsqueeze(0)
            if abn_idx.numel() == 0: continue
            #if len(abn_idx) == 0: continue   
            #log.debug(f"gkm {b} {abn_idx=}\n{score[b]=}")
                
            # Process most left boundary
            if abn_idx[0] > 0:
                for j in range(abn_idx[0]-1, -1, -1):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[0]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
            #log.debug(f"gkm {b} {abn_snippet[b]=}")
            
            # Process most right boundary (respect sequence length)
            if abn_idx[-1] < (valid_len-1):
                for j in range(abn_idx[-1]+1, valid_len):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[-1]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
            #log.debug(f"gkm {b} {abn_snippet[b]=}")    
            
            # Process between abnormal points
            for i in range(len(abn_idx)-1):
                if abn_idx[i+1] - abn_idx[i] <= 1:
                    continue
                # Forward pass
                for j in range(abn_idx[i]+1, abn_idx[i+1]):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[i]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
                # Backward pass
                for j in range(abn_idx[i+1]-1, abn_idx[i], -1):
                    abn_thresh = self.abn_ratio * score[b, abn_idx[i+1]]
                    if score[b, j] >= abn_thresh:
                        abn_snippet[b, j] = 1
                    else: break
            #log.debug(f"gkm {b} {abn_snippet[b]=}")            
        return abn_snippet

    def temporal_gaussian_splatting(self, point_label, distribution='normal', params=None, seqlen=None):
        point_label = point_label.clone().detach().cpu()
        bs, max_len = point_label.shape
        distribution_weight = torch.zeros_like(point_label)
        
        for b in range(bs):
            N = max_len
            if seqlen is not None:
                N = min(seqlen[b].item(), max_len)
                if N == 0: continue
                if N != max_len: log.error(f"tgs {b} {N=} != {max_len=}")
                
            #point_label = point_label[b, :N]
            abn_idx = torch.nonzero(point_label[b]).squeeze(1)
            if len(abn_idx.shape) == 0 and abn_idx.numel() > 0:  # Handle single index case
                abn_idx = abn_idx.unsqueeze(0)
            if abn_idx.numel() == 0: continue
            #log.debug(f"tgs {b} {abn_idx=} {N=}")
            
            temp_weight = torch.zeros([len(abn_idx), N])
            for i, point in enumerate(abn_idx):
                # Normalize to [-1, 1] range within valid sequence
                i_arr = torch.arange(N, dtype=torch.float32)
                if N > 1:
                    h_i = 2 * (i_arr - 1) / (N - 1) - 1
                    h_p = 2 * (point - 1) / (N - 1) - 1
                else:
                    h_i = h_p = torch.zeros(1)

                # Calculate weights based on distribution
                if distribution == 'normal':
                    weight = torch.exp(-(h_i - h_p) ** 2 / (2 * params['sigma']**2)) / (params['sigma'] * (2 * torch.pi)**0.5)
                elif distribution == 'cauchy':
                    weight = 1 / (1 + ((h_i - h_p) / params['gamma'])**2) / (torch.pi * params['gamma'])
                elif distribution == 'laplace':
                    weight = 0.5 * torch.exp(-torch.abs(h_i - h_p) / params['b']) / params['b']
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")

                # Normalize weights to [0, 1]
                if torch.min(weight) != torch.max(weight):
                    weight = (weight - torch.min(weight)) / (torch.max(weight) - torch.min(weight))
                else:
                    weight = torch.ones_like(weight)
                    
                temp_weight[i, :] = weight

            temp_weight = torch.max(temp_weight, dim=0)[0]
            # Normalize 
            if torch.min(temp_weight) != torch.max(temp_weight):
                temp_weight = (temp_weight - torch.min(temp_weight)) / (torch.max(temp_weight) - torch.min(temp_weight))
            # Assign weights to valid sequence positions only
            distribution_weight[b, :N] = temp_weight

        return distribution_weight

    def forward(self, ndata, ldata):
        """Unified forward method handling both interpolated and padded data"""
        labels = ldata['label']
        seqlen = ldata['seqlen']
        point_label = ldata['point_label']
        step = ldata["step"]
        idxs_seg = ldata["idxs_seg"]
        #log.debug(f"\n{idxs_seg=}")
        
        scores = self.sig(ndata['scores'])
        scores = self.pfu.uncrop(scores, 'mean')

        abn_scores, _ = self.pfu.unbag(scores, labels) ## bag,t
        abn_pnt_lbl, _ = self.pfu.unbag(point_label, labels) ## bag,t
        
        abn_seqlen = None
        if seqlen is not None:
            abn_seqlen, _ = self.pfu.unbag(seqlen, labels) ## bag
        
        needs_padding = False
        if seqlen is not None and torch.min(seqlen) < scores.shape[-1]:
            needs_padding = True
            log.debug(f"{needs_padding=}")
        
        # Gaussian Mining
        abn_kernel = self.gaussian_kernel_mining(
            abn_scores.detach().cpu(), 
            abn_pnt_lbl,
            abn_seqlen if needs_padding else None
        )
        
        # Temporal Gaussian Splatting
        if step < self.min_mining_step:
            rendered_score = self.temporal_gaussian_splatting(
                abn_pnt_lbl, 
                'normal', 
                params={'sigma': self.sigma},
                seqlen=abn_seqlen if needs_padding else None
            )
        else:
            rendered_score = self.temporal_gaussian_splatting(
                abn_kernel, 
                'normal', 
                params={'sigma': self.sigma},
                seqlen=abn_seqlen if needs_padding else None
            )
        rendered_score = rendered_score.to(self.pfu.dvc)

        per_element_loss = self.bce(abn_scores, rendered_score)
        
        if needs_padding:
            mask = torch.zeros_like(abn_scores, dtype=torch.bool)
            for i, length in enumerate(abn_seqlen):
                mask[i, :length] = True
            log.debug(f"{abn_seqlen=}  {mask=}")    
            
            # Apply mask and calculate mean over valid elements only
            masked_loss = per_element_loss * mask
            log.debug(f"{per_element_loss=}  {masked_loss=}")
            valid_count = torch.sum(mask, dim=1)
            batch_loss = torch.sum(masked_loss, dim=1) / torch.clamp(valid_count, min=1.0)
            loss = torch.mean(batch_loss)
        else:
            loss = torch.mean(per_element_loss)
        
        return {
            'glance': loss
        }
        
