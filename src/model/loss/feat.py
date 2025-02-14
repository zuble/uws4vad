import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from pytorch_metric_learning import distances
from pytorch_metric_learning import reducers
from pytorch_metric_learning import regularizers

from src.model.loss.score import smooth, sparsity
from src.model.pstfwd.utils import PstFwdUtils


from functools import partial
from src.utils.logger import get_log
log = get_log(__name__)
    
    
class MPP(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs // 2
        if self.pfu.seg_sel != 'itp':
            log.warning("MPP WITHOUT ITP, PADDED SEQ NOT INVESTIGATED")
        self.margin = _cfg.margin
        self.w_triplet = _cfg.w_triplet
        self.w_mpp = _cfg.w_mpp

    def preproc(self, norm_feats, anchors, variances, labels):
        assert norm_feats[0].shape[-1] == self.pfu.seg_len
        assert norm_feats[1].shape[-1] == self.pfu.seg_len
        ## [(b,fred1,t), (b,fred2,t)] ;; [fred1,fred2] ;; [fred1,fred2]

        abn_dists, nor_dists = self.pfu._get_mtrcs_dfm(norm_feats, anchors, variances)
        
        #sel_abn_feats = torch.zeros(0, device=self.dvc)
        #sel_nor_feats = torch.zeros(0, device=self.dvc)
        sel_abn_feats, sel_nor_feats = [], []
        for feats, abn_dist, nor_dist in zip(norm_feats, abn_dists, nor_dists):
            
            feats = feats.permute(0,2,1) ## bs, t, f
            feats = self.pfu.uncrop(feats, 'mean') ## bs, t, f
            abn_feats, nor_feats = self.pfu.unbag(feats, labels) ## bag, t, f
            
            ## bag,k,f
            #tmp_abn2, tmp_nor2 = self.pfu.sel_feats_by_dist(abn_feats, nor_feats, abn_dist, nor_dist)
            tmp_abn, tmp_nor = self.pfu.sel_feats_sbs(abn_feats, nor_feats, abn_dist, nor_dist)
            #self.pfu.is_equal(tmp_abn,tmp_abn2)
            #self.pfu.is_equal(tmp_nor,tmp_nor2)
            sel_abn_feats.append(tmp_abn[..., None]) ## _, red1, 1 ;; _, red2, 1
            sel_nor_feats.append(tmp_nor[..., None]) ## _, red1, 1 ;; _, red2, 1
            
        return sel_abn_feats, sel_nor_feats
    
    def forward(self, ndata, ldata):
        anchors = ndata['anchors']
        variances = ndata["variances"]
        norm_feats = ndata["norm_feats"]
        labels = ldata['label']
        seqlen = ldata['seqlen']
        
        abn_sel, nor_sel = self.preproc(norm_feats, anchors, variances, labels)
        #self.pfu.logdat({'abn': abn_sel,'nor': nor_sel})
        
        def mahalanobis_distance(mu, x, var):
            return torch.sqrt(torch.sum((x - mu)**2 / var, dim=-1))
        
        # nor_sel / abn_sel are result of DFM (topk malh_dist embd:anchor  ) 
        
        L = []
        for anchor, var, pos, neg, wt in zip(anchors, variances, nor_sel, abn_sel, self.w_triplet):
            lossf = nn.TripletMarginWithDistanceLoss(margin=self.margin, distance_function=partial(mahalanobis_distance, var=var))
            
            B, C, k = pos.shape ## 5699, red1, 1 // 5699, red2, 1
            pos = pos.permute(0, 2, 1).reshape(B*k, -1)
            neg = neg.permute(0, 2, 1).reshape(B*k, -1)
            
            loss_triplet = lossf(anchor[None, ...].repeat(B*k, 1), pos, neg)
            L.append(loss_triplet * wt)
            
        #L = torch.stack(L, dim=0).sum()
        L = sum(L)
        log.debug(f'{L=}')
        return {
            'mpp': self.w_mpp * L,
            #'mpp2': self.w_mpp * l[1]
        } 
        
class Rtfm(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils = None):
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs // 2
        
        if _cfg.preproc == 'og': self.preproc = self._preproc
        elif _cfg.preproc == 'sbs': self.preproc = self._preproc2
        
        self.alpha = _cfg.alpha
        self.margin = _cfg.margin
        #self.bce = nn.BCELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        self.smooth = smooth
        self.sparse = sparsity
    
    def _preproc2(self, feats, labels):
        ####################
        ## experiment batch-level sel as bndfm
        feats_uc = self.pfu.uncrop(feats, 'mean') ## bs, t, f 
        abn_feats, nor_feats = self.pfu.unbag(feats_uc, labels) ## bag, t, f
        abn_fmgnt, nor_fmgnt = self.pfu.get_mtrcs_magn(feats, labels=labels, apply_do=True) ## bag,t
        
        #abn_fmgnt, nor_fmgnt = self.pfu.get_mtrcs_magn(feats, labels=labels, apply_do=True)
        #feats_pnc = self.pfu.uncrop(feats, force=True) ## bs,nc,t,f
        #abn_feats, nor_feats = self.pfu.unbag(feats_pnc, labels=labels, permute='1023') ## nc,bag,t,f
        
        sel_abn_feats, sel_nor_feats = self.pfu.sel_feats_sbs(abn_feats, nor_feats, abn_fmgnt, nor_fmgnt)
        
        k_sample = self.pfu._get_k_sample(sel_lvl='static')
        idx_abn = torch.topk(abn_fmgnt, k_sample, dim=1)[1] ## (bag, k_sample)
        idx_nor = torch.topk(nor_fmgnt, k_sample, dim=1)[1] ## (bag, k_sample)
        
        return sel_abn_feats, idx_abn, sel_nor_feats, idx_nor
        #log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        #abn_feat = sel_abn_feats
        #nor_feat = sel_nor_feats
        ### 4 scores
        #idx_abn = self.pfu._get_topk_idxs(abn_fmgnt, self.k, self.do)
        #idx_nor = self.pfu._get_topk_idxs(nor_fmgnt, self.k, self.do)
        #L0 = lossfx['mgnt'](abn_feat, nor_feat) ## video
        ########################
    
    def _preproc(self, feats, labels):
        ## bag*nc,f

        abn_fmgnt, nor_fmgnt = self.pfu.get_mtrcs_magn(feats, labels=labels, apply_do=True)

        feats_pnc = self.pfu.uncrop(feats, force=True) ## bs,nc,t,f
        abn_feats, nor_feats = self.pfu.unbag(feats_pnc, labels=labels, permute='1023') ## nc,bag,t,f
        
        ## nc,bag,k,f ->  bag*nc, f (crop dim kept exposed) > key point, sufffers w/ nc=1 on ucf
        sel_abn_feats, idx_abn = self.pfu.gather_feats_per_crop(abn_feats, abn_fmgnt, avg=True)
        sel_nor_feats, idx_nor = self.pfu.gather_feats_per_crop(nor_feats, nor_fmgnt, avg=True)
        
        ## bag*nc,k,f ->  bag*nc, f (crop dim kept exposed)
        #sel_abn_feat, sel_nor_feats, idx_abn, idx_nor = super().sel_feats_ss(abn_feats, nor_feats, 
        #                                                                abn_fmgnt, nor_fmgnt,
        #                                                                per_crop=True, avg=True)
        return sel_abn_feats, idx_abn, sel_nor_feats, idx_nor
    
    def forward(self, ndata, ldata):
        feats = ndata['feats']
        scores = ndata['scores']
        labels = ldata['label']

        sel_abn_feats, idx_abn, sel_nor_feats, idx_nor = self.preproc(feats, labels)
        log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        
        
        l2n_abn = torch.norm(sel_abn_feats, p=2, dim=1) ## (bag*nc)
        l_fabn = torch.abs(self.margin - l2n_abn)
        l_fnor = torch.norm(sel_nor_feats, p=2, dim=1) ## (bag*nc)
        loss_mgnt = torch.mean((l_fabn + l_fnor) ** 2)
        
        log.debug(f"{l_fabn.shape=} {l_fnor.shape=} {loss_mgnt=} ")
        
        
        ## SCORE
        scores = self.pfu.uncrop( scores, 'mean') ## bs*nc,t -> bs, t
        abn_scors, nor_scors = self.pfu.unbag(scores, labels) ## bag, t
        vls_abn, vls_nor = self.pfu.sel_scors( abn_scors, nor_scors, idx_abn, idx_nor, avg=True) ## bag
        log.debug(f"{vls_abn.mean()=}  {vls_nor.mean()=}")
        
        #vls_abn, vls_nor = self.sig(vls_abn), self.sig(vls_nor)
        #log.debug(f"{vls_abn.mean()=}  {vls_nor.mean()=}")
        
        loss_scor = self.bce(torch.cat((vls_abn,vls_nor)), ldata['label']) ## vls
        ## (bag*t)
        #loss_smooth = self.smooth(abn_scors.view(-1)) ## sls
        loss_spars = self.sparse(abn_scors.view(-1), rtfm=True) ## sls
        
        return {
                'rtfm': self.alpha * loss_mgnt,
                'bce': loss_scor,
            }
        #return self.pfu.merge( {
        #        'rtfm': self.alpha * loss_mgnt,
        #        'bce': loss_scor,
        #    }, 
        #    #loss_smooth,
        #    loss_spars
        #    )
        
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0): 
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True) ## bag*nc, 1
        loss_contrastive = torch.mean(
                            (1-label) * torch.pow(euclidean_distance, 2) +
                            (label) * torch.pow( torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
                        )
        return loss_contrastive
    
class MagnCont(nn.Module):
    def __init__(self, _cfg, pfu: PstFwdUtils = None): 
        super().__init__()
        self.pfu = pfu
        assert self.pfu.bat_div == self.pfu.bs // 2
        
        self.alpha = _cfg.alpha
        
        self.crit = ContrastiveLoss(_cfg.margin)
        
    def preproc(self, feats, labels):
        ## bag*nc,f
        
        abn_fmgnt, nor_fmgnt = self.pfu.get_mtrcs_magn(feats, labels=labels, apply_do=True)

        feats_pnc = self.pfu.uncrop(feats, force=True) ## bs,nc,t,f
        abn_feats, nor_feats = self.pfu.unbag(feats_pnc, labels=labels, permute='1023') ## nc,bag,t,f
        
        ## nc,bag,k,f ->  bag*nc, k, f (crop dim kept exposed and no avg)
        sel_abn_feats, idx_abn = self.pfu.gather_feats_per_crop(abn_feats, abn_fmgnt, avg=False)
        sel_nor_feats, idx_nor = self.pfu.gather_feats_per_crop(nor_feats, nor_fmgnt, avg=False)
        
        ## bag*nc,k,f ->  bag*nc, k, f  not tested
        #sel_abn_feat, sel_nor_feats, idx_abn, idx_nor = super().sel_feats_ss(abn_feats, nor_feats, 
        #                                                                abn_fmgnt, nor_fmgnt,
        #                                                                per_crop=True, avg=False)
        return sel_abn_feats, idx_abn, sel_nor_feats, idx_nor
    
    def forward(self, ndata, ldata):
        feats = ndata['feats']
        #scores = ndata['scores']
        labels = ldata['label']
        
        sel_abn_feats, _, sel_nor_feats, _ = self.preproc(feats, labels)
        log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        
        seperate = sel_abn_feats.shape[0] // 2

        ## different from rtfm, no mean over topk sel feats
        ## thus sel_abn_feats/sel_nor_feats = bag*nc, k, f 
        loss_con = self.crit(torch.norm(sel_abn_feats, p=1, dim=2), ## bag*nc, k
                                    torch.norm(sel_nor_feats, p=1, dim=2), ## bag*nc, k
                                    1)  # try tp separate normal and abnormal
        ## bag*nc, k, f 
        loss_con_n = self.crit(torch.norm(sel_nor_feats[seperate:], p=1, dim=2), ## (bag*nc)/2, k
                                    torch.norm(sel_nor_feats[:seperate], p=1, dim=2),  ## (bag*nc)/2, k
                                    0)  # try to cluster the same class 
        ## bag*nc, k, f 
        loss_con_a = self.crit(torch.norm(sel_abn_feats[seperate:], p=1, dim=2), ## (bag*nc)/2, k
                                    torch.norm(sel_abn_feats[:seperate], p=1, dim=2), ## (bag*nc)/2, k
                                    0)
        #loss_total = loss_cls + self.alpha * (0.001 * loss_con + loss_con_a + loss_con_n )
        
        ## SCORE
        #scores = self.pfu.uncrop( scores, 'mean') ## bs*nc,t -> bs, t
        #abn_scors, nor_scors = self.pfu.unbag(scores, labels) ## bag, t
        #vls_abn, vls_nor = self.pfu.sel_scors( abn_scors, nor_scors, idx_abn, idx_nor, avg=True) ## bag
        #log.debug(f"{vls_abn.mean()=}  {vls_nor.mean()=}")
        #
        #loss_scor = self.bce(torch.cat((vls_abn,vls_nor)), ldata['label']) ## vls
        ##L1 = lossfx['bce'](vls_abn, torch.ones_like(vls_abn), 'abn') ## vls
        ##L2 = lossfx['bce'](vls_nor, torch.zeros_like(vls_nor), 'nor') ## vls
        
        ## (bag*t)
        #loss_smooth = self.smooth(abn_scors.view(-1)) ## sls
        #loss_spars = self.sparse(abn_scors.view(-1), rtfm=True) ## sls
        
        return {
            'abn-nor': loss_con * self.alpha**2,
            'abn':  loss_con_a * self.alpha,
            'nor':  loss_con_n * self.alpha,
        }


class Dev(nn.Module):
    def __init__(self, _cfg):
        super().__init__()
        
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
        