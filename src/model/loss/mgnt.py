import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)


class Rtfm(nn.Module):
    def __init__(self, _cfg):
        super(Rtfm, self).__init__()
        
        self.alpha = _cfg.alpha
        self.margin = _cfg.margin
        self.k = _cfg.k
        self.bce = nn.BCELoss()
        self.do = nn.Dropout(0.7)
        
    def gather_feats(self, feat_magn, feats):
        nc, b, t, f = feats.shape
        
        feat_magn_drop = feat_magn * self.do( torch.ones_like(feat_magn) ) ## (b, t) drop some
        idx = torch.topk(feat_magn_drop, self.k , dim=1)[1] ## (b, 3)
        idx_feat = idx.unsqueeze(2).expand([-1, -1, feats.shape[2]]) ## (b, 3, f)
        #log.debug(f"{feat_magn.shape=} -topk-> {idx.shape=}->{idx_feat.shape=}")
        
        feats_sel = torch.zeros(0, device=feats.device)
        for i, feat in enumerate(feats):
            feats_sel = torch.gather(feat, 1, idx_feat)   # top 3 features magnitude in abnormal bag
            ## feats_sel = feat[torch.arange(bs)[:, None, None], idx_abn[:, :, None], :]
            ## (bs/2, 3, f)
            feats_sel = torch.cat((feats_sel, feats_sel))
        # (bs/2*nc, 3, f) 
        #log.debug(f"{feats.shape=} gather {idx_feat.shape=} over nc dim -> {feats_sel.shape=} ")
        
        return feats_sel, idx
    
    
    def forward(self, abnr_fmagn, norm_fmagn, abnr_feats, norm_feats, abnr_sls, norm_sls, ldata):
        
        ########
        ## FEATS
        ## abnormal
        feat_sel_abn, idx_abn = self.gather_feats(abnr_fmagn, abnr_feats) ## (bag*nc, k, f)
        afeat = feat_sel_abn.mean(dim=1) ## (bag*nc, f)
        l2norm_abn = torch.norm(afeat, p=2, dim=1) ## (bag*nc)
        loss_abn = torch.abs(self.margin - l2norm_abn)

        ## normal
        feat_sel_norm, idx_norm = self.gather_feats(norm_fmagn, norm_feats) ## (bag*nc, k, f)
        nfeat = torch.mean(feat_sel_norm, dim=1) ## topk mean
        loss_norm = torch.norm(nfeat, p=2, dim=1)
        
        loss_rtfm = torch.mean((loss_abn + loss_norm) ** 2)
        
        
        #################
        ## SCORES        
        #idx_abn_sls = idx_abn.unsqueeze(2).expand([-1, -1, abnr_sls.shape[2]]) ## (bag, 3, 1)
        sls_sel_abn = torch.gather(abnr_sls, dim=1, index=idx_abn) ## bag, k
        vls_abn = torch.mean( sls_sel_abn , dim=1) ## bag
        
        ## normal
        #idx_norm_sls = idx_norm.unsqueeze(2).expand([-1, -1, norm_sls.shape[2]]) ## (bag, 3, 1)
        sls_sel_norm = torch.gather(norm_sls, dim=1, index=idx_norm)
        vls_norm = torch.mean( sls_sel_norm , dim=1) ## bag
        
        
        vls = torch.cat((vls_norm, vls_abn)) ## (2*bag)
        #log.error(f"{ldata['label']} , {vls}")
        loss_vls = self.bce(vls, ldata['label'])
        #log.debug(f"RTFM/ loss_vls {loss_vls.item()} ")
        
        return  {
            'rtfm': self.alpha * loss_rtfm,
            'bce': loss_vls
        } 


if __name__ == '__main__':
    nc = 10
    bag = 16
    t = 32
    dfeat = 512
    cfg = {
        'k': 3,
        'alpha': 0.0001,
        'margin': 100
    }
    L = Rtfm(cfg)
    _ = L(
        abnr_fmagn=torch.randn(bag, t),
        norm_fmagn=torch.randn(bag, t),
        abnr_feats=torch.randn(nc, bag, t, dfeat),
        norm_feats=torch.randn(nc, bag, t, dfeat),
        abnr_sls=torch.randn(bag, t),
        norm_sls=torch.randn(bag, t),
        ldata=
        {
            'label': torch.cat( torch.zeros(bag), torch.ones(bag) )
        }
    )
    



    
### TORCH
                
#############
## MAGNITUDES
#feat_magn = torch.norm(feats, p=2, dim=2) ## ((bs)*nc, t)
#feat_magn = feat_magn.view(bs, nc, -1).mean(1) ## (bs, t)
#norm_fmagn = feat_magn[0:self.batch_size]  ## (bs//2, t) 
#abnr_fmagn = feat_magn[self.batch_size:]  ## (bs//2, t) 
#bag_size = norm_fmagn.shape[0] ## bs/2
#log.debug(f'{feat_magn.shape=} {norm_fmagn.shape=} {abnr_fmagn.shape=}\n')


#################
## ABNORMAL FEAT
## select topk idxs from features_magnitudes
#sel_idx = torch.ones_like(norm_fmagn)
#sel_idx = self.do(sel_idx)
#afea_magn_drop = abnr_fmagn * self.do( torch.ones_like(norm_fmagn) ) ## (bs//2, t)
#idx_abn = torch.topk(afea_magn_drop, self.k , dim=1)[1] ## (bs//2, 3)
####
## abnr_feats gather of idx_abn_feat over ncrop dim
#idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnr_feats.shape[2]]) ## (bs/2, 3, f) ## for gather element wise selection of feat
#abnr_feats = abnr_feats.view(bag_size, nc, t, f) ## (bs/2, nc, t, f)
#abnr_feats = abnr_feats.permute(1, 0, 2, 3)  ## (nc, bs/2, t, f)
#feat_sel_abn = torch.zeros(0, device=device)
#for i, abnr_feat in enumerate(abnr_feats):
#    feat_sel_abn = torch.gather(abnr_feat, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
#    ## feat_sel_abn = abnr_feat[torch.arange(bs)[:, None, None], idx_abn[:, :, None], :]
#    ## (bs/2, 3, f)
#    feat_sel_abn = torch.cat((feat_sel_abn, feat_sel_abn))
## (bs/2*nc, 3, f)  
        
#################
## ABNORMAL SCORES
## abnr_sls gather of idx_abn
#idx_abn_sls = idx_abn.unsqueeze(2).expand([-1, -1, abnr_sls.shape[2]]) ## (bs/2, 3, 1)
#sls_sel_abn = torch.gather(abnr_sls, 1, idx_abn_sls)  # top 3 scores in abnormal bag based on the top-3 magnitude
#vls_abn = torch.mean( sls_sel_abn , dim=1)
##############
## BERT mentioned it, scores are at VL
############


#################
## NORMAL FEAT
#select_idx_normal = torch.ones_like(norm_fmagn)
#select_idx_normal = self.do(select_idx_normal)
#nfea_magn_drop = norm_fmagn * select_idx_normal
#idx_norm = torch.topk(nfea_magn_drop, k_nor, dim=1)[1]
####
## norm_feats gather of idx_norm_feat over ncrop dim
#idx_norm_feat = idx_norm.unsqueeze(2).expand([-1, -1, norm_feats.shape[2]])
#norm_feats = norm_feats.view(bag_size, nc, t, f)
#norm_feats = norm_feats.permute(1, 0, 2, 3)
#feat_sel_norm = torch.zeros(0, device=device)
#for nor_fea in norm_feats:
#    feat_sel_norm = torch.gather(nor_fea, 1, idx_norm_feat)  # top 3 features magnitude in normal bag (hard negative)
#    feat_sel_norm = torch.cat((feat_sel_norm, feat_sel_norm))
## (bs*nc, 3, f)

#################
## NORMAL SCORE
#idx_norm_sls = idx_norm.unsqueeze(2).expand([-1, -1, nrom_sls.shape[2]])
#sls_sel_norm = torch.gather(norm_sls, 1, idx_norm_sls) # top 3 scores in abnormal bag based on the top-3 magnitude
#vls_norm = torch.mean( sls_sel_norm, dim=1 ) 
#log.debug(f"{norm_sls.shape=} gather {idx_norm_sls.shape=} -> mean {sls_sel_norm.shape=} = {vls_norm.shape=} ")
##############
## BERT mentioned it, scores are at VL
############