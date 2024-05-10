import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *

log = None
def init(l):
    global log
    log = l



class RTFM(nn.Module):
    def __init__(self, rgbnf, cfg_net, cfg_cls):
        super(RTFM, self).__init__()
        self.rgbnf = rgbnf
        
        self.aggregate = Aggregate(self.rgbnf)
        self.do = nn.Dropout(0.7)
        self.sig = nn.Sigmoid()
        
        ## classifier
        aaa = { 'MLP':  lambda cfg, fl: MLP( feat_len=fl, neurons=cfg.NEURONS, activa=cfg.ACTIVA, dropout=cfg.DO),
                'LSTM': lambda cfg, fl: LSTMSCls(feat_len=fl, lstm_dim=cfg.HIDDIM, lstm_bd=cfg.BD),
                'CONV': lambda cfg, fl: ConvCLS(feat_len=fl, ks=cfg.KS) }
        self.slcls = aaa.get(cfg_net.CLS_VRS)(cfg_cls, rgbnf)
        if not self.slcls: raise Exception(f'{cfg_net.CLS_VRS} not supported ')
        ## or 
        #self.slcls = get_cls(cfg_net.CLS_VRS, cfg_cls)
        
    def forward(self, x):
        b, t, f = x.shape
        log.debug(f'RTFM/x {x.shape}')
        
        rgbf = x[:, :, :self.rgbnf]
        #audf = x[:, :, self.rgbnf:]
        
        ## rgbf temporal refinement/ennahncment
        x_new = self.aggregate( rgbf.permute(0,2,1) ).permute(0,2,1) ## (b, t, f)
        x_new = self.do(x_new)
        log.debug(f'RTFM/aggregate {x_new.shape}')
        
        slscores = self.sig( self.slcls(x_new) )
        log.debug(f'RTFM/slscores {slscores.shape} ')
        
        return {
            'id': 'rtfm',
            'slscores': slscores,
            'feats': x_new
        }
        

class NetPstFwd(BasePstFwd):
    def __init__(self, bs, ncrops, dvc):
        super().__init__(bs, ncrops, dvc)

    def train(self, ndata, ldata, lossfx):
        
        ## FEATS
        #super().rshp_out(ndata, 'feats', 'mean')
        ##super().rshp_out(ndata, '', 'crop0')
        #log.debug(f" pos_rshp: {ndata['feats'].shape}")
        t, f = ndata['feats'].shape[1:3]
        feats = ndata['feats'] ## (bs*ncrops, t, f)

        ## magnitudes
        feat_magn = torch.norm(feats, p=2, dim=2)  ## ((bs)*ncrops, t)
        feat_magn = super().rshp_out2(feat_magn, 'mean') ## (bs, t)
        abnr_feat_magn = feat_magn[self.bs//2:]  ## (bs//2, t) 
        norm_feat_magn = feat_magn[0:self.bs//2]  ## (bs//2, t) 
        
        ### express feats ready to be gathered
        abnr_feats = feats[self.bs//2*self.ncrops:] 
        abnr_feats = abnr_feats.view(self.bs//2, self.ncrops, t, f)
        abnr_feats = abnr_feats.permute(1, 0, 2, 3) ## (ncrops, bs/2, t, f)
        log.debug(f"{abnr_feats.shape=}")
        
        norm_feats = feats[0:self.bs//2*self.ncrops]
        norm_feats = norm_feats.view(self.bs//2, self.ncrops, t, f)
        norm_feats = norm_feats.permute(1, 0, 2, 3) ## (ncrops, bs/2, t, f)
        log.debug(f"{norm_feats.shape=}")
        
        
        ## SCORES 
        super().rshp_out(ndata, 'slscores', 'mean')
        #super().rshp_out(ndata, '', 'crop0')
        log.debug(f" pos_rshp: {ndata['slscores'].shape}")
        slscores = ndata['slscores'] ## (bs, t)

        abnr_sls = slscores[self.bs//2:] ## (bs/2, t)
        norm_sls = slscores[0:self.bs//2] ## (bs/2, t)        
        
        ########
        ## LOSS 
        ## levarage frist returned dict and update only
        L = lossfx['rtfm'](abnr_feat_magn, norm_feat_magn, abnr_feats, norm_feats, abnr_sls, norm_sls, ldata)

        tmp_abnr_sls = abnr_sls.view(-1) ## (bs/2*t)
        loss_smooth = lossfx['rnkg'].smooth(tmp_abnr_sls)
        loss_spars = lossfx['rnkg'].sparsity(tmp_abnr_sls, rtfm=True)
        log.error(f"{loss_smooth.item()} ")
        
        return L.update({
            'loss_smooth': loss_smooth,
            'loss_spars': loss_spars
        })


    def infer(self, ndata):
        ## output is excepted to be segment level 
        log.debug(f"")
        log.debug(f"slscores: {ndata['slscores']=}")
        
        return ndata['slscores']    
    

### TORCH
                
#############
## MAGNITUDES
#feat_magn = torch.norm(feats, p=2, dim=2) ## ((bs)*ncrops, t)
#feat_magn = feat_magn.view(bs, ncrops, -1).mean(1) ## (bs, t)
#norm_feat_magn = feat_magn[0:self.batch_size]  ## (bs//2, t) 
#abnr_feat_magn = feat_magn[self.batch_size:]  ## (bs//2, t) 
#bag_size = norm_feat_magn.shape[0] ## bs/2
#log.debug(f'{feat_magn.shape=} {norm_feat_magn.shape=} {abnr_feat_magn.shape=}\n')


#################
## ABNORMAL FEAT
## select topk idxs from features_magnitudes
#sel_idx = torch.ones_like(norm_feat_magn)
#sel_idx = self.do(sel_idx)
#afea_magn_drop = abnr_feat_magn * self.do( torch.ones_like(norm_feat_magn) ) ## (bs//2, t)
#idx_abn = torch.topk(afea_magn_drop, self.K , dim=1)[1] ## (bs//2, 3)
####
## abnr_feats gather of idx_abn_feat over ncrop dim
#idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnr_feats.shape[2]]) ## (bs/2, 3, f) ## for gather element wise selection of feat
#abnr_feats = abnr_feats.view(bag_size, ncrops, t, f) ## (bs/2, ncrops, t, f)
#abnr_feats = abnr_feats.permute(1, 0, 2, 3)  ## (ncrops, bs/2, t, f)
#feat_sel_abn = torch.zeros(0, device=device)
#for i, abnr_feat in enumerate(abnr_feats):
#    feat_sel_abn = torch.gather(abnr_feat, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
#    ## feat_sel_abn = abnr_feat[torch.arange(bs)[:, None, None], idx_abn[:, :, None], :]
#    ## (bs/2, 3, f)
#    feat_sel_abn = torch.cat((feat_sel_abn, feat_sel_abn))
## (bs/2*ncrops, 3, f)  

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
#select_idx_normal = torch.ones_like(norm_feat_magn)
#select_idx_normal = self.do(select_idx_normal)
#nfea_magn_drop = norm_feat_magn * select_idx_normal
#idx_norm = torch.topk(nfea_magn_drop, k_nor, dim=1)[1]
####
## norm_feats gather of idx_norm_feat over ncrop dim
#idx_norm_feat = idx_norm.unsqueeze(2).expand([-1, -1, norm_feats.shape[2]])
#norm_feats = norm_feats.view(bag_size, ncrops, t, f)
#norm_feats = norm_feats.permute(1, 0, 2, 3)
#feat_sel_norm = torch.zeros(0, device=device)
#for nor_fea in norm_feats:
#    feat_sel_norm = torch.gather(nor_fea, 1, idx_norm_feat)  # top 3 features magnitude in normal bag (hard negative)
#    feat_sel_norm = torch.cat((feat_sel_norm, feat_sel_norm))
## (bs*ncrops, 3, f)

#################
## NORMAL SCORE
#idx_norm_sls = idx_norm.unsqueeze(2).expand([-1, -1, nrom_sls.shape[2]])
#sls_sel_norm = torch.gather(norm_sls, 1, idx_norm_sls) # top 3 scores in abnormal bag based on the top-3 magnitude
#vls_norm = torch.mean( sls_sel_norm, dim=1 ) 
#log.debug(f"{norm_sls.shape=} gather {idx_norm_sls.shape=} -> mean {sls_sel_norm.shape=} = {vls_norm.shape=} ")
##############
## BERT mentioned it, scores are at VL
############