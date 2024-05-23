import torch
import torch.nn as nn
import torch.nn.functional as F

log = None
def init(l):
    global log
    log = l
    
    
class BCE(nn.Module):
    def __init__(self):
        super(BCE, self).__init__()
        self.crit = nn.BCELoss()
        
    def forward(self, scores, label):
        #log.debug(f"BCE/{scores.shape} {scores.context} {label.shape} {label.context}")
        l = self.crit(scores, label)
        return {
            'loss_bce': l
            }


class RankingLoss(nn.Module):
    def __init__(self, bs, nsegments, dvc, lambda12=[8e-5,8e-5], version ='deepmil'):
        super(RankingLoss, self).__init__()
        
        self.bs = bs
        self.nsegments = nsegments
        self.lambda1 = lambda12[0] #np.array( lambda12[0] , ctx=dvc)
        self.lambda2 = lambda12[1] #np.array( lambda12[1] , ctx=dvc)
        self.dvc = dvc
        
        if version == 'deepmil': self.fx = self.ranking_deepmil
        elif version == 'milbert': self.fx = self.ranking_milbert
        elif version == 'tempatt': self.fx = self.ranking_tempatt
        
        
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

    def ranking_deepmil(self, slscores):
        ## https://github.com/Roc-Ng/DeepMIL
        
        if slscores.ndim == 2:
            slscores = np.reshape(slscores, -1)
            log.debug(f"{slscores.shape}")
            
        L = []
        for i in range(self.bs//2):
            ## norm
            startn = i * self.nsegments
            endn = (i + 1) * self.nsegments
            maxn = torch.max( slscores[ startn : endn ] ) 
            #maxn = torch.mean( torch.topk( slscores[ startn : endn ], k=self.nsegments//4) )
            
            ## anom
            starta = (i * self.nsegments + (self.bs//2) * self.nsegments)
            enda = (i + 1) * self.nsegments + (self.bs//2) * self.nsegments
            maxa = torch.max( slscores[ starta : enda ] ) ##that
            #maxa = torch.mean( torch.topk( slscores[ starta : enda ], k=self.nsegments//4) )
            
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
        return L 

    def ranking_milbert(self, slscores):
        ## https://github.com/wjtan99/BERT_Anomaly_Video_Classification/tree/main/MIL-BERT
        slscores = slscores['slscores']
        loss = np.zeros(1,ctx=self.dvc)
        #sparsity = np.zeros(1,ctx=self.dvc)
        #smooth = np.zeros(1,ctx=self.dvc)
        
        ## each batch is ( 2 * bs , 32 , 1)
        ## frist bs of 32 slscores are normal , second is abnormal
        for i in range(self.bs):
            normal_index = np.random.choice(self.nsegments, size=self.nsegments, replace=False, ctx=self.dvc)
            y_normal = slscores[i][normal_index]
            y_normal_max = np.max(y_normal)
            y_normal_min = np.min(y_normal)
            
            anomaly_index = np.random.choice(self.nsegments, size=self.nsegments, replace=False, ctx=self.dvc)
            y_anomaly = slscores[i + self.bs][anomaly_index]
            y_anomaly_max = np.max(y_anomaly)
            y_anomaly_min = np.min(y_anomaly)
            
            ## original milbert
            ## sparsity uses anomaly slscores shuffled
            ## smooth uses original anomaly slscores
            tmp = npx.relu(1.0 - y_anomaly_max + y_normal_max) 
            sparsity = np.sum(y_anomaly) * self.lambda1
            smooth = np.sum(np.square(slscores[i + self.bs, :31] - slscores[i + self.bs, 1:self.nsegments])) * self.lambda1
            ## catuon this is embbed fx, only works for 32 segments, if 64 gotta slide backwards two elements
            loss = loss + tmp + sparsity + smooth
        
        return loss
    
    def ranking_tempatt(self, slscores):
        ## Motion-Aware Feature for Improved Video Anomaly Detection
        ## train of optical flow autoencoder
        ## used later to get motion-aware feats
        ## zzz atm
        slscores = np.reshape(slscores['slscores'], -1) ## (b*t)
        attw = np.reshape(slscores['attw'], -1) ## (b*t)
        
        L = []
        for i in range(self.bs//2):
            ## normal
            startn = i * self.nsegments
            endn = (i + 1) * self.nsegments
            tmpn = np.sum( attw[startn:endn] * slscores[startn:endn])

            ## anom 
            starta = (i * self.nsegments + self.bs * self.nsegments)
            enda = (i + 1) * self.nsegments + self.bs * self.nsegments
            tmpa = np.sum( attw[starta:enda] * slscores[starta:enda])

            tmp = F.relu(1.0 - tmpa + tmpn)
            loss = tmp + self.sparsity(attw[starta:enda] * slscores[starta:enda])
            L.append(loss)

        L = np.stack(L, axis=0)
        return L
    
    
    def forward(self, slscores):
        loss_mil = self.fx( slscores )
        log.debug(f'RNKG/{loss_mil.shape=} {loss_mil.context=}')
        loss_mil = torch.mean(loss_mil)
        log.debug(f'RNKG/{loss_mil=} {loss_mil.shape=}')
        return {
            'loss_mil': loss_mil
        }
    
    
class RTFML(nn.Module):
    def __init__(self, cfg, dvc):
        super(RTFML, self).__init__()
        self.alpha = cfg.ALPHA
        self.margin = cfg.MARGIN
        self.k = cfg.K
        self.dvc = dvc
        self.bce = nn.BCELoss()
        self.do = nn.Dropout(0.7, inplace=True)
    
    def get_topk(self, feat_magn):
        ## select topk idxs from feat_magn
        feat_magn_drop = feat_magn * self.do( torch.ones_like(feat_magn) ) ## (bag, t) drop some
        idx = torch.topk(feat_magn_drop, k=self.k, dim=1)[1] ## (bag, 3)
        log.debug(f"{feat_magn.shape=} -topk-> {idx.shape=}")
        return idx
        
        
    def gather_feats(self, feats, idx):
        ## gathers selected idx from feats
        nc, b, t, f = feats.shape
        
        idx_feat = idx.unsqueeze(2).expand([-1, -1, f])
        feats_sel = torch.zeros(0, ) #device=self.dvc
        for i, feat in enumerate(feats):
            feat_sel = torch.gather(feat, dim=1, index=idx_feat)
            ## (bag, 3, f)
            feats_sel = torch.cat((feats_sel , feat_sel))
        ## (nc*bags, 3, f)

        log.debug(f"{feats.shape=} gather {idx.shape=} over ncrops dim -> {feats_sel.shape=} ")
        return feats_sel
    
    
    def forward(self, abnr_feat_magn, norm_feat_magn, abnr_feats, norm_feats, abnr_sls, norm_sls, ldata):
        '''
            _magn expects (bag, t)
            _feats expects (ncrops, bag, t, f)
            _sls expects (bag, t)
        '''
        
        idx_abnr = self.get_topk(abnr_feat_magn)
        idx_norm = self.get_topk(norm_feat_magn)
        
        ########
        ## FEATS
        #if abnr_feats is not None and norm_feats is not None:
            
        ## abnormal
        afeat = torch.mean( self.gather_feats(abnr_feats, idx_abnr) , dim=1 ) ## (bag*nc, k, f) -> (bag*nc, f)
        l2norm_abn = torch.norm(afeat, p=2, dim=1) 
        log.debug(f"RTFML/{l2norm_abn.shape=}")
        loss_abn = torch.abs(self.margin - l2norm_abn)
        
        ## normal
        nfeat = torch.mean( self.gather_feats(norm_feats, idx_norm), dim=1) ## (bag*nc, k, f) -> (bag*nc, f)
        loss_norm = torch.norm(nfeat, p=2, dim=1)
        
        loss_rtfm = torch.mean((loss_abn + loss_norm) ** 2)
        log.debug(f"RTFM/ loss_rtfm {loss_rtfm.item()} {loss_rtfm.device} ")
            
            
        #########
        ## SCORES
        ## BERT mentioned it, VIDEO LEVEL SPACE
        #if abnr_sls is not None and norm_sls is not None:
            
        ## abnormal
        sls_sel_abn = torch.gather( abnr_sls, dim=1, index=idx_abnr ) ## (bag, k)
        vls_abn = torch.mean( sls_sel_abn, dim=1 ) ## (bag)
        loss_bcea = self.bce( vls_abn, torch.ones_like(vls_abn).to(self.dvc) )
        
        ## normal
        sls_sel_norm = torch.gather(norm_sls, dim=1, index=idx_norm) ## (bag, k)
        vls_norm = torch.mean( sls_sel_norm, dim=1 ) ## (bag)
        loss_bcen = self.bce( vls_norm, torch.zeros_like(vls_norm).to(self.dvc) )
        
        loss_vls = loss_bcea + loss_bcen
        
        #loss_vls = self.bce( torch.cat((vls_norm, vls_abn)) , ldata['label'])
        log.debug(f"RTFM/ loss_vls {loss_vls.item()} {loss_vls.device} ")
        
        return {
            'loss_rtfm': self.alpha * loss_rtfm, 
            'loss_vls': loss_vls 
        } 
