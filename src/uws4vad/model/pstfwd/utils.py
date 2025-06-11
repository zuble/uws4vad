import torch
import torch.nn as nn
import torch.nn.functional as F

from uws4vad.utils import get_log
log = get_log(__name__)


class PstFwdUtils:
    """Provides utility functions for preprocesing loss calculations or inference."""
    def __init__(self, _cfg, tst=False):
        self.dvc = _cfg.dvc
        
        if not tst:
            self.bs = _cfg.bs
            self.ncrops = _cfg.ncrops[0]
            log.debug(f"PFU TRAIN{self.ncrops=}")
        else: 
            self.bs = 1
            self.ncrops = _cfg.ncrops[1]
            log.debug(f"PFU TEST {self.ncrops=}")
        
        ## 4train
        ## batch sampler
        self.abn_bal = _cfg.bal_abn_bag
        self.bat_div = int(self.bs*_cfg.bal_abn_bag)
        ## segmentation
        self.seg_len = _cfg.seg_len
        self.seg_sel = _cfg.seg_sel
        
        ## segment selection
        ## static sel
        self.k = _cfg.k
        self.do = nn.Dropout(_cfg.do)
        ## dyn sel
        self.slsr = _cfg.sls_ratio
        self.blsr = _cfg.bls_ratio
            
        ## additional
        self._cfg = _cfg     


    #################    
    def uncrop(self, arr, meth='', force=False):
        """
        Uncrops an array, combining values from multiple crops if applicable.

        Args:
            arr (torch.Tensor): The array to uncrop.
            meth (str, optional): The method to use for uncropping. 
                Options: 'mean' (average across crops), 'crop0' (select the first crop), 
                '' (no uncropping). Defaults to ''.
            force (bool, optional): If True, forces the crop dimension to be exposed 
                even if ncrops is 1. Defaults to False.

        Returns:
            torch.Tensor: The uncropped array.
        """
        ## !!! experiment w einops ?
        ## https://forums.fast.ai/t/lesson-24-official-topic/104358/3
        
        if self.ncrops > 1:
            log.debug(f"uncrop pre {list(arr.shape)}")
            if arr.ndim == 1:
                arr = arr.view(self.bs,self.ncrops)
            elif arr.ndim == 2:  ## scores: b*nc,_
                arr = arr.view(self.bs, self.ncrops, -1)
            elif arr.ndim == 3:  ## features: b*nc*_*_
                arr = arr.view(self.bs, self.ncrops, arr.shape[-2], arr.shape[-1])
            
            if meth == 'mean':
                arr = torch.mean(arr, dim=1)
            elif meth == 'crop0':
                arr = arr[:, 0, ...]  # Select the first crop
            elif meth != '':
                raise ValueError(f"Invalid uncrop method: {meth}")
            log.debug(f"uncrop pst {list(arr.shape)}")
        elif force: ## expose crop dim
            log.debug(f"uncrop pre {list(arr.shape)}")
            if arr.ndim == 2: ## sl
                raise NotImplementedError
            elif arr.ndim == 3: ## fl
                arr = arr.view(self.bs, 1, arr.shape[-2], arr.shape[-1]) 
            log.debug(f"uncrop pst {list(arr.shape)}")
        ## tests are a whole nooda lvl
        #assert self.bs == arr.shape[0]    
            
        return arr
            
    def unbag(self, arr, labels=None, permute=''):
        """
        Separates data into abnormal and normal samples based on labels.
        Args:
            arr (torch.Tensor): The data to unbag.
            labels (torch.Tensor, optional): The labels for the data. 
                If None, MIL is assumed, and the data is split in half.
        Returns:
            tuple: A tuple containing the abnormal and normal data.
        """
        log.debug(f"unbag pre") 
        
        if labels is None: #raise ValueError
            assert self.abn_bal == 0.5, "Labels are needed for non-MIL scenarios."
            abn = arr[:self.bat_div]
            nor = arr[self.bat_div:]
        else:
            #mask = labels != 0
            #log.error(f"\n{arr.shape}\n {labels}\n {mask}")
            ## !!! innificient, prefer array index
            abn = arr[labels != 0]
            nor = arr[labels == 0]
            
            #self.is_equal(abn2,abn)
            #self.is_equal(nor2,nor)

        ## permute here so indexing operates on right dim
        if permute == '1023':
            assert arr.ndim == 4
            abn, nor = abn.permute(1, 0, 2, 3), nor.permute(1, 0, 2, 3)
        elif permute == '021':
            assert arr.ndim == 3
            abn, nor = abn.permute(0, 2, 1), nor.permute(0, 2, 1)
        elif permute != '': 
            raise NotImplementedError
        
        log.debug(f"unbag from {list(arr.shape)} into A:{list(abn.shape)} N:{list(nor.shape)}")    
        return abn, nor
    #################


    ####################################
    ## METRICS
    ## dfm
    def calc_mahalanobis_dist(self, features, anchor, variance):
        assert features.ndim == anchor.ndim == variance.ndim == 3
        return torch.sqrt(torch.sum((features - anchor) ** 2 / variance, dim=1))
    
    def _get_mtrcs_dfm(self, feats, anchors, variances, infer=False, seqlen=None):
        dists, abn_dists, nor_dists = [], [], []
        for feat, anchor, var in zip(feats, anchors, variances):
            assert feat.ndim == 3
            assert anchor.ndim == var.ndim == 1
            
            ## calculate the malh_dist in seperate for each sample in batch 
            ## truncating to seqlen[i]
            #if seqlen is not None:
            #    log.error(f"{seqlen.shape}  {seqlen}")
            #    for i in range(len(seqlen)):
            #        log.debug(f"FEAT {i + 1} ")
            #        log.debug(feat[i, :, seqlen[i]:])  # Inspect distances after seqlen[i]
            
            ## bs*nc,f,t -> bs*nc,t        
            dist = self.calc_mahalanobis_dist(feat, anchor[None, :, None], var[None, :, None])
            log.debug(f"{feat.shape}&{anchor.shape}&{var.shape} -> {dist.shape}")
            # --- Inspection and Truncation ---
            #if seqlen is not None:
            #    for i in range(len(seqlen)):
            #        log.debug(f"DIST1 {i + 1} ")
            #        log.debug(dist[i, seqlen[i]:])  # Inspect distances after seqlen[i]
            
            dist = self.uncrop(dist, 'mean') ## bs,t
            dists.append(dist)
            
            tmp_abn, tmp_nor = self.unbag(dist) ## bag , t
            abn_dists.append(tmp_abn)
            nor_dists.append(tmp_nor)
            
            log.debug(f"dist {dist.shape}")# max {dist.max()}  mean {dist.mean()}")#{dist}
            log.debug(f"dist abn {tmp_abn.shape} mean {tmp_abn.mean()}")# max {tmp_abn.max()}  {tmp_abn}
            log.debug(f"dist nor {tmp_nor.shape} mean {tmp_nor.mean()}")# max {tmp_nor.max()}  {tmp_nor}
        
        if infer: return dists
        return abn_dists, nor_dists
    
    ############
    ## magnitude
    def calc_l2_norm(self, tensor, dim):
        return torch.linalg.norm(tensor, ord=2, dim=dim)
    
    def get_mtrcs_magn(self, feats, labels=None, apply_do=False):
        assert feats.ndim == 3
        if self.ncrops: assert feats.shape[0] == self.bs*self.ncrops
        fmagn = self.calc_l2_norm(feats, 2) ## bs*nc, t
        fmagn_uncp = self.uncrop(fmagn, 'mean') ## bs,t
        abn_fmagn, nor_fmagn = self.unbag(fmagn_uncp, labels)
        log.debug(f"fmagn {list(feats.shape)}->{list(fmagn.shape)}->{list(fmagn_uncp.shape)}->({list(abn_fmagn.shape)}, {list(nor_fmagn.shape)})")
        
        if apply_do:
            #log.debug(f"fmagn do {abn_fmagn} {nor_fmagn}")
            abn_fmagn = abn_fmagn * self.do(torch.ones_like(abn_fmagn))
            nor_fmagn = nor_fmagn * self.do(torch.ones_like(nor_fmagn))
            #log.debug(f"fmagn do {abn_fmagn} {nor_fmagn}")
        ## bag,t ; bag,t
        return abn_fmagn, nor_fmagn
    ####################################
    
    
    ###########
    ## GENERAL 
    ## selections is done per batch according to ratios
    def _sel_feats_by_batx(self, feats, metric, slsr=None, blsr=None):
        if slsr is None:
            slsr = self.slsr
        if blsr is None:
            blsr = self.blsr

        bag, t, f = feats.shape
        assert self.seg_len == t
        assert feats.shape[:-1] == metric.shape
        
        ## dynamic batch selection
        k_sample = int(t * slsr)
        k_batch = int(bag * t * blsr)
        log.debug(f"{k_sample=} {k_batch=}")
        
        mtrc_smpl = metric
        mtrc_batx = metric.reshape(-1)
        log.debug(f"{mtrc_smpl.shape=} {mtrc_batx.shape=}")
        
        ## SLS: sample -level/wise sel/mask
        topk_smpl = torch.topk(mtrc_smpl, k_sample, dim=-1)[1]
        mask_sel_smpl = torch.zeros_like(mtrc_smpl, dtype=torch.bool)
        mask_sel_smpl.scatter_(1, topk_smpl, True)
        
        ## BLS: batch -level/wise sel/mask
        topk_batx = torch.topk(mtrc_batx, k_batch, dim=-1)[1]
        mask_sel_batx = torch.zeros_like(mtrc_batx, dtype=torch.bool)
        mask_sel_batx.scatter_(0, topk_batx, True)
        
        mask_sel = mask_sel_batx | mask_sel_smpl.reshape(-1)
        sel_feats = feats.reshape(-1, f)[mask_sel] 
        
        assert torch.sum(mask_sel) == sel_feats.shape[0]
        num_sel = torch.sum(mask_sel) 
        
        #log.debug(f"1 mask:{mask_sel.shape} {mask_sel} ")
        log.debug(f"FEAT BATX SEL: from {list(feats.shape)} w/ {bag*t} segs -> sel {num_sel} -> {list(sel_feats.shape)}")
        return sel_feats
    
    def _sel_feats_by_smpl(self, feats, metric, k):
        assert feats.shape[:-1] == metric.shape
        assert metric.ndim == 2, f"metric.ndim {metric.ndim} != 2"
        bag, t, f = feats.shape
        idx_smpl = torch.topk(metric, k, dim=-1)[1]  ## (bag, k)
        #log.warning(f"{idx_smpl} {metric.shape} ")
        sel_feats = torch.gather(feats, dim=1, index=idx_smpl.unsqueeze(-1).expand(-1, -1, f)) ## bag,k,f
        
        log.debug(f"FEAT SMPL SEL: from {list(feats.shape)} w/ {bag*t} segs -> sel {k=} -> {list(sel_feats.shape)}")
        return sel_feats
    
    
    ## Sample-Batch Strategie
    def sel_feats_sbs(self, abn_feats, nor_feats, abn_mtrc, nor_mtrc, slsr=None, blsr=None):
        ## bag1,t,f / bag0,t,f / bag1,t / bag0,t
        
        sel_abn_feats = self._sel_feats_by_batx(abn_feats, abn_mtrc, slsr, blsr)
        num_sel_abn = sel_abn_feats.shape[0] ## 2 balanc
        
        k_nor_sample = self._get_k_sample(num_sel=num_sel_abn)
        sel_nor_feats = self._sel_feats_by_smpl(nor_feats, nor_mtrc, k_nor_sample) 
            
        ## Ensure both selections have the same number of samples
        ## bag0,k,f -> k,bag0,f -> k*bag0,f -> num_sel_abn,f
        sel_nor_feats = sel_nor_feats.permute(1, 0, 2).reshape(-1, sel_abn_feats.shape[-1])  
        sel_nor_feats = sel_nor_feats[:num_sel_abn]  
        
        return sel_abn_feats, sel_nor_feats     
    
    
    def _get_k_sample(self, sel_lvl='dyn', num_sel=None, slsr=None, k=None):
        if sel_lvl == 'dyn':
            if num_sel: ## normal , originaly set as bs//2
                k_sample = int(num_sel / (self.bs - self.bat_div) ) + 1
            else: ### abnormal select_num_sample
                if slsr is None: slsr = self.slsr
                k_sample = int(t * slsr)
                
        elif sel_lvl == 'static':
            ## selection idoes not take in account any ratio, and uses a static k
            if k is None: k_sample = self.k
            else: k_sample = k
        
        return k_sample
    
    ##################
    def gather_feats_per_crop(self, feats, metric, avg=False):
        assert feats.ndim == 4
        assert metric.ndim == 2
        nc, bag, t, f = feats.shape
        idx = torch.topk(metric, self.k , dim=1)[1] ## bag,k
        idx_feat = idx.unsqueeze(2).expand([-1, -1, feats.shape[-1]]) ## bag,k,f
        #log.debug(f"{metric.shape=} -top_{k}-> {idx.shape=}->{idx_feat.shape=}")
        
        feats_sel = torch.zeros(0, device=feats.device) ## bag*nc,k,f
        for i, feat in enumerate(feats):
            tmp = torch.gather(feat, 1, idx_feat) ## bag,k,f
            feats_sel = torch.cat((feats_sel, tmp))
        
        if avg: feats_sel = feats_sel.mean(dim=1) ## (bag, f)
        
        log.debug(f"gather_feats_per_crop: from {list(feats.shape)} w/ {bag*t} segs -> sel {self.k=} per {nc} crop -> {list(feats_sel.shape)}")
        return feats_sel, idx

    
    ## Sample Strategie
    def sel_feats_ss(self, abn_feats, nor_feats, abn_mtrc, nor_mtrc, 
                        k=None, per_crop=True, avg=False, labels=None):
        assert abn_feats.ndim in [3,4], f"abn_feats.ndim {abn_feats.ndim} != 3 or 4"
        assert nor_feats.ndim in [3,4], f"nor_feats.ndim {nor_feats.ndim} != 3 or 4"
        
        k_sample = self._get_k_sample(sel_lvl='static',k=k)
        idx_abn = torch.topk(abn_mtrc, k_sample, dim=1)[1] ## (bag, k_sample)
        idx_nor = torch.topk(nor_mtrc, k_sample, dim=1)[1] ## (bag, k_sample)
        
        if per_crop:  ##  nc,bag,t,f -> nc*bag,k,f
            assert abn_feats.ndim == nor_feats.ndim == 4
            assert idx_abn.ndim == idx_nor.ndim == 2
            ## abn
            #nc, bag, t, f = abn_feats.shape
            #sel_abn_feats2 = torch.zeros(0, device=abn_feats.device)
            #for i, abn_feat in enumerate(abn_feats):
            #    tmp = torch.gather(abn_feat, dim=1, index=idx_abn.unsqueeze(-1).expand(-1, -1, f)) ## bag,k,f 
            #    sel_abn_feats2 = torch.cat((sel_abn_feats2, tmp))
            #    
            #    tmp2 = self._sel_feats_by_smpl(abn_feat, abn_mtrc, k_sample)
            #    self.is_equal(tmp, tmp2)
            sel_abn_feats, idx_abn = self.gather_feats_per_crop(abn_feats, abn_mtrc, avg)
            #self.is_equal(sel_abn_feats2, sel_abn_feats)
            
            ## nor    
            #nc, bag, t, f = nor_feats.shape
            #sel_nor_feats2 = torch.zeros(0, device=nor_feats.device)
            #for i, nor_feat in enumerate(nor_feats):
            #    tmp = torch.gather(nor_feat, dim=1, index=idx_nor.unsqueeze(-1).expand(-1, -1, f)) ## bag,k,f
            #    sel_nor_feats2 = torch.cat((sel_nor_feats2, tmp))
            #    
            #    tmp2 = self._sel_feats_by_smpl(nor_feat, nor_mtrc, k_sample)
            #    self.is_equal(tmp, tmp2)
            sel_nor_feats, idx_nor = self.gather_feats_per_crop(nor_feats, nor_mtrc, avg)
            #self.is_equal(sel_nor_feats, sel_nor_feats2)
            
        else: ## bag,t,f -> bag,k,f
            #feats = self.uncrop(feats, 'mean') ## bs,t,f
            #abn_feats, nor_feats = self.unbag(feats, labels) ## bag,t,f
            assert abn_feats.ndim == nor_feats.ndim == 3
            
            sel_abn_feats = torch.gather(abn_feats, dim=1, index=idx_abn.unsqueeze(-1).expand(-1, -1, f))  ## bag,k,f
            sel_nor_feats = torch.gather(nor_feats, dim=1, index=idx_nor.unsqueeze(-1).expand(-1, -1, f))  ## bag,k,f
            
            sel_abn_feats2 = self._sel_feats_by_smpl(abn_feats, abn_mtrc, k_sample) ## bag,k,f
            sel_nor_feats2 = self._sel_feats_by_smpl(nor_feats, nor_mtrc, k_sample) ## bag,k,f
            
            self.is_equal(sel_abn_feats, sel_abn_feats2)
            self.is_equal(sel_nor_feats, sel_nor_feats2)
        
        log.debug(f"{k_sample=}")
        log.debug(f"{abn_mtrc.shape=} {nor_mtrc.shape=}")
        #log.debug(f"{abn_mtrc}\n{nor_mtrc}")
        log.debug(f"{idx_abn.shape=} {idx_nor.shape=}")
        #log.debug(f"{idx_abn}\n{idx_nor}")
        log.debug(f"{abn_feats.shape=} {nor_feats.shape=}")
        log.debug(f"{sel_abn_feats.shape=} {sel_nor_feats.shape=}")
        
        if avg: ## (bag, f)
            sel_abn_feats = sel_abn_feats.mean(dim=1)
            sel_nor_feats = sel_nor_feats.mean(dim=1)
            
        log.debug(f"FEAT SEL SS: {abn_feats.shape=} {nor_mtrc.shape=}")
        log.debug(f"sel_feats_ss: from {list(feats.shape)} w/ {bag*t} segs -> sel {self.k=} -> {list(feats_sel.shape)}")
        
        return sel_abn_feats, sel_nor_feats, idx_abn, idx_nor 


    def sel_scors(self, abn_scors, nor_scors, idxs_abn, idxs_nor, avg=False):
        assert abn_scors.ndim == idxs_abn.ndim == 2, f"{abn_scors.ndim=} {idxs_abn.ndim=}"
        assert nor_scors.ndim == idxs_nor.ndim == 2, f"{nor_scors.ndim=} {idxs_nor.ndim=}"

        sel_abn = torch.gather(abn_scors, dim=1, index=idxs_abn)
        sel_nor = torch.gather(nor_scors, dim=1, index=idxs_nor)
        
        if avg: ## bag
            sel_abn = sel_abn.mean(dim=1)
            sel_nor = sel_nor.mean(dim=1) 
            
        log.debug(f"sel_scors: from {list(abn_scors.shape)} sel {list(idxs_abn.shape)} -> {list(sel_abn.shape)} ")
        return sel_abn, sel_nor
    
    
    ###########
    ## TODO: pel4vad infer modules
    def fixed_smooth(self, logits, t_size):
        ins_preds = torch.zeros(0).cuda()
        assert t_size > 1
        if len(logits) % t_size != 0:
            delta = t_size - len(logits) % t_size
            logits = F.pad(logits, (0,  delta), 'constant', 0)

        seq_len = len(logits) // t_size
        for i in range(seq_len):
            seq = logits[i * t_size: (i + 1) * t_size]
            avg = torch.mean(seq, dim=0)
            avg = avg.repeat(t_size)
            ins_preds = torch.cat((ins_preds, avg))

        return ins_preds

    def slide_smooth(self, logits, t_size, mode='zero'):
        assert t_size > 1
        ins_preds = torch.zeros(0).cuda()
        padding = t_size - 1
        if mode == 'zero':
            logits = F.pad(logits, (0, padding), 'constant', 0)
        elif mode == 'constant':
            logits = F.pad(logits, (0, padding), 'constant', logits[-1])

        seq_len = int(len(logits) - t_size) + 1
        for i in range(seq_len):
            seq = logits[i: i + t_size]
            avg = torch.mean(seq, dim=0).unsqueeze(dim=0)
            ins_preds = torch.cat((ins_preds, avg))

        return ins_preds

    def out(self, scors):
        ## dev
        if scors.shape == (1, 1): return scors.reshape(1)
        else: 
            return scors.squeeze()
            #return scores.view(-1)
    
    
    #######
    ## misc    
    def is_equal(self, dado1, dado2):
        ## torch.testing.assert_close(dado1, dado2)
        assert torch.all(dado1.eq(dado2)) == True, f"{dado1.shape} != {dado2.shape}"
        
    def logdat(self, *ds):
        """Logs information about data dictionaries."""
        log_string = ""
        for i, d in enumerate(ds):
            log_string += f"\nD[{i+1}]:"
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            log_string += f"  - {key}: {value.item()}\n"
                        else:
                            log_string += f"  - {key}: shape={list(value.shape)}, dtype={value.dtype}, device={value.device}\n"
                    elif isinstance(value, (list, tuple)):
                        log_string += f"  - {key}: (list/tuple) len={len(value)}\n"
                        for j, v in enumerate(value):
                            if isinstance(v, torch.Tensor):
                                log_string += f"    - Item {j+1}: shape={list(v.shape)}, dtype={v.dtype}, device={v.device}\n"
                            else:
                                log_string += f"    - Item {j+1}: {v}\n"
                    else:
                        log_string += f"  - {key}: {value}\n"
            else:
                log_string += f"  - (Not a dictionary): {d}\n"
                
        log.info(log_string)

    ## aggregatio on return
    def merge(self, *dicts):
        res = {}
        for d in dicts: res.update(d)
        return res
    

    '''
    ## original sbs strategie that used distances, raw implementation - OK
    def sel_feats_by_dist(self, abn_feats, nor_feats, abn_dists, nor_dists, slsr=None, blsr=None):
        """
        Selects features based on distance using sample-level and batch-level selection ratios.

        Args:
            features (torch.Tensor): The feature tensor (shape: bs, c, t or bs, t, f).
            distances (torch.Tensor): The distance matrix (shape: bs, t).
            slsr (float, optional): Sample-level selection ratio. If None, uses self.slsr.
            blsr (float, optional): Batch-level selection ratio. If None, uses self.blsr.

        Returns:
            tuple: A tuple containing the selected normal and abnormal features.
        """
        if slsr is None:
            slsr = self.slsr
        if blsr is None:
            blsr = self.blsr

        bag, t, f = abn_feats.shape
        assert self.tlen == t
        assert abn_feats.shape[:-1] == abn_dists.shape
        
        ## as its using interpolate all idxs have content
        ## how can i adapt to work with selorpad
        ## so selection happens with non corrupted segments
        ## make use of seqlen 
        
        mtrc_smpl = abn_dists
        mtrc_batx = abn_dists.reshape(-1)
        log.debug(f"DIST {mtrc_smpl.shape=} {mtrc_batx.shape=}")
        
        ## dynamic batch selection
        k_sample = int(t * slsr)
        #k_batch = int(bs // 2 * t * blsr)
        k_batch = int(bag * t * blsr) ## set modular and batch may be unbalenc
        log.debug(f"{k_sample=} {k_batch=}")
        
        
        ## SLS: sample -level/wise sel/mask
        topk_smpl = torch.topk(mtrc_smpl, k_sample, dim=-1)[1]
        mask_sel_smpl = torch.zeros_like(mtrc_smpl, dtype=torch.bool)
        mask_sel_smpl.scatter_(1, topk_smpl, True)
        
        ## BLS: batch -level/wise sel/mask
        topk_batx = torch.topk(mtrc_batx, k_batch, dim=-1)[1]
        mask_sel_batx = torch.zeros_like(mtrc_batx, dtype=torch.bool)
        mask_sel_batx.scatter_(0, topk_batx, True)
        
        
        ############################
        ## Abnormal feature selection
        ## bag*t,f -> num_sel_abn,f
        mask_sel_abn = mask_sel_batx | mask_sel_smpl.reshape(-1)
        num_sel_abn = torch.sum(mask_sel_abn)
        sel_abn_feats = abn_feats.reshape(-1, f)[mask_sel_abn] 
        
        log.debug(f"DIST ABN: from {bag*t} sel {sel_abn_feats.shape[0]} {num_sel_abn=}")
        log.debug(f"DIST ABN: mask{mask_sel_abn.shape} {mask_sel_abn} ")
        
        ## what to do with those that are abnormal
        ## grab and orientate to a specif subspcae
        ## as abnormals are all treated as 1 big class
        
        
        bag,_,_ = nor_feats.shape
        #num_sel_nor = int(num_sel_abn / (bs // 2)) + 1 ## why?
        num_sel_nor = int(num_sel_abn / bag) + 1
        log.debug(f"{nor_dists.shape}{num_sel_nor=} ")
        
        ##########################
        # Normal feature selection (match number of abnormal selections)
        ## bag,t,f -> bag,num_sel_nor,f 
        idx_nor = torch.topk(nor_dists, num_sel_nor, dim=-1)[1] ## bag,nsn
        idx_feat_nor = idx_nor[..., None].expand(-1, -1, f) ## bag,nsn,f
        sel_nor_feats = torch.gather(nor_feats, dim=1, index=idx_feat_nor) 
        log.debug(f"DIST NOR: from {bag*t} sel {num_sel_nor} -> {sel_nor_feats.shape[0]}")

        #######
        ### or
        sel_nor_feats2 = self._sel_feats_by_smpl(nor_feats, nor_dists, num_sel=num_sel_abn)
        self.is_equal(sel_nor_feats, sel_nor_feats2)
        
        ## -> num_sel_nor*bag,f -> num_sel_abn, f
        sel_nor_feats = sel_nor_feats.permute(1, 0, 2).reshape(-1, f)
        sel_nor_feats = sel_nor_feats[:num_sel_abn]
        log.debug(f"DIST NOR: trunc2 {num_sel_abn=}")
        return sel_abn_feats, sel_nor_feats'''    



## OLD IDEA !
## every netowrk has a NetPstFwd class
## which contains the pst processing depending if train / infer
## they inherit BasePstFwd so rsp_out is acessible w/o import or repeats
## relying on the cfg.TRAIN.BS and cfg.DATA.RGB.NCROPS 
## in pair with the creation of network in nets/_nets/get_net()
## while keepig the info regarding every post processing after net fwd 
## in the same file as the net def for both 
##  train, directly interact w net outs and inputs of chosen lossfx's
##      levarage that each lossfx returns a dict
##      furthermore the train_epo expects a dict of losses
##  infer, to adapt values 4 inference
## train_ep/train_epo (use netpstfwd.train)
## vldt/vldt/Validate (use netpstfwd.infer)
## both enabling the principal loop to stay static nd focus on architect
## if really needed realy on id key in ndata to post process both cases