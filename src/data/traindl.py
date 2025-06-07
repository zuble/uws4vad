import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, WeightedRandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from src.data.samplers import AbnormalBatchSampler, analyze_sampler, dummy_train_loop
import numpy as np
import pandas as pd
import glob, os, os.path as osp, math, time
from multiprocessing import Pool

from src.utils import hh_mm_ss, seed_sade, logger
log = logger.get_log(__name__)
    

## dirt .. needed ? 
NPRNG = None
TCRNG = None
def get_rng(seed):
    global NPRNG, TCRNG
    NPRNG = np.random.default_rng(seed)
    ## https://pytorch.org/docs/1.8.1/generated/torch.Generator.html#torch.Generator
    TCRNG = torch.Generator().manual_seed(seed)


def get_trainloader(cfg, vis=None):
    from ._data import FeaturePathListFinder, debug_cfg_data
    if cfg.get("debug"): debug_cfg_data(cfg)
    
    cfg_ds = cfg.data
    cfg_dload = cfg.dataload
    cfg_dproc = cfg.dataproc
    
    get_rng(cfg.seed)
    
    ## THIS CAN BE SIMPLYFIED
    ## rather than find on the fly, use list files ? 
    ## we can still have some check logic (number of files match, resolve in edge cases, etc)
    rgbfplf = FeaturePathListFinder(cfg, 'train', 'rgb')
    argbfl, nrgbfl = rgbfplf.get('ANOM', cfg.dataproc.culum), rgbfplf.get('NORM')
    len_abn = len(argbfl); len_nor = len(nrgbfl)
    log.info(f'TRAIN: RGB {len_abn} abnormal, {len_nor} normal') 
    
    aaudfl, naudfl = [], []
    if cfg_ds.get('faud'):
        log.debug("AUD")
        audfplf = FeaturePathListFinder(cfg, 'train', 'aud', auxrgbflist=argbfl+nrgbfl)
        aaudfl, naudfl = audfplf.get('ANOM', cfg.dataproc.culum),  audfplf.get('NORM')
        log.info(f'TRAIN: AUD ON {len(aaudfl)} abnormal, {len(naudfl)} normal')    
    
    rgbfl = argbfl + nrgbfl
    audfl = aaudfl + naudfl
    
    ds = TrainDS(cfg_dload, cfg_ds, cfg_dproc, rgbfl, audfl)  
    
    ## --- Sampler ---
    bal_abn_bag = cfg_dload.balance.bag
    bal_abn_set = cfg_dload.balance.set
    
    if bal_abn_bag == -1:
        ## https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/
        from src.data import LBL
        
        labels_id = cfg_ds.lbls.info[:-2]
        log.info(f"MPerClassSampler: {labels_id}") ## len in xdv ~ 7
        
        lbl_mng = LBL(ds=cfg_ds.id, cfg_lbls=cfg_ds.lbls)
        labels=[]
        for path in rgbfl: ## apeend frist one
            labels.append( lbl_mng.encod(osp.basename(path))[0] )
        log.info(f"MPerClassSampler: {len(labels)}")
        
        m = cfg_dload.bs // len(labels_id)  
        ## 128 -> 126 | 64 -> 63 | 32 -> 28
        bs = cfg_dload.bs - cfg_dload.bs // len(labels_id)
        niters = len(argbfl) #*2 #len(rgbfl) 
        log.info(f"MPerClassSampler: {m=} {bs=} {niters=} ")
        #assert m * len(labels_id) >= cfg_dload.bs
        #assert cfg_dload.bs % m == 0
        sampler = MPerClassSampler(
                    labels=labels, 
                    m=m, 
                    #batch_size=cfg_dload.bs, #bs,
                    length_before_new_iter = niters
                    )
    elif bal_abn_bag == 0:
        ## same as batch_size=cfg_dload.bs, shuffle=True, drop_last=True
        sampler = RandomSampler(ds, 
                    #replacement=False,
                    #num_samples=samples_per_epoch, # !!!! set intern to len(ds)
                    generator=TCRNG
                    )
    else:
        labels = [1]*len_abn + [0]*len_nor
        sampler = AbnormalBatchSampler(
                        labels,
                        bal_abn_bag=bal_abn_bag,
                        bs=cfg_dload.bs,
                        bal_abn_set=bal_abn_set,
                        generator=NPRNG
                    )

    bsampler = BatchSampler(sampler, cfg_dload.bs, True) #cfg_dload.droplast
    log.debug(f"TRAIN: {sampler=} w/ {len(sampler)} -> [{bal_abn_bag=} {bal_abn_set=}]")
    log.debug(f"TRAIN: {bsampler=} w/ {len(bsampler)} ")
    
    #sampler_debug(bal_abn_bag, bsampler, ds, )
    if cfg.get("debug"):
        if cfg.debug.id == 'smplr':
            a={ f'abs_{bal_abn_bag}': bsampler, 
                'org_xdv': BatchSampler(
                            RandomSampler(ds,
                            generator=TCRNG
                            ),cfg_dload.bs , False),
                #'abs_bal_os': BatchSampler(
                #            AbnormalBatchSampler(
                #                labels,
                #                bal_abn_bag=0.5,
                #                bs=cfg_dload.bs,
                #                bal_abn_set=bal_abn_set,
                #                generator=NPRNG
                #            ),cfg_dload.bs, False) 
                }
            analyze_sampler(a, labels, cfg_ds.id, iters=1,vis=vis)
            dummy_train_loop(vis)
    elif bal_abn_bag > 0:
        analyze_sampler({f'abs_{bal_abn_bag}': bsampler}, labels, cfg_ds.id, iters=1,vis=None)
    
    dataloader = DataLoader( ds,
                        batch_sampler=bsampler,
                        #batch_size=cfg_dload.bs, shuffle=True, drop_last=True,
                        num_workers=cfg_dload.nworkers,
                        worker_init_fn=seed_sade,
                        #prefetch_factor=cfg_dload.pftch_fctr,
                        pin_memory=cfg_dload.pinmem, 
                        persistent_workers=cfg_dload.prstwrk 
                        )
    cfg.dataload.itersepo = len(dataloader)
    assert len(bsampler) == len(dataloader) 
    
    log.info(f"TRAIN **: {len(ds)}  {len(sampler)=}  {len(bsampler)=}=iters_per_epo  {len(dataloader)=}")
    return dataloader, DataCollator(cfg_dload.bs, cfg_dproc.seg.sel, cfg_dproc.crops2use.train)


####
class TrainDS(Dataset):
    def __init__(self, cfg_dload, cfg_ds, cfg_dproc, rgbflst, audflst):
        self.norm_lbl = cfg_ds.lbls.id[0]
        
        self.rgbflst = rgbflst
        self.audflst = audflst
        
        self.trnsfrm = FeatSegm(cfg_dproc.seg)
        log.debug(f'TRAIN TRNSFRM: {self.trnsfrm=}')
        
        self.frgb_ncrops = cfg_dproc.crops2use.train
        self.seg_len = cfg_dproc.seg.len
        self.seg_sel = cfg_dproc.seg.sel
        #self.l2n = cfg_dproc.l2n
        #self.rgbl2n = cfg_dproc.rgbl2n
        #self.audl2n = cfg_dproc.audl2n
        
        self.fsfeat = cfg_ds.frgb.fstep
        self.dfeat = cfg_ds.frgb.dfeat
        if audflst: self.peakboo_aud(cfg_ds.faud.dfeat)
        
        log.info(f'TrainDS ({self.frgb_ncrops} ncrops, {self.seg_len} maxseqlen/nsegments, {self.dfeat} feats)')    
        
        self.get_feat = {
            True: self.get_feat_wcrop,
            False: self.get_feat_wocrop
        }.get(self.frgb_ncrops)
        log.debug(f'TRAIN get_feat {self.get_feat}')
        
        ## here glance can be seen as metric to pick anomaly-present segments furhter down 
        self.glance = pd.read_csv(cfg_ds.glance)
        
        if cfg_dload.in2mem: 
            ## rnd state will b de same troughout epo iters
            ## or load full arrays int memory
            self.loadin2mem( cfg_dload.nworkers )
        else: self.in2mem = 0
            
    def peakboo_aud(self, dfeat):
        ## pekaboo AUD features
        assert len(self.rgbflst) == len(self.audflst)
        peakboo2 = f"{self.audflst[-1]}.npy"
        tojo2 = np.load(peakboo2)    
        assert dfeat == tojo2.shape[-1]
        self.dfeat += dfeat
        log.debug(f'TRAIN: AUD {osp.basename(peakboo2)} {np.shape(tojo2)}')    
    
    
    def load_data(self,idx): return *self.get_feat(idx), self.get_label(idx)
    
    def loadin2mem(self, nworkers):
        self.in2mem = 1
        log.info(f'LOADING TRAIN DS IN2 MEM'); t=time.time()
        with Pool(processes=nworkers) as pool:
            self.data = pool.map(self.load_data, range(len(self.rgbflst)))  
        log.info(f'COMPLETED TRAIN LOAD IN {hh_mm_ss(time.time()-t)}')
        
    
    def l2normfx(self, x, a=-1): return x / np.linalg.norm(x, ord=2, axis=a, keepdims=True)
    

    def get_label(self,idx):
        fn = osp.basename(self.rgbflst[int(idx)])
        if self.norm_lbl in fn: return torch.scalar_tensor(0) #np.float32(0.0)
        else: return torch.scalar_tensor(1) #np.float32(1.0)

    def get_pnt_lbl(self, idx, feat_len, idxs_seg):
        fn = osp.basename(self.rgbflst[int(idx)])
        points = self.glance.loc[self.glance['video-id'] == fn, 'glance'].values
        #log.error(f'{points=} {idxs_seg=}')
        
        pnt_lbl = np.zeros([self.seg_len], dtype=np.float32)
        if len(points) == 0: return pnt_lbl
        
        if len(idxs_seg.nonzero()[0]) == 0: ## pad
            for p in points:
                #idx_seg = int((p/16)/feat_len*self.seg_len)
                idx_seg = int((p/self.fsfeat)/feat_len*self.seg_len)
                log.debug(f"Frame {p} → Feature idx {p/self.fsfeat} → Segment idx {idx_seg}")
                if 0 <= idx_seg < self.seg_len:
                    pnt_lbl[idx_seg] = np.float32(1.0)
                    
        elif self.seg_sel == 'seq':
            # Filter points to only those in the extracted segment
            start_idx = idxs_seg[0]
            end_idx = idxs_seg[-1]
            assert len(idxs_seg) == self.seg_len 
            
            segment_start_frame = start_idx * self.fsfeat  # Convert feature idx to frame idx
            segment_end_frame = end_idx * self.fsfeat
            
            for p in points: # check if it falls in our segment
                if segment_start_frame <= p < segment_end_frame:
                    # Convert global frame index to segment-relative index
                    relative_idx = (p - segment_start_frame) / self.fsfeat
                    # Scale to seg_len
                    idx_seg = int(relative_idx / (end_idx - start_idx) * self.seg_len)
                    #if 0 <= idx_seg < self.seg_len:
                    pnt_lbl[idx_seg] = np.float32(1.0)
                    
        elif self.seg_sel == 'uni':
            for p in points: # find the nearest sampled index
                feat_idx = p / self.fsfeat  # Convert frame idx to feature idx
                # Find closest sampled index
                closest_idx = np.argmin(np.abs(idxs_seg - feat_idx))
                #if 0 <= closest_idx < self.seg_len:  # Safety check
                pnt_lbl[closest_idx] = 1
                    
        elif self.seg_sel == 'itp':
            #feat_len = feat.shape[-2]            
            for p in points:
                #idx_seg = int((p/16)/feat_len*self.seg_len)
                idx_seg = int((p/self.fsfeat)/feat_len*self.seg_len)
                log.debug(f"Frame {p} → Feature idx {p/self.fsfeat} → Segment idx {idx_seg}")
                #if 0 <= idx_seg < self.seg_len:
                pnt_lbl[idx_seg] = np.float32(1.0)
        
        #log.error(f'{pnt_lbl=}')
        return pnt_lbl
    

    ## (ncrops, len, dfeat)
    def get_feat_wcrop(self, idx):
        log.debug(f"**** {idx} ****")

        crop_lens, crop_data = [], []
        for crop_i in range(self.frgb_ncrops): ## !!
            # RGB
            rgb_fp_crop = f"{self.rgbflst[int(idx)]}__{crop_i}.npy"
            frgb_crop = np.load(rgb_fp_crop).astype(np.float32) 
            crop_lens.append(frgb_crop.shape[0])
            crop_data.append(frgb_crop)  
        
        ## some crops have different lens (i3drocng@ucf/xdv) !!!
        unified_len = max(set(crop_lens), key=crop_lens.count)

        feats = np.zeros((self.frgb_ncrops, self.seg_len, self.dfeat), dtype=np.float32)
        #for crop_i in range(self.frgb_ncrops):
        #    rgb_fp_crop = f"{self.rgbflst[int(idx)]}__{crop_i}.npy"
        #    frgb_crop = np.load(rgb_fp_crop).astype(np.float32)
        for crop_i, frgb_crop in enumerate(crop_data):
            log.debug(f'vid[{idx}][{crop_i}][RGB] {frgb_crop.shape} {frgb_crop.dtype}  {osp.basename(rgb_fp_crop)}')

            if frgb_crop.shape[0] != unified_len:
                log.debug(f"crop[{crop_i}] has different length ({frgb_crop.shape[0]}) than unified length ({unified_len}), handling outlier...")
                frgb_crop = frgb_crop[:unified_len]  # Trimming for now

            if crop_i == 0: idxs_trnsf = self.trnsfrm.get_idxs(unified_len)

            frgb_crop_seg = self.trnsfrm.fx(frgb_crop, idxs_trnsf['idxs'])
            log.debug(f'vid[{idx}][{crop_i}][RGB]: PST-SEG {frgb_crop_seg.shape}')

            ## AUD
            if self.audflst:
                if not crop_i: ## load
                    aud_fp = f"{self.audflst[int(idx)]}.npy"
                    faud = np.load(aud_fp).astype(np.float32)
                    log.debug(f'vid[{idx}][AUD] {faud.shape} {faud.dtype}  {osp.basename(aud_fp)}')
                    
                    ## there can be some mismatch betwehn both, if aud fext got diff window
                    ## mainly from CLIP feats
                    if 1 <= abs(faud.shape[0]-frgb_crop.shape[0]) <= 2:
                        ## as long as its only 2, leave as is, as let trnsfrm deal with it
                        log.debug(f'vid[{idx}][{crop_i}][AUD] : seg in2 {faud.shape}')
                    elif faud.shape[0] != frgb_crop.shape[0]:
                        raise ValueError(f'vid[{idx}][{crop_i}][AUD] : seg mismatch {faud.shape} {frgb_crop.shape}')
                    
                    faud_seg = self.trnsfrm.fx(faud, idxs_trnsf['idxs']) 
                    log.debug(f'vid[{idx}][{crop_i}][AUD] PST-SEG {faud_seg.shape}')
                    #if self.audl2n: faud = self.l2normfx(faud)
                    
                try: feat_crop = np.hstack((frgb_crop_seg, faud_seg))
                except: log.error(f'{frgb_crop_seg.shape} {faud_seg.shape}')
                log.debug(f'vid[{idx}][{crop_i}][MIX] {feat_crop.shape} {feat_crop.dtype}')
                
            else: feat_crop = frgb_crop_seg
            
            if idxs_trnsf['rnd_glob'] is not None:
                feats[crop_i] = feat_crop[idxs_trnsf['rnd_glob']]
            else: 
                feats[crop_i] = feat_crop

        if idxs_trnsf['idxs'] is None: ## selorpad
            seqlen = unified_len 
        else: ## interpolate 
            seqlen = self.seg_len
        
        ## TODO dev for seg.sel:
        pnt_lbl = self.get_pnt_lbl(idx, unified_len, idxs_trnsf['idxs'])
        
        log.debug(f'vid[{idx}] {feats.shape=} {feats.dtype} {seqlen=} {pnt_lbl.shape=}')
        return feats, seqlen, pnt_lbl, idxs_trnsf['idxs']

    ## (len, dfeat)
    def get_feat_wocrop(self,idx):
        log.debug(f"**** {idx} ****")
        
        rgb_fp = f"{self.rgbflst[int(idx)]}.npy"
        frgb = np.load(rgb_fp).astype(np.float32)
        log.debug(f"vid[{idx}][RGB] {frgb.shape} {frgb.dtype}  {osp.basename(rgb_fp)}")
        
        #if self.rgbl2n: frgb = self.l2normfx(frgb)
        ## TODO: pass points here, if seq/uni return idxs rndm w/ anom within
        idxs_trnsf = self.trnsfrm.get_idxs(frgb.shape[0])
        frgb_seg = self.trnsfrm.fx(frgb, idxs_trnsf['idxs'])
        log.debug(f'vid[{idx}][RGB] PST-SEG {frgb_seg.shape}')
        
        if self.audflst:
            aud_idx = int(idx)
            
            aud_fp = f"{self.audflst[aud_idx]}.npy"
            faud = np.load( aud_fp ).astype(np.float32)
            #if self.audl2n: faud = self.l2normfx(faud)
            log.debug(f'vid[{idx}][AUD] {faud.shape} {faud.dtype}  {osp.basename(aud_fp)}')
            
            #############################
            ## use FeatComp class 
            if faud.shape[0] != frgb.shape[0]:
                #log.debug(f'preseg {np.mean(faud,axis=0)}')
                #for af in faud[:16]: log.debug(f'{af[:40]}')
                #faud = self.self.trnsfrm.segmentation(faud, frgb.shape[0], None) 
                #log.debug(f'posseg {np.mean(faud,axis=0)}')
                #for af in faud[:16]: log.debug(f'{af[:40]}')
                log.debug(f'vid[{idx}][AUD] mismatch AUD RGB {faud.shape}')
            
            faud_seg = self.trnsfrm.fx(faud, idxs_trnsf['idxs']) 
            feats = np.hstack((frgb_seg, faud_seg))
            log.debug(f'vid[{idx}][AUD] PST-SEG {faud_seg.shape}')
            log.debug(f'vid[{idx}][MIX] {feats.shape} {feats.dtype}')

        else: feats = frgb_seg     

        if idxs_trnsf['rnd_glob'] is not None:
            feats = feats[idxs_trnsf['rnd_glob']]
            
        if idxs_trnsf['idxs'] is None: ## padded
            seqlen = frgb.shape[0]
        else: ## interpolate
            seqlen = self.seg_len
        
        ## TODO: dev for segsel uni 
        pnt_lbl = self.get_pnt_lbl(idx, frgb.shape[0], idxs_trnsf['idxs'])
        
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype} {seqlen=} {pnt_lbl.shape} {idxs_trnsf["idxs"]=}')
        return feats, seqlen, pnt_lbl, idxs_trnsf['idxs']
    
    
    def __getitem__(self, idx):
        ## as n normal videos > abnormal
        ## if idx out of range, pick a random idx
        #if idx >= len(self.rgbflst):
        #    idx = np.random.randint(0,len(self.rgbflst))
        #    idx = NPRNG.integers(feat_len-self.seg_len)
            
        if self.in2mem:
            feats, seqlen, pnt_lbl, idxs_seg, label = self.data[int(idx)]  
        else:
            label = self.get_label(idx)
            feats, seqlen, pnt_lbl, idxs_seg = self.get_feat(idx)
            
        #log.debug(f'f[{idx}]: {feats.shape}')    
        return feats, seqlen, pnt_lbl, idxs_seg, label

    def __len__(self):
        return len(self.rgbflst)


##############
## PREP / FRMT 
## 1) Segmentation prepares the features matrix for each .npy
class FeatSegm():
    def __init__(self, cfg, seed=None):
        assert cfg.sel in ['itp','uni','seq']
        self.sel = cfg.sel
        self.len = cfg.len
        self.jit = cfg.jit
        self.rnd = cfg.rnd
        
        self.fx = self.interpolate if cfg.sel == 'itp' else self.sel_or_pad 
        log.info(f"FeatSegm w {self.fx=}")
        
        #self.intplt = cfg.intplt
        #self.fx = {
        #    1: self.interpolate,
        #    0: self.sel_or_pad
        #}.get(cfg.intplt)
        #self.RNG = np.random.default_rng(seed)
        #self.rng = NPRNGZ              
        #log.error(f" {np.random.default_rng.}")
    
    def get_idxs(self, feat_len, points=None):
        if self.sel == 'itp': ## special case for avg adjcent linspace
            idxs = np.linspace(0, feat_len, self.len+1, dtype=np.int32)
            idxs = self.rnd_jit(idxs, feat_len)
            log.debug(f"FSeg: grabbed intrplt {len(idxs)} idxs")
            
        else:
            if feat_len <= self.len: ## latter pad
                #idxs = None
                idxs = np.zeros(self.len, dtype=np.int32)
                log.debug(f"grabbed None idxs")
                
            elif self.sel == 'uni': ## differ from intplt seq
                idxs = np.linspace(0, feat_len-1, self.len, dtype=np.int32)
                idxs = self.rnd_jit(idxs, feat_len)
                log.debug(f"FSeg: grabbed uni {len(idxs)} idxs")
                    
            elif self.sel == 'seq':
                start = NPRNG.integers(0, feat_len-self.len)
                if points: ## pick a random start assuring a anom within
                    # Convert frame points to feature indices
                    point_indices = [p / self.fsfeat for p in points]
                    # Find valid starting positions that would include at least one anomaly
                    valid_starts = []
                    for p_idx in point_indices:
                        # Calculate range of starting positions that would include this point
                        earliest_start = max(0, int(p_idx - self.len + 1))
                        latest_start = min(int(p_idx), feat_len - self.len)
                        
                        if latest_start >= earliest_start:
                            valid_starts.extend(range(earliest_start, latest_start + 1))
                    
                    if valid_starts:
                        # Randomly select from valid starts
                        start = NPRNG.choice(valid_starts)
                
                idxs = np.arange(start, start+self.len, dtype=np.int32)
                log.debug(f"FSeg: grabbed seq {len(idxs)} idxs")
                
        idxs_glob = None
        if self.rnd:
            ## aply glob rnd only after self.trnsfrm.fx is call
            idxs_glob = np.arange(self.len)
            NPRNG.shuffle(idxs_glob)
            #log.debug(f"GLOB / NEW {idxs_glob=}")
            #return feat[idxx]
            log.debug(f"FSeg: grabbed rnd_glob {len(idxs_glob)} idxs")
        
        return {
            'idxs': idxs,
            'rnd_glob': idxs_glob,
        }
        
    def rnd_jit(self, idxx, feat_len):
        ## jitter betwen adjacent chosen idxx
        ## only when theres no repetead idxs
        ## taken from MIST random_peturb
        if self.jit:
            if feat_len > self.len:
                #log.debug(f'JIT / OLD {idxx=}')
                for i in range(self.len):
                    if i < self.len - 1:
                        if idxx[i] != idxx[i+1]: ## contemple if sel != seq
                            idxx[i] = NPRNG.choice(range(idxx[i], idxx[i+1]))  
                #log.debug(f'JIT / NEW {idxx=}')
        return idxx
    
    def interpolate(self, feat, idxs):
        #log.debug(f"FSeg interpolate")
        #if isinstance(idxs) is int: ## aud temp dim align
        #    idxs = np.linspace(0, len(feat), self.len+1, dtype=np.int32)
        
        new_feat = np.zeros((self.len, feat.shape[1]), dtype=np.float32)
        for i in range(self.len):
            #log.debug(f"{fn} {crop}")
            if idxs[i] != idxs[i+1]:
                new_feat[i, :] = np.mean(feat[idxs[i]:idxs[i+1], :], axis=0)
                #new_feat[i, :] = np.sort(feat[idxs[i]:idxs[i+1], :], axis=0)[0]
            else:
                new_feat[i, :] = feat[idxs[i], :]
        return new_feat

    def sel_or_pad(self, feat, idxs):
        ## https://github.com/Roc-Ng/XDVioDet/blob/master/utils.py
        def pad(feat):
            if np.shape(feat)[0] <= self.len:
                return np.pad(feat, ((0, self.len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
            else: return feat
        
        #if len(idxs) == 0:
        #if idxs is None:
        if len(idxs.nonzero()[0]) == 0:
            #log.debug(f"FSeg selorpad padding")
            return pad(feat)
        else:
            #log.debug(f"FSeg selorpad indexing")
            ## make use of idxs already pre seleted at crop0
            ## find a way to modulate each crop differently
            ## ovverride given idxs
            #if cfg.RNDCROP:
            #    log.error(f"{cfg.RNDCROP} in dev")
            #    #idxs = self.get_idxs(len(feat))['idxs']

            #assert len(feat) == len(idxs)
            #log.error(f"{feat.shape} {len(idxs)} {idxs}")    
            ## diffrence ??
            f = feat[ idxs ]
            assert f.shape == feat[idxs, :].shape

            return f    

## 2) pass to collate_fn
class DataCollator:
    def __init__(self, bs, seg_sel, ncrops):
        self.bs = bs
        self.ncrops = ncrops
        self.seg_sel = seg_sel
        
    def _rshp_in(self, x):    
        if x.ndim == 4: 
            bs, ncrops, slen, nfeat = x.shape
            x = x.view(-1, slen, nfeat) 
        else: assert x.ndim == 3 
        return x    
    
    def __call__(self, tdata, trn_inf):  
        ldata={}
        cfeat, seqlen, pnt_lbl, idxs_seg, label = tdata
        
        #if self.seg_sel != 'itp':
        ldata["seqlen"] = seqlen.to(trn_inf['dvc'])
        ldata["label"] = label.to(trn_inf['dvc'])
        ldata["point_label"] = pnt_lbl.to(trn_inf['dvc']) #glance
        ldata["step"] = trn_inf['step'] #glance
        ldata["idxs_seg"] = idxs_seg #glance
        
        log.debug(f"E[{trn_inf['epo']}]B[{trn_inf['bat']}]S[{trn_inf['step']}][{self.seg_sel}] feat: {list(cfeat.shape)}, seqlen: {list(seqlen.shape)} {seqlen}, lbl: {label} {list(label.shape)} {label.device}")
        return self._rshp_in(cfeat).to(trn_inf['dvc']), ldata

