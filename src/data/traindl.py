import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, WeightedRandomSampler
from pytorch_metric_learning.samplers import MPerClassSampler
from src.data.samplers import MPerClassSampler as MPerClassSamplerZu
import numpy as np

import glob , os, os.path as osp , math , time
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


def get_sampler(cfg_dproc, cfg_dload, len_abn, len_nor):
    
    max_len = max( len_abn , len_nor)
    #min_len = min( len_abn , len_nor)
    samples_per_epoch = int(cfg_dproc.sampler.len * max_len ) #len(ds)
    bal_wgh = cfg_dproc.sampler.balance
    if bal_wgh == 0.5: ## sultani        
        sampler = MPerClassSampler(labels, 
                                m=cfg_dload.bs//2, 
                                batch_size=cfg_dload.bs, 
                                length_before_new_iter=len(rgbfl)
                                ) 
        #sampler = MPerClassSamplerZu(labels, 
        #                        m=cfg_dload.bs//2, 
        #                        batch_size=cfg_dload.bs, 
        #                        #length_before_new_iter=samples_per_epoch ## no use atm
        #                        ) 
    elif bal_wgh == 0: ## rocng
        sampler = RandomSampler(ds, 
                            replacement=False,
                            num_samples=samples_per_epoch, # !!!! set intern to len(ds)
                            generator=TCRNG
                            )
    elif 0 < bal_wgh < 1:
        rebal_wgh = 0.5 - ( 1 - (len_nor/len_abn) )
        weights = [rebal_wgh]*len_abn + [1-rebal_wgh]*len_nor
        sampler = WeightedRandomSampler(weights, 
                            replacement=False,
                            num_samples=samples_per_epoch, 
                            generator=TCRNG
                            )
        log.warning(f'TRAIN: WeightSammpler {rebal_wgh=} {len(weights)=}')    
    else: raise ValueError 
    log.info(f"TRAIN: SMPLR {sampler=} w/ {len(sampler)} [{bal_wgh=} {samples_per_epoch=}]")


def get_trainloader(cfg):
    from ._data import FeaturePathListFinder, debug_cfg_data, analyze_sampler
    if cfg.get("debug"): debug_cfg_data(cfg)
    
    log.info(f'TRAIN: getting trainloader')

    cfg_ds = cfg.data
    cfg_dload = cfg.dataload.train
    cfg_dproc = cfg.dataproc
    
    get_rng(cfg.seed)
    
    rgbfplf = FeaturePathListFinder(cfg, 'train', 'rgb')
    argbfl, nrgbfl = rgbfplf.get('ANOM'), rgbfplf.get('NORM')
    len_abn = len(argbfl); len_nor = len(nrgbfl)
    log.info(f'TRAIN: RGB {len_nor} normal, {len_abn} abnormal') 
    
    aaudfl, naudfl = [], []
    if cfg_ds.get('faud'):
        audfplf = FeaturePathListFinder(cfg, 'train', 'aud', auxrgbflist=argbfl+nrgbfl)
        aaudfl, naudfl = audfplf.get('ANOM'),  audfplf.get('NORM')
        log.info(f'TRAIN: AUD on {len(naudfl)} normal, {len(aaudfl)} abnormal')    
    
    rgbfl = argbfl + nrgbfl
    audfl = aaudfl + naudfl
    labels = [1]*len_abn + [0]*len_nor
    
    ds = TrainDS(cfg_dload, cfg_ds, cfg_dproc, rgbfl, audfl)  
    
    sampler = get_sampler(cfg_dproc, cfg_dload, len_abn, len_nor )
    ## safe to set droplast true for mperclass
    bsampler = BatchSampler(sampler , cfg_dload.bs , True) #cfg_dload.droplast
    #if cfg.get("debug"): 
    analyze_sampler(bsampler, labels, cfg_ds.id, iters=1)
    
    dataloader = DataLoader( ds,
                        batch_sampler=bsampler,
                        #batch_size=cfg_dload.bs, shuffle=True, drop_last=True,
                        num_workers=cfg_dload.nworkers,
                        worker_init_fn=seed_sade,
                        #prefetch_factor=cfg_dload.pftch_fctr,
                        pin_memory=cfg_dload.pinmem, 
                        persistent_workers=cfg_dload.prstwrk 
                        )
    cfg.dataload.train.itersepo = len(dataloader)
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
        
        self.cropasvideo = cfg_dproc.cropasvideo
        self.frgb_ncrops = cfg_dproc.crops2use.train
        self.seg_len = cfg_dproc.seg.len
        #self.l2n = cfg_dproc.l2n
        #self.rgbl2n = cfg_dproc.rgbl2n
        #self.audl2n = cfg_dproc.audl2n
        
        self.dfeat = cfg_ds.frgb.dfeat
        if audflst: self.peakboo_aud(cfg_ds.faud.dfeat)
        
        log.info(f'TRAIN ({self.frgb_ncrops} ncrops, {self.seg_len} maxseqlen/nsegments, {self.dfeat} feats')    
        log.info(f'TRAIN cropasvideo is {self.cropasvideo}')

        self.get_feat = {
            True: self.get_feat_wcrop,
            False: self.get_feat_wocrop
        }.get(self.frgb_ncrops and not self.cropasvideo)
        log.info(f'TRAIN get_feat {self.get_feat}')

        if cfg_dload.in2mem: 
            ## rnd state will b de same troughout epo iters
            ## or load full arrays int memory
            self.loadin2mem( cfg_dload.nworkers )
        else: self.in2mem = 0
            
    def peakboo_aud(self, dfeat):
        ## pekaboo AUD features
        if not self.cropasvideo:
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
        #if 'label_A' in self.rgbflst[int(idx)]: return int(0)
        #else: return int(1)
        if self.norm_lbl in self.rgbflst[int(idx)]: return np.float32(0.0)
        else: return np.float32(1.0)
    
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

        if idxs_trnsf['idxs'] is None: ## pad
            seqlen = unified_len 
        else: ## interpolate 
            seqlen = self.seg_len
        
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype} {seqlen=}')
        return feats, seqlen

    ## (len, dfeat)
    def get_feat_wocrop(self,idx):
        log.debug(f"**** {idx} ****")
        
        rgb_fp = f"{self.rgbflst[int(idx)]}.npy"
        frgb = np.load(rgb_fp).astype(np.float32)
        log.debug(f"vid[{idx}][RGB] {frgb.shape} {frgb.dtype}  {osp.basename(rgb_fp)}")
        
        #if self.rgbl2n: frgb = self.l2normfx(frgb)
        idxs_trnsf = self.trnsfrm.get_idxs(frgb.shape[0])
        frgb_seg = self.trnsfrm.fx(frgb, idxs_trnsf['idxs'])
        log.debug(f'vid[{idx}][RGB] PST-SEG {frgb_seg.shape}')
        
        if self.audflst:
            if self.cropasvideo: aud_idx = int(idx)//self.frgb_ncrops
            else: aud_idx = int(idx)
            
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
        
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype} {seqlen=}')
        return feats, seqlen
    
    
    def __getitem__(self, idx):
        ## as n normal videos > abnormal
        ## if idx out of range, pick a random idx
        #if idx >= len(self.rgbflst):
        #    idx = np.random.randint(0,len(self.rgbflst))
        #    idx = NPRNG.integers(feat_len-self.seg_len)
            
        if self.in2mem:
            feats, seqlen, label = self.data[int(idx)]  
        else:
            label = self.get_label(idx)
            feats, seqlen = self.get_feat(idx)
            
        #log.debug(f'f[{idx}]: {feats.shape}')    
        return feats, seqlen, label

    def __len__(self):
        return len(self.rgbflst)


################################
## PREP / FRMT 
## preprocessing of train input

## 1) FeatSegm prepares the features matrix for each .npy
class FeatSegm():
    def __init__(self, cfg, seed=None):
        assert cfg.sel in ['itp','uni','seq']
        self.sel = cfg.sel
        self.len = cfg.len
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
    
    def get_idxs(self, feat_len):
        if self.sel == 'itp': ## special case for avg adjcent linspace
            idxs = np.linspace(0, feat_len, self.len+1, dtype=np.int32)
            idxs = self.rnd_jit(idxs, feat_len)
            log.debug(f"FSeg: grabbed intrplt {len(idxs)} idxs")
            
        else:
            if feat_len <= self.len: ## latter pad
                #idxs = []
                idxs = None
                log.debug(f"grabbed None idxs")
                
            elif self.sel == 'uni': ## differ from intplt seq
                idxs = np.linspace(0, feat_len-1, self.len, dtype=np.int32)
                idxs = self.rnd_jit(idxs, feat_len)
                log.debug(f"FSeg: grabbed uni {len(idxs)} idxs")
                    
            elif self.sel == 'seq': ## start point
                start = NPRNG.integers(0, feat_len-self.len)
                idxs = list(range(start, start+self.len))
                log.debug(f"FSeg: grabbed seq {len(idxs)} idxs")
                
        #idxs_glob = []
        idxs_glob = None
        if 'glob' in self.rnd:
            ## aply glob rnd only after self.trnsfrm.fx is call
            idxs_glob = np.arange(self.len)
            NPRNG.shuffle(idxs_glob)
            #log.debug(f"GLOB / NEW {idxs_glob=}")
            #return feat[idxx]
            log.debug(f"FSeg: grabbed rnd_glob {len(idxs_glob)} idxs")
        
        return {
            'idxs': idxs,
            'rnd_glob': idxs_glob
        }
        
    def rnd_jit(self, idxx, feat_len):
        ## jitter betwen adjacent chosen idxx
        ## only when theres no repetead idxs
        ## taken from MIST random_petrub
        if 'jit' in self.rnd:
            if feat_len > self.len:
                #log.debug(f'JIT / OLD {idxx=}')
                for i in range(self.len):
                    if i < self.len - 1:
                        if idxx[i] != idxx[i+1]: ## contemple if sel != seq
                            idxx[i] = self.RNG.choice(range(idxx[i], idxx[i+1]))  
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
        if idxs is None:
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
        cfeat, seqlen, label = tdata
        
        if self.seg_sel != 'itp': ## it may come w/ pad
            ldata["seqlen"] = seqlen.to(trn_inf['dvc'])
        ldata["label"] = label.to(trn_inf['dvc'])
        
        log.debug(f"E[{trn_inf['epo']}]B[{trn_inf['bat']}][{self.seg_sel}] feat: {cfeat.shape} ,seqlen: {seqlen.shape} {seqlen}, lbl: {label} {label.shape} {label.device}")
        return self._rshp_in(cfeat).to(trn_inf['dvc']), ldata



################################
#ads = TrainDS(cfg_dload, cfg_ds, cfg_dproc, argbfl, aaudfl, 'ABNORMAL')
#nds = TrainDS(cfg_dload, cfg_ds, cfg_dproc, nrgbfl, naudfl, 'NORMAL')
#
#max_len = max(len(ads), len(nds))
#log.debug(f'TRAIN: {max_len=} {bs=}') #iterations_per_epoch {max_len // (bs // 2)=}
### as abnormal/normal not balanced set replacement True for smaller ds
### handled in TrainDS/_get_item_
###is_oversampling = len(ds) < max_len
#samplers = [
#    RandomSampler(ds, 
#                replacement=len(ds) < max_len, 
#                num_samples=max_len if len(ds) < max_len else None, 
#                generator=TCRNG) 
#    for ds in [ads, nds]  # Abnormal, then Normal
#]
#
#dataloaders = [
#    DataLoader(
#        ds, 
#        batch_sampler=BatchSampler(sampler, batch_size=cfg_dload.bs//2, drop_last=cfg_dload.droplast),
#        num_workers=cfg_dload.nworkers,
#        worker_init_fn=seed_sade,
#        #prefetch_factor=cfg_dload.pftch_fctr,
#        pin_memory=cfg_dload.pinmem, 
#        persistent_workers=cfg_dload.prstwrk 
#    )
#    for ds, sampler in [(ads, samplers[0]), (nds, samplers[1])]  # Abnormal then Normal  
#]
#cfg.dataload.train.itersepo = max( len(dataloaders[0]), len(dataloaders[1]))
#log.debug(f'TRAIN: iterations_per_epoch {cfg.dataload.train.itersepo}')
'''
def sequencial(self, feat, crop=0, fn=''):
        
    def unfrm_extrt(feat):
        idxs = np.linspace(0, len(feat)-1, self.len, dtype=np.int32)
        #idxs = np.linspace(0, len(feat)-1, SLEN+1 , dtype=np.int32)
        #if 'jit' in self.rnd: idxs = self.rnd_jit(idxs, len(feat))
        #idxs = idxs[:-1]            
        return feat[idxs, :]
    
    def rnd_extrt(feat):
        #start = np.random.randint(len(feat)-self.len)
        start = self.RNG.integers(len(feat)-self.len)
        return feat[start:start+self.len]

    def pad(feat):
        if np.shape(feat)[0] <= self.len:
            return np.pad(feat, ((0, self.len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
        else: return feat

    if len(feat) > self.len:
        
        if self.rnd[0] == 'uni': f = unfrm_extrt(feat)
        elif self.rnd[0] == 'rnd': f = rnd_extrt(feat)
        
        ## assert the irrgularity betwen crops of rnd
        ## and make it chossable
        ## maybe trough a dict
        #if 'glob' in self.rnd: f = self.rnd_glob(f)
        
        return f    
            
    else: return pad(feat)


def segment_feat_crop(feat, length ):
    """
        segments (ncrops,ts,feats) into (ncrops,length,feats)
    """
    nc, t, f = feat.shape
    divided_features = []        
    for idxf, f in enumerate(feat): 
        new_f = np.zeros((length, f.shape[1]) , dtype=np.float32)
        print(f'{idxf} {new_f.shape}')
        ## len(f) = t
        r = numpy.linspace(0, len(f), length+1, dtype=numpy.int32)
        for i in range(length):
            if r[i]!=r[i+1]:
                new_f[i,:] = np.mean(f[r[i]:r[i+1],:], 0)
            else:
                new_f[i,:] = f[r[i],:]
        divided_features.append(new_f)
    return np.array(divided_features, dtype=np.float32)

    new_feat = np.zeros((self.len, feat.shape[1]), dtype=np.float32)
    idxs = np.linspace(0, len(feat), self.len+1, dtype=np.int32)
    
    #if 'jit' in self.rnd:
    #    if crop == 1: ## atual vid calling frist time, store jit_idxxs
    #        log.error(f"{fn} {crop}")
    #        self.jit_idxx = idxs = self.rnd_jit(idxs, len(feat))
    #        log.error(f'{idxs=}')
    #    elif crop != 0: ## consequent
    #        idxs = self.jit_idxx
    #        log.warning(f"{fn} {crop}")
    #        log.warning(f'{idxs=}')
    #    else: ## only call for vid in question, no need to store
    #        idxs = self.rnd_jit(idxs, len(feat))    
        
        
    for i in range(self.len):
        #log.debug(f"{fn} {crop}")
        if idxs[i] != idxs[i+1]:
            new_feat[i, :] = np.mean(feat[idxs[i]:idxs[i+1], :], axis=0)
        else:
            new_feat[i, :] = feat[idxs[i], :]
    
    #if 'glob' in self.rnd: new_feat = self.rnd_glob(new_feat)
        
    return new_feat
'''


'''

    def get_feat_wcrop(self,idx):
        log.debug(f"**** {idx} ****")
        
        ## Determine unified length from crop0
        
        crop0_len = np.load(f"{self.rgbflst[int(idx)]}__0.npy").astype(np.float32).shape[0]
        
        feats = np.zeros((self.frgb_ncrops, self.len, self.dfeat), dtype=np.float32)
        for i in range(self.frgb_ncrops):
            
            ## RGB
            rgb_fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
            frgb_crop = np.load(rgb_fp_crop).astype(np.float32)
            log.debug(f'vid[{idx}][{i}][RGB] {frgb_crop.shape} {frgb_crop.dtype}  {osp.basename(rgb_fp_crop)}')
            
            if not i: ## crop 0
                ## peakaboo next
                lens.append(frgb_crop.shape[0])
                len_next = np.load(f"{self.rgbflst[int(idx)]}__{i+1}.npy").astype(np.float32).shape[0]
                if lens[0] != len_next:
                    frgb_crop = frgb_crop[:len_next]
                idxs_trnsf = self.trnsfrm.get_idxs( len(frgb_crop) )
            
            ## control the indexing in selorpad
            ## as interpolate impact will be low (ithink)
            elif lens[0] != frgb_crop.shape[0]:
                log.warning(f"crop[{i}] w dif len")
            
            #if self.rgbl2n: frgb_crop = self.l2normfx(frgb_crop)
            #if idx == 0: view_feat(feat_crop.asnumpy())
                
            ## segment prior to hstack
            frgb_crop_seg = self.trnsfrm.fx(frgb_crop, idxs_trnsf['idxs']) 
            log.debug(f'vid[{idx}][{i}][RGB]: PST-SEG {frgb_crop_seg.shape}')
            
            ## AUD
            if self.audflst:
                if not i: ## load
                    aud_fp = f"{self.audflst[int(idx)]}.npy"
                    faud = np.load(aud_fp).astype(np.float32)
                    log.debug(f'vid[{idx}][AUD] {faud.shape} {faud.dtype}  {osp.basename(aud_fp)}')
                    
                    ## there can be some mismatch betwehn both, if aud fext got diff window
                    ## mainly from CLIP feats
                    if 1 <= abs(faud.shape[0]-frgb_crop.shape[0]) <= 2:
                        ## as long as its only 2, leave as is, as let trnsfrm deal with it
                        log.debug(f'vid[{idx}][{i}][AUD] : seg in2 {faud.shape}')
                    elif faud.shape[0] != frgb_crop.shape[0]:
                        raise ValueError(f'vid[{idx}][{i}][AUD] : seg mismatch {faud.shape} {frgb_crop.shape}')
                    
                    faud_seg = self.trnsfrm.fx(faud, idxs_trnsf['idxs']) 
                    log.debug(f'vid[{idx}][{i}][AUD] PST-SEG {faud_seg.shape}')
                    #if self.audl2n: faud = self.l2normfx(faud)
                    
                try: feat_crop = np.hstack((frgb_crop_seg, faud_seg))
                except: log.error(f'{frgb_crop_seg.shape} {faud_seg.shape}')
                log.debug(f'vid[{idx}][{i}][MIX] {feat_crop.shape} {feat_crop.dtype}')
                
            else: feat_crop = frgb_crop_seg

            feats[i] = feat_crop
            
            ## rndmiz temporal order
            #if len(idxs_trnsf['rnd_glob']) != 0:
            if idxs_trnsf['rnd_glob'] is not None:
                ## rnd ovre same idxs per crop
                feats[i] = feats[i, idxs_trnsf['rnd_glob']]
            
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype}')
        return feats
'''