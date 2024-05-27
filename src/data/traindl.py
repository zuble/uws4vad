import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
import numpy as np

import glob , os, os.path as osp , math , time
from multiprocessing import Pool

from src.utils import hh_mm_ss, seed_sade, logger
log = logger.get_log(__name__)
    

## dirt .. needed ? 
NPRNG = None
TCRNG = None
def get_rng(seed):
    NPRNG = np.random.default_rng(seed)
    ## https://pytorch.org/docs/1.8.1/generated/torch.Generator.html#torch.Generator
    TCRNG = torch.Generator().manual_seed(seed)


#############################
## TRAIN
class Zuader:
    ## encapsulates the dataloader independtly of elements yielded
    def __init__(self, mil, *loaders):
        self.mil = mil
        self.loaders = loaders
    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]
        return self
    def __next__(self):
        if self.mil: return next(self.loader_iters[0]), next(self.loader_iters[1])
        elif not self.mil: return next(self.loader_iters[0])
        else: raise StopIteration


def get_trainloader(cfg):
    from ._data import FeaturePathListFinder
    log.info(f'TRAIN: getting trainloader')

    cfg_ds = cfg.dl.ds
    cfg_loader = cfg.dl.loader.train
    cfg_trnsfrm = cfg.dl.trnsfrm.train
    
    get_rng(cfg.seed)
    
    if cfg_ds.get('faud'):
        audfplf = FeaturePathListFinder(cfg.dl, 'train', 'AUD')
        aaudfl, naudfl = audfplf.get('ANOM'),  audfplf.get('NORM')
        log.info(f'TRAIN: AUD on {len(naudfl)} normal, {len(aaudfl)} abnormal')    
    else: aaudfl, naudfl = [], []
    
    rgbfplf = FeaturePathListFinder(cfg.dl, 'train', 'RGB')
    argbfl, nrgbfl = rgbfplf.get('ANOM'), rgbfplf.get('NORM')
    log.info(f'TRAIN: RGB {len(nrgbfl)} normal, {len(argbfl)} abnormal') 
    
    if cfg_trnsfrm.mil:
        
        log.info(f"MIL ITS ON")
        ads = TrainDS(cfg_loader, cfg_ds, cfg_trnsfrm, argbfl, aaudfl, 'ABNORMAL')
        nds = TrainDS(cfg_loader, cfg_ds, cfg_trnsfrm, nrgbfl, naudfl, 'NORMAL')
        
        arsampler, nrsampler, cfg.train.epoch_nbatch = get_milRS(cfg_loader.bs, ads, nds)
        ald = DataLoader(ads, 
                        batch_sampler=BatchSampler(arsampler, batch_size=cfg_loader.bs//2 , drop_last=cfg_loader.droplast),
                        num_workers=cfg_loader.nworkers,
                        worker_init_fn=seed_sade,
                        prefetch_factor=cfg_loader.pftch_fctr,
                        pin_memory=cfg_loader.pinmem, 
                        persistent_workers=cfg_loader.prstwrk 
                        ) 
        nld = DataLoader(nds, 
                        batch_sampler=BatchSampler(nrsampler, batch_size=cfg_loader.bs//2 , drop_last=cfg_loader.droplast),
                        num_workers=cfg_loader.nworkers,
                        worker_init_fn=seed_sade,
                        prefetch_factor=cfg_loader.pftch_fctr,
                        pin_memory=cfg_loader.pinmem, 
                        persistent_workers=cfg_loader.prstwrk 
                        )
        return Zuader(True, nld, ald), TrainFrmter(cfg_loader.bs, cfg_trnsfrm)
    
    else:
        rgbfl = argbfl + nrgbfl
        audfl = aaudfl + naudfl
        log.info(f'TRAIN: RGB {len(rgbfl)} feats')
        
        ds = TrainDS(cfg_loader, cfg_ds, cfg_trnsfrm, rgbfl, audfl)  
        rsampler = get_seqRS(cfg_loader.bs, ds)
        ld0 = DataLoader( ds,
                            batch_sampler= BatchSampler(rsampler , cfg_loader.bs , cfg_loader.droplast),
                            num_workers=cfg_loader.nworkers,
                            worker_init_fn=seed_sade,
                            prefetch_factor=cfg_loader.pftch_fctr,
                            pin_memory=cfg_loader.pinmem, 
                            persistent_workers=cfg_loader.prstwrk 
                            )
        return Zuader(False, ld0), TrainFrmter(cfg_loader.bs, cfg_trnsfrm)


def get_milRS(bs, ads, nds):
    
    def get_rnd_smp(ds, nsamples, gen):
        is_oversampling = len(ds) < nsamples
        log.info(f"{is_oversampling = }")
        return RandomSampler(ds, 
                            replacement=is_oversampling, 
                            num_samples=nsamples if is_oversampling else None, 
                            generator=gen)
        
    maxlends = max(len(ads),len(nds))
    epoch_nbatch = maxlends // (bs // 2) #math.ceil(maxlends / cfg.TRAIN.BS)
    #cfg.TRAIN.EPOCHBATCHS = epoch_nbatch
    log.error(f'TRAIN: {maxlends=} {bs=}  -> epoch_nbatch not set , set in future trainer')
    
    #gen = torch.Generator().manual_seed(cfg.SEED) #log.error(f"{gen.initial_seed()}")
    
    ## as abnormal/normal not balanced set replacement True for smaller ds
    log.info(f'TRAIN: {len(ads)=}')
    arsampler = get_rnd_smp(ads, maxlends, TCRNG)
    log.info(f'TRAIN: {len(nds)=}')
    nrsampler = get_rnd_smp(nds, maxlends, TCRNG)
    
    return arsampler, nrsampler, epoch_nbatch

def get_seqRS(bs, ds):
    epoch_nbatch = len(ds) // bs
    #cfg.TRAIN.EPOCHBATCHS = epoch_nbatch
    log.error(f'TRAIN: {len(ds)=} {bs=}  -> epoch_nbatch not set , set in future trainer')
    
    #gen = torch.Generator().manual_seed(cfg.seed)
    return RandomSampler(ds, generator=TCRNG), epoch_nbatch


class TrainDS(Dataset):
    def __init__(self, cfg_loader, cfg_ds, cfg_trnsfrm, rgbflst, audflst, xtra=''):
        self.xtra = xtra
        self.norm_lbl = cfg_ds.info.lbls[0]
        
        self.rgbflst = rgbflst
        self.audflst = audflst
        
        self.trnsfrm = FeatSegm(cfg_trnsfrm)
        log.info(f'TRAIN TRNSFRM: {self.trnsfrm=}')
        
        self.cropasvideo = cfg_trnsfrm.cropasvideo
        self.rgb_ncrops = cfg_trnsfrm.crops2use
        self.len = cfg_trnsfrm.len
        #self.l2n = cfg_trnsfrm.l2n
        #self.rgbl2n = cfg_trnsfrm.rgbl2n
        #self.audl2n = cfg_trnsfrm.audl2n
        
        self.nfeats = cfg_ds.frgb.nfeats
        if audflst: self.peakboo_aud(cfg_ds.faud.nfeats)
        
        log.info(f'TRAIN {xtra}: ({self.rgb_ncrops} ncrops, {self.len} maxseqlen/nsegments, {self.nfeats} feats')    
        log.info(f'TRAIN {xtra}: cropasvideo is {self.cropasvideo}')

        self.get_feat = {
            True: self.get_feat_wcrop,
            False: self.get_feat_wocrop
        }.get(self.rgb_ncrops and not self.cropasvideo)

        if cfg_loader.in2mem: 
            ## rnd state will b de same troughout epo iters
            ## or load full arrays int memory
            self.loadin2mem( cfg_loader.nworkers )
        else: self.in2mem = 0
            
    def peakboo_aud(self, nfeats):
        ## pekaboo AUD features
        if not self.cropasvideo:
            assert len(self.rgbflst) == len(self.audflst)
        peakboo2 = f"{self.audflst[-1]}.npy"
        tojo2 = np.load(peakboo2)    
        assert nfeats == tojo2.shape[-1]
        self.nfeats += nfeats
        log.info(f'TRAIN: AUD {osp.basename(peakboo2)} {np.shape(tojo2)}')    
    
    
    def load_data(self,idx): return self.get_feat(idx), self.get_label(idx)
    
    def loadin2mem(self, nworkers):
        self.in2mem = 1
        log.info(f'LOADING TRAIN {self.xtra} DS IN2 MEM'); t=time.time()
        with Pool(processes=nworkers) as pool:
            self.data = pool.map(self.load_data, range(len(self.rgbflst)))  
        log.info(f'COMPLETED TRAIN {self.xtra} LOAD IN {hh_mm_ss(time.time()-t)}')
        
    
    def l2normfx(self, x, a=-1): return x / np.linalg.norm(x, ord=2, axis=a, keepdims=True)
    
    
    def get_label(self,idx):
        #if 'label_A' in self.rgbflst[int(idx)]: return int(0)
        #else: return int(1)
        if self.norm_lbl in self.rgbflst[int(idx)]: return np.float32(0.0)
        else: return np.float32(1.0)
    
        
    ## (ncrops, len, nfeats)
    def get_feat_wcrop(self,idx):
        feats = np.zeros((self.rgb_ncrops, self.len, self.nfeats), dtype=np.float32)
        
        for i in range(self.rgb_ncrops):
            
            ## RGB
            rgb_fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
            rgb_feat_crop = np.load(rgb_fp_crop).astype(np.float32)
            
            if not i: ## crop 0
                idxs = self.trnsfrm.get_idxs(len(rgb_feat_crop))
            
            #if self.rgbl2n: rgb_feat_crop = self.l2normfx(rgb_feat_crop)
            #if idx == 0: view_feat(feat_crop.asnumpy())
            log.debug(f'vid[{idx}][{i}][RGB] {rgb_feat_crop.shape} {rgb_feat_crop.dtype}  {osp.basename(rgb_fp_crop)}')
            
            
            ## AUD
            if self.audflst:
                if not i: ## load
                    aud_fp = f"{self.audflst[int(idx)]}.npy"
                    aud_feats = np.load(aud_fp).astype(np.float32)
                    log.debug(f'vid[{idx}][AUD] {aud_feats.shape} {aud_feats.dtype}  {osp.basename(aud_fp)}')
                    
                    if aud_feats.shape[0] != rgb_feat_crop.shape[0]: 
                        aud_feats = self.trnsfrm.interpolate(aud_feats, rgb_feat_crop.shape[0], None) 
                        log.debug(f'vid[{idx}] AUD : seg in2 {aud_feats.shape}')
                    
                    if self.audl2n: aud_feats = self.l2normfx(aud_feats)
                    
                try: feat_crop = np.hstack((rgb_feat_crop, aud_feats))
                except: log.info(f'{rgb_feat_crop} {aud_feats.shape}')
                log.debug(f'vid[{idx}][{i}][MIX] {feat_crop.shape} {feat_crop.dtype}')
                
            else: feat_crop = rgb_feat_crop

            ## if segmentation the means betwen adjacent selected segmetns is done
            ## if aud enabled, as its cat over feat dim
            #if self.l2n == 1: feat_crop = self.l2normfx(feat_crop)
            #log.info(f'vid[{idx}]: PRE-SEQ {feat_crop.shape}')
            feats[i] = self.trnsfrm.fx(feat_crop, idxs['idxs']) 
            #log.info(f'vid[{idx}]: PST-SEQ {feats[i].shape}')
            #if self.l2n == 2: feats[i] = self.l2normfx(feats[i],a=2)

            if idxs['rnd_glob'] is not None:
                ## rnd ovre same idxs per crop
                feats[i] = feats[i, idxs['rnd_glob']]
            
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype}')
        return feats
    
    
    ## (len, nfeats)
    def get_feat_wocrop(self,idx):
        rgb_fp = f"{self.rgbflst[int(idx)]}.npy"
        rgb_feats = np.load(rgb_fp).astype(np.float32)

        #if self.rgbl2n: rgb_feats = self.l2normfx(rgb_feats)
        log.debug(f"vid[{idx}][RGB] {rgb_feats.shape} {rgb_feats.dtype}  {osp.basename(rgb_fp)}")

        fprep_idxs = self.trnsfrm.get_idxs(len(rgb_feats))
        
        if self.audflst:
            if self.cropasvideo: aud_idx = int(idx)//self.rgb_ncrops
            else: aud_idx = int(idx)
            
            aud_fp = f"{self.audflst[aud_idx]}.npy"
            aud_feats = np.load( aud_fp ).astype(np.float32)
            
            if self.audl2n: aud_feats = self.l2normfx(aud_feats)
            log.debug(f'vid[{idx}][AUD] {aud_feats.shape} {aud_feats.dtype}  {osp.basename(aud_fp)}')
            
            #############################
            ## use FeatComp class 
            if aud_feats.shape[0] != rgb_feats.shape[0]:
                #log.debug(f'preseg {np.mean(aud_feats,axis=0)}')
                #for af in aud_feats[:16]: log.debug(f'{af[:40]}')
                aud_feats = self.self.trnsfrm.segmentation(aud_feats, rgb_feats.shape[0], None) 
                #log.debug(f'posseg {np.mean(aud_feats,axis=0)}')
                #for af in aud_feats[:16]: log.debug(f'{af[:40]}')
                log.debug(f'vid[{idx}][AUD] seg in2 {aud_feats.shape}')
            
            #log.info(f'vid[{idx}]: PRE-SEQ {aud_feats.shape}')
            aud_feats = self.trnsfrm.fx(aud_feats, fprep_idxs['idxs']) 
            #log.info(f'vid[{idx}]: PST-SEQ {aud_feats.shape}')
        
            feats = np.hstack((rgb_feats, aud_feats))
            log.debug(f'vid[{idx}][MIX] {feats.shape} {feats.dtype}')

        else: feats = rgb_feats     

        
        #log.info(f'vid[{idx}]: PRE-SEQ {rgb_feats.shape}')
        feats = self.trnsfrm.fx(feats, fprep_idxs['idxs']) 
        #log.info(f'vid[{idx}]: PST-SEQ {rgb_feats.shape}')
        
        if fprep_idxs['rnd_glob'] is not None:
            feats = feats[fprep_idxs['rnd_glob']]
        
        log.debug(f'vid[{idx}] {feats.shape} {feats.dtype}')
        return feats
    
    
    def __getitem__(self, idx):
        
        ## as n normal videos > abnormal is case of TRAIN.FRMT.SEG
        ## if idx out of range, pick a random idx
        if idx >= len(self.rgbflst):
            idx = np.random.randint(0,len(self.rgbflst))
            idx = NPRNG.integers(feat_len-self.len)
            
        if self.in2mem:
            feats , label = self.data[int(idx)]  
        else:
            label = self.get_label(idx)
            feats = self.get_feat(idx)
        
        #log.info(f'f[{idx}]: {feats.shape}')    
        return feats, label

    def __len__(self):
        return len(self.rgbflst)


################################
## PREP / FRMT 
## preprocessing of train input

## 1) FeatSegm prepares the features matrix for each .npy
class FeatSegm():
    def __init__(self, cfg, seed=None):
        
        if cfg.intplt and cfg.sel != 'uni':
            log.error(f"selection != uni while interpolate its on, overiding")
            self.sel = 'uni'
        else: self.sel = cfg.sel

        self.len = cfg.len
        self.rnd = cfg.rnd
        self.intplt = cfg.intplt
        
        self.fx = {
            1: self.interpolate,
            0: self.sel_or_pad
        }.get(cfg.intplt)
        
        #self.RNG = np.random.default_rng(seed)
        #self.rng = NPRNG
        #log.error(f" {np.random.default_rng.}")
    
    def get_idxs(self, feat_len):
        ## call this at the start of get_feat
        ## if crops in use 
        ## it usefull to return idxs, so later when calling the fprep.fx, idxxs are passed
        ## and the same idxxs are used for all crops
        ## if not in use this is called only one time 
        ## possible to dont call this at all or have idxs returned as NNone
        ## if cfg.RNDCROP is True
        ## so each crop will have different idxs chosen
        
        if self.intplt:
            idxs = np.linspace(0, feat_len, self.len+1, dtype=np.int32)
            idxs = self.rnd_jit(idxs, feat_len)
            
        else:
            if feat_len <= self.len:
                ## latter pad
                idxs = None 
                
            elif self.sel == 'uni':
                ## differ from intplt sel
                idxs = np.linspace(0, feat_len-1, self.len, dtype=np.int32)
                idxs = self.rnd_jit(idxs, feat_len)
                
            elif self.sel == 'seq':
                ## start point
                start = NPRNG.integers(0, feat_len-self.len)
                idxs = list(range(start, start+self.len))
        
        idxx_glob = None
        if 'glob' in self.rnd:
            ## aply glob rnd only after .fx is call
            idxx_glob = np.arange(self.len)
            NPRNG.shuffle(idxx_glob)
            log.debug(f"GLOB / NEW {idxx_glob=}")
            #return feat[idxx]

        return {
            'idxs': idxs,
            'rnd_glob': idxx_glob
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
        ## sultani interpolate
        
        ## ovverride given idxs
        #if cfg.RNDCROP: 
        #    log.error(f"{cfg.RNDCROP} in dev")
        #    #idxs = self.get_idxs(len(feat))['idxs']
        
        if idxs is None: ## aud temp dim align
            idxs = np.linspace(0, len(feat), self.len+1, dtype=np.int32)
        
        new_feat = np.zeros((self.len, feat.shape[1]), dtype=np.float32)
        for i in range(self.len):
            #log.info(f"{fn} {crop}")
            if idxs[i] != idxs[i+1]:
                ## t, f
                new_feat[i, :] = np.mean(feat[idxs[i]:idxs[i+1], :], axis=0)
                ## topk vvfeat ar
                #new_feat[i, :] = np.sort(feat[idxs[i]:idxs[i+1], :], axis=0)[0]
                # topk !!!!!
            else:
                new_feat[i, :] = feat[idxs[i], :]
        return new_feat

    def sel_or_pad(self, feat, idxs):
        ## https://github.com/Roc-Ng/XDVioDet/blob/master/utils.py
        def pad(feat):
            if np.shape(feat)[0] <= self.len:
                return np.pad(feat, ((0, self.len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
            else: return feat
        
        if idxs is None: return pad(feat)
        else:
            ## make use of idxs already pre seleted at crop0
            ## find a way to modulate each crop differently
            ## ovverride given idxs
            #if cfg.RNDCROP:
            #    log.error(f"{cfg.RNDCROP} in dev")
            #    #idxs = self.get_idxs(len(feat))['idxs']
            
            ## diffrence ??
            f = feat[ idxs ]
            assert f.shape == feat[idxs, :].shape

            return f    


## 2) frmter unifies the netowork input formating 
## handle MIL segmentation / SEQ sequencial
## called upon batchs of trainloader in trainep/trainepo
class TrainFrmter():
    def __init__(self, bs, cfg_trnsfrm):
        
        self.fx = {
            1:self.milfrmter,
            2:self.frmter
        }.get(cfg_trnsfrm.mil)
        
        self.bs = bs
        self.ncrops = cfg_trnsfrm.crops2use
        
    def rshp_in(self, x):
        ## reshapes the post frmt feats to handle the crop dimension
        if x.ndim == 4: 
            bs, ncrops, seqlen, nfeat = x.shape
            x = x.view(-1, seqlen, nfeat) ## ( bs*(ncrops), maxseqlen/seglen , nfeat)
        else: assert x.ndim == 3 ## no crops
        return x 

    def milfrmter(self, tdata, ldata, trn_inf):
        (nfeat, nlabel), (afeat, alabel) = tdata
        
        ## simply know that bs/2 *0 , bs/2 *1 ?
        ## or add no mather what ?
        ldata["label"] = torch.cat((nlabel, alabel), 0).to(trn_inf['dvc']) 
        
        cfeat = torch.cat((nfeat, afeat), 0) ## (2*bs, (ncrops), seglen , nfeats)
        
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] nfeat: {nlabel[0]} {nfeat.dtype} {nfeat.shape} {nfeat.device} , afeat: {alabel[0]} {afeat.dtype} {afeat.shape} {afeat.device}")
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] cfeat: {cfeat.shape} {cfeat.dtype} {cfeat.device}")

        return self.rshp_in(cfeat).to(trn_inf['dvc'])
        
    def frmter(self, tdata, ldata, trn_inf):
        cfeat, label = tdata
        
        ldata["label"] = label.to(trn_inf['dvc'])
        
        ## better add no mater what 
        #if "seqlen" in ldata: ## for CLAS loss fx
        if cfeat.ndim == 4: ##  b4 rshp in2 (bs*ncrops,maxseqlen,nfeats), as its equals to all ncrops views 
            seqlen = np.sum(np.max(np.abs(cfeat[:,0,:,:]), axis=2) > 0, axis=1)
        else: seqlen = np.sum(np.max(np.abs(cfeat), axis=2) > 0, axis=1)
        ldata["seqlen"] = seqlen.to(trn_inf['dvc'])
        
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] feat: {cfeat.shape} , lbl: {label} {label.shape} {label.device}")
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] seqlen: {seqlen.shape}")
        
        return self.rshp_in(cfeat).to(trn_inf['dvc'])

################################
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
        #log.info(f"{fn} {crop}")
        if idxs[i] != idxs[i+1]:
            new_feat[i, :] = np.mean(feat[idxs[i]:idxs[i+1], :], axis=0)
        else:
            new_feat[i, :] = feat[idxs[i], :]
    
    #if 'glob' in self.rnd: new_feat = self.rnd_glob(new_feat)
        
    return new_feat
'''