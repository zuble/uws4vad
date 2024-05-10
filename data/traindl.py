import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
import numpy as np

import glob , os, os.path as osp , math , time
from multiprocessing import cpu_count, Pool
CPU_COUNT = cpu_count()

from utils import FeaturePathListFinder, hh_mm_ss, seed_sade

log = None
def init(l):
    global log
    log = l
    

#############################
## TRAIN
class Zuader:
    ## encapsulates the dataloader independtly of elements yielded
    def __init__(self, frmt, *loaders):
        self.frmt = frmt
        self.loaders = loaders
    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]
        return self
    def __next__(self):
        if self.frmt == 'SEG': return next(self.loader_iters[0]), next(self.loader_iters[1])
        elif self.frmt == 'SEQ': return next(self.loader_iters[0])
        else: raise StopIteration


def get_trainloader(cfg):
    log.info(f'TRAIN: getting trainloader')
    
    
    if cfg.TRAIN.FRMT == 'SEG':
        ads, nds, arsampler, nrsampler = get_segds(cfg)
        ald = DataLoader(ads, 
                        batch_sampler=BatchSampler(arsampler, batch_size=cfg.TRAIN.BS//2 , drop_last=cfg.TRAIN.DROPLAST),
                        num_workers=cfg.DATA.NWORKERS,
                        worker_init_fn=seed_sade,
                        pin_memory=cfg.DATA.PINMEM, 
                        persistent_workers=cfg.DATA.PERSISTWRK 
                        ) 
        nld = DataLoader(nds, 
                        batch_sampler=BatchSampler(nrsampler, batch_size=cfg.TRAIN.BS//2 , drop_last=cfg.TRAIN.DROPLAST),
                        num_workers=cfg.DATA.NWORKERS,
                        worker_init_fn=seed_sade,
                        pin_memory=cfg.DATA.PINMEM, 
                        persistent_workers=cfg.DATA.PERSISTWRK 
                        )
        log.info(f'TRAIN: NWORKERS {cfg.DATA.NWORKERS} CPU_COUNT {CPU_COUNT} , ')
        return Zuader('SEG', nld, ald) 
    
    elif cfg.TRAIN.FRMT == 'SEQ': ## sequence
        ds, bsampler = get_seqds(cfg)
        ld0 = data.DataLoader( ds,
                            batch_sampler=bsampler,
                            num_workers=cfg.DATA.NWORKERS , 
                            pin_memory=cfg.DATA.PINMEM , 
                            thread_pool=cfg.DATA.THREAD 
                            )
        log.info(f'TRAIN: NWORKERS {cfg.DATA.NWORKERS} CPU_COUNT {CPU_COUNT} , ')
        return Zuader('SEQ', ld0)

def get_segds(cfg):
    def get_rnd_smp(ds, nsamples, gen):
        is_oversampling = len(ds) < nsamples
        log.info(f"{is_oversampling = }")
        return RandomSampler(ds, 
                            replacement=is_oversampling, 
                            num_samples=nsamples if is_oversampling else None, 
                            generator=gen)
        
    opts=[]

    if cfg.DATA.NWORKERS == 0: 
        opts.extend(['DATA.NWORKERS', int(CPU_COUNT / 4)])
    
    cfg_ds = getattr(cfg.DS, cfg.TRAIN.DS)
    
    
    aaudfl, naudfl = [], []
    if cfg.DATA.AUD.ENABLE and cfg.TRAIN.DS != 'UCF':
        audfplf = FeaturePathListFinder(cfg, 'train', 'aud', cfg_ds)
        aaudfl = audfplf.get('BG') 
        naudfl = audfplf.get('A')
        log.info(f'TRAIN: AUD {len(naudfl)} normal, {len(aaudfl)} abnormal')    
        
    rgbfplf = FeaturePathListFinder(cfg, 'train', 'rgb', cfg_ds)
    argbfl = rgbfplf.get('BG')
    nrgbfl = rgbfplf.get('A')
    log.info(f'TRAIN: RGB {len(nrgbfl)} normal, {len(argbfl)} abnormal')  
    
    cfg_prep = cfg.TRAIN.SEG
    prepfx = FeatPrep('seg', cfg_prep, cfg.SEED).fx
    log.info(f'TRAIN: {prepfx=}')  
    
    ads = TrainDS(cfg, cfg_ds, cfg_prep, prepfx, argbfl, aaudfl, 'ABNORMAL')
    nds = TrainDS(cfg, cfg_ds, cfg_prep, prepfx, nrgbfl, naudfl, 'NORMAL')
    
    maxlends = max(len(ads),len(nds))
    epoch_nbatch = maxlends // (cfg.TRAIN.BS // 2) #math.ceil(maxlends / cfg.TRAIN.BS)
    opts.extend(["TRAIN.EPOCHBATCHS", epoch_nbatch])
    log.info(f'TRAIN: {maxlends=} {cfg.TRAIN.BS//2=} -> {epoch_nbatch=} ')
    
    ## https://pytorch.org/docs/1.8.1/generated/torch.Generator.html#torch.Generator
    gen = torch.Generator().manual_seed(cfg.SEED) #log.error(f"{gen.initial_seed()}")
    
    ## as abnormal/normal not balanced set replacement True for smaller ds
    log.info(f'TRAIN: {len(ads)=}')
    arsampler = get_rnd_smp(ads, maxlends, gen)
    log.info(f'TRAIN: {len(nds)=}')
    nrsampler = get_rnd_smp(nds, maxlends, gen)

    cfg.merge_from_list(opts)
    
    return ads, nds, arsampler, nrsampler

def get_seqds(cfg):
    opts=[]
    
    if cfg.DATA.NWORKERS == 0: 
        opts.extend(['DATA.NWORKERS', int(CPU_COUNT / 4)])
    cfg_ds = getattr(cfg.DS, cfg.TRAIN.DS)
    
    audfl = []
    if cfg.DATA.AUD.ENABLE and cfg.TRAIN.DS != 'UCF':
        audfplf = FeaturePathListFinder(cfg, 'train', 'aud')
        audfl = audfplf.get('BG') + audfplf.get('A')
        log.info(f'TRAIN: AUD {len(audfl)} feats')
        
    rgbfplf = FeaturePathListFinder(cfg, 'train')
    rgbfl = rgbfplf.get('BG') + rgbfplf.get('A')
    log.info(f'TRAIN: RGB {len(rgbfl)} feats')

    cfg_prep = cfg.TRAIN.SEQ
    prepfx = FeatPrep('seq', cfg_prep, cfg.SEED).fx
    log.info(f'TRAIN: {prepfx=}')  
    
    ds = TrainDS(cfg, cfg_ds, cfg_prep, prepfx, rgbfl, audfl)   
    
    epoch_nbatch = len(ds) // cfg.TRAIN.BS #math.ceil(len(ds) / cfg.TRAIN.BS)
    ##assert epoch_nbatch == len(ds) // cfg.TRAIN.BS
    opts.extend(["TRAIN.EPOCHBATCHS", epoch_nbatch])
    log.info(f'TRAIN: len ds {len(ds)} , epoch_nbatch: {cfg.TRAIN.EPOCHBATCHS}')
    rsampler = data.RandomSampler(len(ds))
    bsampler = data.BatchSampler(rsampler , cfg.TRAIN.BS , 'discard')
    
    cfg.merge_from_list(opts)
    return ds, bsampler 


class TrainDS(Dataset):
    def __init__(self, cfg, cfg_ds, cfg_prep, prepfx, rgbflst, audflst, xtra=''):
        
        self.xtra = xtra
        self.cfg = cfg
        self.cfg_ds = cfg_ds
        self.prepfx = prepfx
        self.len = cfg_prep.LEN
        self.l2n = cfg_prep.L2N
        
        self.cropasvideo = cfg.DATA.CROPASVIDEO
        self.rgb_ncrops = cfg.DATA.RGB.NCROPS
        self.rgbflst = rgbflst
        self.rgbl2n = cfg.DATA.RGB.L2N
        self.audflst = audflst
        self.audl2n = cfg.DATA.AUD.L2N
        
        
        ## peakboo RGB features
        if osp.exists(f'{rgbflst[-1]}.npy'): pekaboo = f'{rgbflst[-1]}.npy'
        else: pekaboo = f"{rgbflst[-1]}__{0}.npy"
        
        tojo = np.load(pekaboo)    
        if not cfg.DATA.RGB.NFEATURES: cfg.DATA.RGB.NFEATURES = tojo.shape[-1]
        self.nfeatures = cfg.DATA.RGB.NFEATURES
        log.info(f'TRAIN {xtra}: RGB {osp.basename(pekaboo)} {np.shape(tojo)}')
        
        
        ## pekaboo AUD features
        if audflst:
            if not self.cropasvideo: assert len(rgbflst) == len(audflst)
            
            peakboo2 = f"{self.audflst[-1]}.npy"
            tojo2 = np.load(peakboo2)    
            if not cfg.DATA.AUD.NFEATURES:  cfg.DATA.AUD.NFEATURES = tojo2.shape[-1]
            self.audnfeatures = tojo2.shape[-1]
            self.nfeatures += self.audnfeatures
            log.info(f'TRAIN: AUD {osp.basename(peakboo2)} {np.shape(tojo2)}')
            
        
        log.info(f'TRAIN {xtra}: ({self.rgb_ncrops} ncrops, {self.len} maxseqlen/nsegments, {self.nfeatures} feats ({cfg.DATA.RGB.NFEATURES}+{cfg.DATA.AUD.NFEATURES}))')    
        log.info(f'TRAIN {xtra}: cropasvideo is {self.cropasvideo}')

        if cfg.DATA.LOADIN2MEM: self.loadin2mem()


    def load_data(self,idx): return self.get_feat(idx), self.get_label(idx)
    
    def loadin2mem(self):
        log.info(f'LOADING TRAIN {self.xtra} DS IN2 MEM'); t=time.time()
        with Pool(processes=self.cfg.DATA.NWORKERS) as pool:
            self.data = pool.map(self.load_data, range(len(self.rgbflst)))  
        log.info(f'COMPLETED TRAIN {self.xtra} LOAD IN {hh_mm_ss(time.time()-t)}')
        
    
    def l2normfx(self, x, a=-1): return x / np.linalg.norm(x, ord=2, axis=a, keepdims=True)
    
    
    def get_label(self,idx):
        #if 'label_A' in self.rgbflst[int(idx)]: return int(0)
        #else: return int(1)
        if self.cfg_ds.LBLS[0] in self.rgbflst[int(idx)]: return int(0)
        else: return int(1)
        
    def get_feat(self,idx):
        
        ## (ncrops, len, nfeatures)
        if self.rgb_ncrops and not self.cropasvideo:
            features = np.zeros((self.rgb_ncrops, self.len, self.nfeatures), dtype=np.float32)
            
            for i in range(self.rgb_ncrops):
                
                ## RGB
                rgb_fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
                rgb_feature_crop = np.load(rgb_fp_crop)
                
                if self.rgbl2n: rgb_feature_crop = self.l2normfx(rgb_feature_crop)
                #if idx == 0: view_feat(feature_crop.asnumpy())
                log.debug(f'vid[{idx}][{i}][RGB] {rgb_feature_crop.shape} {rgb_feature_crop.dtype}  {osp.basename(rgb_fp_crop)}')
                
                
                ## AUD
                if self.audflst:
                    if not i: ## load
                        aud_fp = f"{self.audflst[int(idx)]}.npy"
                        aud_features = np.array( np.load(aud_fp) )
                        log.debug(f'vid[{idx}][AUD] {aud_features.shape} {aud_features.dtype}  {osp.basename(aud_fp)}')
                        
                        if aud_features.shape[0] != rgb_feature_crop.shape[0]: 
                            aud_features = segmentation_feat(aud_features,rgb_feature_crop.shape[0]) 
                            log.debug(f'vid[{idx}] AUD : seg in2 {aud_features.shape}')
                        
                        if self.audl2n: aud_features = self.l2normfx(aud_features)
                        
                    try: feature_crop = np.hstack((rgb_feature_crop, aud_features))
                    except: log.info(f'{rgb_feature_crop} {aud_features.shape}')
                    log.debug(f'vid[{idx}][{i}][MIX] {feature_crop.shape} {feature_crop.dtype}')
                    
                else: feature_crop = rgb_feature_crop


                if self.l2n == 1: feature_crop = self.l2normfx(feature_crop)
                #log.info(f'vid[{idx}]: PRE-SEQ {feature_crop.shape}')
                features[i] = self.prepfx(feature_crop,self.len) 
                #log.info(f'vid[{idx}]: PST-SEQ {features[i].shape}')
                if self.l2n == 2: features[i] = self.l2normfx(features[i],a=2)

        ## (len, nfeatures)
        else:
            rgb_fp = f"{self.rgbflst[int(idx)]}.npy"
            rgb_features = np.array( np.load(rgb_fp) )
            
            if self.rgbl2n: rgb_features = self.l2normfx(rgb_features)
            log.debug(f"vid[{idx}][RGB] {rgb_features.shape} {rgb_features.dtype}  {osp.basename(rgb_fp)}")

            
            if self.audflst:
                if self.cropasvideo: aud_idx = int(idx)//self.rgb_ncrops
                else: aud_idx = int(idx)
                
                aud_fp = f"{self.audflst[aud_idx]}.npy"
                aud_features = np.array( np.load( aud_fp ) )
                
                if self.audl2n: aud_features = self.l2normfx(aud_features)
                log.debug(f'vid[{idx}][AUD] {aud_features.shape} {aud_features.dtype}  {osp.basename(aud_fp)}')
                
                #############################
                ## use FeatComp class 
                if aud_features.shape[0] != rgb_features.shape[0]:
                    #log.debug(f'preseg {np.mean(aud_features,axis=0)}')
                    #for af in aud_features[:16]: log.debug(f'{af[:40]}')
                    aud_features = segmentation_feat(aud_features,rgb_features.shape[0]) 
                    #log.debug(f'posseg {np.mean(aud_features,axis=0)}')
                    #for af in aud_features[:16]: log.debug(f'{af[:40]}')
                    log.debug(f'vid[{idx}][AUD] seg in2 {aud_features.shape}')
                    
                features = np.hstack((rgb_features, aud_features))
                log.debug(f'vid[{idx}][MIX] {features.shape} {features.dtype}')
    
            else: features = rgb_features     
        
            
            if self.l2n == 1: features = self.l2normfx(features)
            #log.info(f'vid[{idx}]: PRE-SEQ {features.shape}')
            features = self.prepfx(features,self.len) 
            #log.info(f'vid[{idx}]: PST-SEQ {features.shape}')
            if self.l2n == 2: features = self.l2normfx(features,a=2)
        
        
        log.debug(f'vid[{idx}] {features.shape} {features.dtype}')
        return features
    
    
    def __getitem__(self, idx):
        
        ## as n normal videos > abnormal is case of TRAIN.FRMT.SEG
        ## if idx out of range, pick a random idx
        if idx >= len(self.rgbflst):
            idx = np.random.randint(0,len(self.rgbflst))

        
        if self.cfg.DATA.LOADIN2MEM:
            features , label = self.data[int(idx)]  
        else:
            label = self.get_label(idx)
            features = self.get_feat(idx)
        
        #log.info(f'f[{idx}]: {features.shape}')    
        return features, label

    def __len__(self):
        return len(self.rgbflst)


################################
## PREP / FRMT 
## preprocessing of train input

## 1) FeatPrep prepares the features matrix for each .npy in order to have consistent shape
## its setup occurs in get_trainloader based on the cfg.TRAIN.FRMT
## according fx is further used in TrainDataset upon each .npy file after load
class FeatPrep():
    def __init__(self, what, cfg, seed):
        self.len = cfg.LEN
        self.rnd = cfg.RND
        self.jit_idxx, self.glob_idxx = None, None
        self.fx = {
            'seg': self.segmentation,
            'seq': self.seqmentation
        }.get(what)
        self.RNG = np.random.default_rng(seed)
        log.error(f" np.random.")
    
    def rnd_jit(self, idxx, len_feat):
        ## jitter betwen adjacent chosen idxx
        ## only when theres no repetead idxs
        ## when used in a ncrop configuration, each crop will have a different idxx
        ## taken from MIST random_petrub
        if len_feat > self.len:
            #log.debug(f'JIT / OLD {idxx=}')
            for i in range(self.len):
                if i < self.len - 1:
                    if idxx[i] != idxx[i+1]:
                        idxx[i] = self.RNG.choice(range(idxx[i], idxx[i+1]))  
            #log.debug(f'JIT / NEW {idxx=}')
        return idxx
        
    def rnd_glob(self, feat):
        idxx = np.arange(self.len)
        self.RNG.shuffle(idxx)
        log.debug(f"GLOB / NEW {idxx=}")
        return feat[idxx]

    ## sultani MIL
    def segmentation(self, feat, crop=0, fn=''):

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

    ## https://github.com/Roc-Ng/XDVioDet/blob/master/utils.py
    def seqmentation(self, feat, crop=0, fn=''):
        
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


## 2) frmter unifies the netowork input formating 
## handle MIL segmentation / SEQ sequencial
## called upon batchs of trainloader in trainep/trainepo
class TrainFrmter():
    def __init__(self,cfg):
        self.fx = {
            'SEG':self.segfrmter,
            'SEQ':self.seqfrmter
        }.get(cfg.TRAIN.FRMT)
        self.bs = cfg.TRAIN.BS
        self.ncrops = cfg.DATA.RGB.NCROPS
        
    def rshp_in(self, x):
        ## reshapes the post frmt feats to handle the crop dimension
        if x.ndim == 4: 
            bs, ncrops, seqlen, nfeat = x.shape
            x = x.view(-1, seqlen, nfeat) ## ( bs*(ncrops), maxseqlen/seglen , nfeat)
        else: assert x.ndim == 3 ## no crops
        return x 

    def segfrmter(self, tdata, ldata, trn_inf):
        ## cant have tdata(traindata) as a dict out of dataloader
        ## so have to ret
        (nfeat, nlabel), (afeat, alabel) = tdata
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] nfeat: {nlabel[0]} {nfeat.dtype} {nfeat.shape} {nfeat.device} , afeat: {alabel[0]} {afeat.dtype} {afeat.shape} {afeat.device}")
        
        ldata["label"] = torch.cat((nlabel, alabel), 0).to(trn_inf['dvc'])
        
        cfeat = torch.cat((nfeat, afeat), 0) ## (2*bs, (ncrops), seglen , nfeats)
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] cfeat: {cfeat.shape} {cfeat.dtype} {cfeat.device}")
        
        return self.rshp_in(cfeat).to(trn_inf['dvc'])
        
    def seqfrmter(self, tdata, ldata, trn_inf):
        cfeat, label = tdata
        log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] feat: {cfeat.shape} , lbl: {label} {label.shape} {label.device}")
        
        ldata["label"] = label.copyto(trn_inf['dvc'])
        if "seqlen" in ldata: ## for CLAS loss fx
            if cfeat.ndim == 4: ##  b4 rshp in2 (bs*ncrops,maxseqlen,nfeats), as its equals to all ncrops views 
                seqlen = np.sum(np.max(np.abs(cfeat[:,0,:,:]), axis=2) > 0, axis=1)
            else: seqlen = np.sum(np.max(np.abs(cfeat), axis=2) > 0, axis=1)
            ldata["seqlen"] = seqlen
            log.debug(f"E[{trn_inf['epo']+1}]B[{trn_inf['bat']+1}] seqlen: {seqlen.shape}")
        
        return self.rshp_in(cfeat).copyto(trn_inf['dvc'])

################################