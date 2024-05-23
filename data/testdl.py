import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import glob , os, os.path as osp , math , time
from multiprocessing import cpu_count, Pool
CPU_COUNT = cpu_count()

from utils import hh_mm_ss

log = None
def init(l):
    global log
    log = l


#############################
## TEST
def get_testloader(cfg):
    from ._data import FeaturePathListFinder
        
    if cfg.DATA.NWORKERS == 0: 
        cfg.merge_from_list(['DATA.NWORKERS', int(CPU_COUNT / 4)])
    log.info(f'TEST: CPU_COUNT {CPU_COUNT} , NWORKERS {cfg.DATA.NWORKERS}')
    
    
    ## ['DS.RGB.FEATTYPE','DS.AUD.FEATYPE']
    ds, modality, featdir = cfg.TEST.DS[0].split(".")
    cfg_ds = getattr(cfg.DS, ds)
        
    audfl = []
    cfg_faud = None
    if len(cfg.TRAIN.DS) == 2: ## aud
        ds2, modality2, featdir2 = cfg.TRAIN.DS[1].split(".")
        assert ds2 != 'UCF' and ds == ds2 and modality2 == 'AUD'
        cfg_faud = getattr( getattr( cfg_ds, modality2 ), featdir2) 
        audfplf = FeaturePathListFinder(cfg, 'test', modality2, featdir2, cfg_ds, cfg_faud)
        audfl = audfplf.get('ANOM') + audfplf.get('NORM')
        log.info(f'TEST: AUD {len(audfl)} feats')
        
    cfg_frgb = getattr( getattr( cfg_ds, modality ), featdir)   
    rgbfplf = FeaturePathListFinder(cfg, 'test', modality, featdir, cfg_ds, cfg_frgb)
    rgbfl = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    log.info(f'TEST: RGB {len(rgbfl)} feats')


    ##################################
    ## FRAME COUNTER COMPARE BETWEN THE VIDEOS AND FEATURES
    #tft1 = 0; tft2 = 0
    #for vp in rgbfl:
    #    ## TOTAL FRAME COUNTER FROM VIDEOS DS
    #    _,tf1,_ = mp4_rgb_info(osp.join(cfg_ds.VROOT,f'{osp.basename(vp)}.mp4'))
    #    tft1 += tf1
    #    ## TOTAL FRAME COUNTER FROM FEATURES DS 
    #    try:
    #        f = numpy.load(f'{vp}.npy')
    #        tf2 = f.shape[0]*cfg.DATA.RGB.SEGMENTNFRAMES
    #    except: 
    #        f = numpy.load(f'{vp}__{0}.npy')
    #        tf2 = f.shape[0]*cfg.DATA.RGB.SEGMENTNFRAMES
    #    tft2 += tf2
    #    log.info(f"{osp.basename(vp)} {tf1=} {tf2=}")
    #log.info(f"\n\n\n\n{tft1=} {tft2=}\n\n\n\n")
    ## both gt.npy for xdv and ucf have same lenght as total number of segments in feats * window length of the feature extractor
    #######################################
    
    ds = TestDS(cfg, rgbfl, audfl)

    loader = DataLoader( ds , 
                        batch_size=cfg.TEST.BS , 
                        shuffle=False , 
                        num_workers=cfg.DATA.NWORKERS, 
                        pin_memory=cfg.DATA.PINMEM, 
                        persistent_workers=cfg.DATA.PERSISTWRK 
                        ) 
    return loader, cfg_frgb, cfg_faud


class TestDS(Dataset):
    def __init__(self, cfg, rgbflst, audflst=[]):
        self.cfg = cfg
        self.rgbflst = rgbflst
        self.audflst = audflst
        self.lbl_mng = LBL(cfg)
        self.crops2use = cfg.TEST.CROPS2USE

        if cfg.DATA.LOADIN2MEM: self.loadin2mem()

    def load_data(self,idx):
        return self.get_feat(idx), self.get_label(idx)
    
    def loadin2mem(self):
        log.info(f'LOADING DS IN2 MEM'); t=time.time()
        with Pool(processes=self.cfg.DATA.NWORKERS) as pool:
            self.data = pool.map(self.load_data, range(len(self.rgbflst)))  
        log.info(f'COMPLETED LOAD IN {hh_mm_ss(time.time()-t)}')

        
    def get_label(self,idx):
        fn = osp.basename(self.rgbflst[int(idx)])
        label = self.lbl_mng.encod(fn)
        log.debug(f'vid[{idx}] {label=} {type(label)} {fn}')
        return label , fn
    
    def get_feat(self,idx):
        ## RGB 
        if self.crops2use:
            rgb_feats = []
            for i in range(self.crops2use):
                fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
                feat_crop = np.load(fp_crop).astype(np.float32)  ## (timesteps, 1024)
                log.debug(f'crop[{i}] {osp.basename(fp_crop)}: {feat_crop.shape} {feat_crop.dtype}')                
                #features[i] = np.array(feat_crop)
                rgb_feats.append(feat_crop)
            rgb_feats = np.array(rgb_feats)
            log.debug(f'f[{idx}]: {type(rgb_feats)} {rgb_feats.shape} {rgb_feats.dtype}') ## (5,32,1024)
        else: 
            rgb_fp = f"{self.rgbflst[int(idx)]}.npy"    
            rgb_feats = np.load(rgb_fp).astype(np.float32)
            log.debug(f'vid[{idx}][RGB] {rgb_feats.shape} {rgb_feats.dtype}  {osp.basename(rgb_fp)} ')
        
        ## AUD
        if self.audflst:
            aud_fp = f"{self.audflst[int(idx)]}.npy"
            aud_feats = np.load(aud_fp).astype(np.float32)
            log.debug(f'vid[{idx}] AUD {osp.basename(aud_fp)}: {aud_feats.shape} {aud_feats.dtype}')
            
            if aud_feats.shape[0] != rgb_feats.shape[0]:
                log.debug(f'vid[{idx}]AUD rshp 2mtch rgbf')
                t, f = rgb_feats.shape
                new_feat = np.zeros((t, f)).astype(np.float32)
                idxs = np.linspace(0, len(aud_feats), t+1, dtype=np.int32)
                for i in range(t):
                    #if idxs[i] != idxs[i+1]:
                    #    new_feat[i, :] = np.mean(feat[idxs[i]:idxs[i+1], :], axis=0)
                    #else:
                    new_feat[i, :] = aud_feats[idxs[i], :]
                aud_feats = new_feat
            ## MIX
            feats = np.concatenate((rgb_feats, aud_feats), axis=1)
            log.debug(f'vid[{idx}] MIX {feats.shape} {feats.dtype}')
            
        else: feats = rgb_feats
        
        return feats
    
    def __getitem__(self, idx):
        
        if self.cfg.DATA.LOADIN2MEM:
            feats, (label, fn)= self.data[int(idx)]    
        else:   
            label,fn = self.get_label(idx)
            feats = self.get_feat(idx)
        
        return feats , label , str(fn)
    
    def __len__(self):
        return len(self.rgbflst)

################################
## LABELS
class LBL:
    '''
        Retrieves VL label for the specified vpath
        either for the XDV or UCF dataset
    '''
    def __init__(self, cfg):
        #log.debug(f'GTFL:\n{cfg_ds}')
        self.cfg_xdv = cfg.DS.XDV
        self.cfg_ucf = cfg.DS.UCF    
        
    def encod(self, fn):
        if 'label_' in fn: ## xdv
            ## norm
            if self.cfg_xdv.LBLS[0] in fn: aux = [self.cfg_xdv.LBLS_INFO[0]]
            else:
                ## A = 0 , B1-B4-B6 = 146 , B1-B5-G = 157 , B4-0-0 = 400
                ## oldie aux = aux.replace('B', '').replace('-', '').replace('A', '0').replace('G', '7')   
                aux = fn[fn.find('label_')+len('label_'):]
                aux = [ a for a in aux.split('-') if a != '0']
                idxs = [self.cfg_xdv.LBLS.index(a) for a in aux]
                aux = [self.cfg_xdv.LBLS_INFO[i] for i in idxs]
        
        else: ## ucf
            aux = [fn[:-3]]
            ##aux = [ osp.basename(fn) ]
            #aux = [fn[:fn.find("_x264")-3]]
            
            idxs = [self.cfg_ucf.LBLS.index(a) for a in aux]
            aux = [self.cfg_ucf.LBLS_INFO[i] for i in idxs]
            
        return aux