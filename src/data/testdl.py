import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import glob , os, os.path as osp , math , time
from multiprocessing import Pool

from src.utils import hh_mm_ss, logger
log = logger.get_log(__name__)
    

#############################
## TEST
def get_testloader(cfg):
    from ._data import FeaturePathListFinder, debug_cfg_data
    debug_cfg_data(cfg.data)
    
    cfg_ds = cfg.data.ds
    cfg_loader = cfg.dataloader.test
    cfg_trnsfrm = cfg.data.trnsfrm.test
    
    if cfg_ds.get('faud'):
        audfplf = FeaturePathListFinder(cfg.data, 'test', 'aud')
        audfl = audfplf.get('ANOM') + audfplf.get('NORM')
        log.info(f'TEST: AUD ON {len(audfl)} feats')
    else: audfl = []
        
    rgbfplf = FeaturePathListFinder(cfg.data, 'test', 'rgb')
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
    
    ds = TestDS(cfg_loader, cfg_ds, cfg_trnsfrm, rgbfl, audfl)
    loader = DataLoader( ds , 
                        batch_size=cfg_loader.bs , 
                        shuffle=cfg_loader.shuffle, 
                        num_workers=cfg_loader.nworkers, 
                        prefetch_factor=cfg_loader.pftch_fctr,
                        pin_memory=cfg_loader.pinmem, 
                        persistent_workers=cfg_loader.prstwrk 
                        )
    return loader


class TestDS(Dataset):
    def __init__(self, cfg_loader, cfg_ds, cfg_trnsfrm, rgbflst, audflst=[]):

        self.rgbflst = rgbflst
        self.audflst = audflst
        self.lbl_mng = LBL(cfg_ds.info)
        self.crops2use = cfg_trnsfrm.crops2use

        if cfg_loader.in2mem: self.loadin2mem(cfg_loader.nworkers)
        else: self.in2mem = 0
        
    def load_data(self,idx):
        return self.get_feat(idx), self.get_label(idx)
    
    def loadin2mem(self, nworkers):
        self.in2mem = 1
        log.info(f'LOADING DS IN2 MEM'); t=time.time()
        with Pool(processes=nworkers) as pool:
            self.data = pool.map(self.load_data, range(len(self.rgbflst)))  
        log.info(f'COMPLETED LOAD IN {hh_mm_ss(time.time()-t)}')

        
    def get_label(self,idx):
        fn = osp.basename(self.rgbflst[int(idx)])
        label = self.lbl_mng.encod(fn)
        #log.debug(f'vid[{idx}] {label=} {type(label)} {fn}')
        return label , fn
    
    def get_feat(self,idx):
        ## RGB 
        if self.crops2use:
            rgb_feats = []
            for i in range(self.crops2use):
                fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
                feat_crop = np.load(fp_crop).astype(np.float32)  ## (timesteps, 1024)
                #log.debug(f'crop[{i}] {osp.basename(fp_crop)}: {feat_crop.shape} {feat_crop.dtype}')                
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
        
        if self.in2mem:
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
    def __init__(self, cfg_ds):
        #log.debug(f'GTFL:\n{cfg_ds}')
        self.cfg_ds = cfg_ds
        self.encod = {
            'ucf': self.ucf_encod,
            'xdv': self.xdv_encod
        }.get(self.cfg_ds.id)
        
    def ucf_encod(self, fn):
        ## ucf
        aux = [fn[:-3]]
        ##aux = [ osp.basename(fn) ]
        #aux = [fn[:fn.find("_x264")-3]]
        
        idxs = [self.cfg_ds.lbls.index(a) for a in aux]
        aux = [self.cfg_ds.lbls_info[i] for i in idxs]
            
        return aux
    
    def xdv_encod(self, fn):
        assert 'label_' in fn ## xdv
        ## norm
        if self.cfg_ds.lbls[0] in fn: aux = [self.cfg_ds.lbls_info[0]]
        else:
            ## A = 0 , B1-B4-B6 = 146 , B1-B5-G = 157 , B4-0-0 = 400
            ## oldie aux = aux.replace('B', '').replace('-', '').replace('A', '0').replace('G', '7')   
            aux = fn[fn.find('label_')+len('label_'):]
            aux = [ a for a in aux.split('-') if a != '0']
            idxs = [self.cfg_ds.lbls.index(a) for a in aux]
            aux = [self.cfg_ds.lbls_info[i] for i in idxs]
        return aux