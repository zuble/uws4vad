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
def get_testloader(cfg, cfg_ds):
    from ._data import FeaturePathListFinder
        
    if cfg.DATA.NWORKERS == 0: cfg.DATA.NWORKERS = int(CPU_COUNT / 4)
    log.info(f'TEST: CPU_COUNT {CPU_COUNT} , NWORKERS {cfg.DATA.NWORKERS}')
    
    audfl = []
    if cfg.DATA.AUD.ENABLE and cfg.TEST.DS != 'UCF':
        audfplf = FeaturePathListFinder(cfg, 'test', 'aud', cfg_ds)
        audfl = audfplf.get('BG')+audfplf.get('A')
        log.info(f'TEST: AUD {len(audfl)} feats')
        
    rgbfplf = FeaturePathListFinder(cfg, 'test', 'rgb', cfg_ds)
    rgbfl = rgbfplf.get('BG') + rgbfplf.get('A')
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
    ## both gt.npy for xdv and ucf have same lenght as total number of segments in features * window length of the feature extractor
    #######################################
    
    ds = TestDS(cfg, rgbfl, audfl)

    loader = DataLoader( ds , 
                        batch_size=cfg.TEST.BS , 
                        shuffle=False , 
                        num_workers=cfg.DATA.NWORKERS, 
                        pin_memory=cfg.DATA.PINMEM, 
                        persistent_workers=cfg.DATA.PERSISTWRK 
                        ) 
    return loader


class TestDS(Dataset):
    def __init__(self, cfg, rgbflst, audflst=[]):
        self.cfg = cfg
        self.rgbflst = rgbflst
        self.audflst = audflst
        self.lbl_mng = LBL(cfg)
        self.rgb_ncrops = cfg.DATA.RGB.NCROPS
        #self.full_or_center = full_or_center = 'center'
        
        
        ## peakboo RGB features
        if osp.exists(f'{self.rgbflst[0]}.npy'): peakboo = f'{self.rgbflst[0]}.npy'
        else: peakboo = f"{self.rgbflst[0]}__{0}.npy"
        
        tojo = np.load(peakboo)    
        if not cfg.DATA.RGB.NFEATURES:  cfg.DATA.RGB.NFEATURES = tojo.shape[-1]
        log.info(f'TEST: RGB {osp.basename(peakboo)} {tojo.shape}')
        
        
        ## peakboo AUD features
        if audflst:
            #if len(rgbflst) != len(audflst):
            #    r = set(osp.basename(path) for path in rgbflst)
            #    a = set(osp.basename(path) for path in audflst)
            #    not_in_both = r.symmetric_difference(a)
            #    log.info(f'TEST: not_in_both {not_in_both}')
            #    for basename in not_in_both:
            #        rgbflst = [path for path in rgbflst if osp.basename(path) != basename]
            #        audflst = [path for path in audflst if osp.basename(path) != basename]
            #assert len(rgbflst) == len(audflst)
            assert len(rgbflst) == len(audflst)
            
            peakboo2 = f"{self.audflst[0]}.npy"
            tojo2 = np.load(peakboo2)    
            if not cfg.DATA.AUD.NFEATURES:  cfg.DATA.AUD.NFEATURES = tojo2.shape[-1]
            log.info(f'TEST: AUD {osp.basename(audflst[0])} {np.shape(tojo2)}')

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
        '''
            if self.full_or_center == 'center':
            elif self.full_or_center == 'full':
                features = []
                for i in range(self.rgb_ncrops):
                    fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
                    feature_crop = np.load(fp_crop)  ## (timesteps, 1024)
                    log.debug(f'crop[{i}] {osp.basename(fp_crop)}: {feature_crop.shape} {feature_crop.dtype}')                
                    #features[i] = np.array(feature_crop)
                    features.append(feature_crop)
                features = np.array(features)
                log.debug(f'f[{idx}]: {type(features)} {features.shape} {features.dtype}') ## (5,32,1024)
        '''

        ## RGB 
        if self.rgb_ncrops: 
            rgb_fp = f"{self.rgbflst[int(idx)]}__{0}.npy"  ## center crop
        else: 
            rgb_fp = f"{self.rgbflst[int(idx)]}.npy"    
            
        rgb_features = np.array( np.load(rgb_fp) )
        log.debug(f'vid[{idx}][RGB] {rgb_features.shape} {rgb_features.dtype}  {osp.basename(rgb_fp)} ')
        
        ## AUD
        if self.audflst:
            aud_fp = f"{self.audflst[int(idx)]}.npy"
            aud_features = np.array( np.load(aud_fp) )
            log.debug(f'vid[{idx}] AUD {osp.basename(aud_fp)}: {aud_features.shape} {aud_features.dtype}')
            
            if aud_features.shape[0] != rgb_features.shape[0]:
                log.debug(f'vid[{idx}]AUD rshp 2mtch rgbf')
                aud_features = segmentation_feat(aud_features,rgb_features.shape[0]) 
                
            ## MIX
            features = np.concatenate((rgb_features, aud_features), axis=1)
            log.debug(f'vid[{idx}] MIX {features.shape} {features.dtype}')
            
        else: features = rgb_features
        
        
        
        return features
    
    def __getitem__(self, idx):
        
        if self.cfg.DATA.LOADIN2MEM:
            features, (label, fn)= self.data[int(idx)]    
        else:   
            label,fn = self.get_label(idx)
            features = self.get_feat(idx)
        
        return features , label , str(fn)
    
    def __len__(self):
        return len(self.rgbflst)

def custom_batchify_fn(data):
    if isinstance(data[0], tuple):
        data = zip(*data)
    return [i for i in data]

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
            aux = [fn[:fn.find("_x264")-3]]
            idxs = [self.cfg_ucf.LBLS.index(a) for a in aux]
            aux = [self.cfg_ucf.LBLS_INFO[i] for i in idxs]
            
        return aux