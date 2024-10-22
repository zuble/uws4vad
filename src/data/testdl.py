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
    from ._data import FeaturePathListFinder, debug_cfg_data, run_dltest
    if cfg.get("debug"): debug_cfg_data(cfg)
    
    cfg_ds = cfg.data
    cfg_dload = cfg.dataload.test
    cfg_dproc = cfg.dataproc
    
    
    rgbfplf = FeaturePathListFinder(cfg, 'test', 'rgb')
    rgbfl = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    log.info(f'TEST: RGB {len(rgbfl)} feats')

    if cfg_ds.get('faud'):
        audfplf = FeaturePathListFinder(cfg, 'test', 'aud', auxrgbflist=rgbfl)
        audfl = audfplf.get('ANOM') + audfplf.get('NORM')
        log.info(f'TEST: AUD ON {len(audfl)} feats')
    else: audfl = []
        

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
    
    ds = TestDS(cfg_dload, cfg_ds, cfg_dproc, rgbfl, audfl)
    loader = DataLoader( ds , 
                        batch_size=cfg_dload.bs , 
                        shuffle=cfg_dload.shuffle, 
                        num_workers=cfg_dload.nworkers, 
                        #prefetch_factor=cfg_dload.pftch_fctr if cfg_dload.pftch_fctr != 0 else None,
                        pin_memory=cfg_dload.pinmem, 
                        persistent_workers=cfg_dload.prstwrk 
                        )
    
    if cfg_dload.dryrun:
        log.warning(f"DBG DRY TEST DL")            
        run_dltest(loader)

    return loader


class TestDS(Dataset):
    def __init__(self, cfg_dload, cfg_ds, cfg_dproc, rgbflst, audflst=[]):

        self.rgbflst = rgbflst
        self.audflst = audflst
        self.lbl_mng = LBL(ds=cfg_ds.id, cfg_lbls=cfg_ds.lbls)
        self.crops2use = cfg_dproc.crops2use.test
        
        self.dfeat = cfg_ds.frgb.dfeat
        if audflst: self.peakboo_aud(cfg_ds.faud.dfeat)

        if cfg_dload.in2mem: self.loadin2mem(cfg_dload.nworkers)
        else: self.in2mem = 0
    
    
    def peakboo_aud(self, dfeat):
        ## pekaboo AUD features
        assert len(self.rgbflst) == len(self.audflst)
        peakboo2 = f"{self.audflst[-1]}.npy"
        tojo2 = np.load(peakboo2)    
        assert dfeat == tojo2.shape[-1]
        self.dfeat += dfeat
        log.debug(f'TEST: AUD {osp.basename(peakboo2)} {np.shape(tojo2)}')    
    
        
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
        log.debug(f'vid[{idx}] {label=} {type(label)} {fn}')
        return label , fn
    
    def get_feat(self, idx):
        """
        Gets RGB and audio features for a given index, handling crops and potential length mismatches.
        Args:
            idx (int): The index of the video/audio sample.
        Returns:
            np.ndarray: The concatenated features (RGB + audio) with shape (ncrops, t, dfeat) or (t, dfeat).
        """
        ## RGB 
        if self.crops2use:
            rgb_feats = []
            for i in range(self.crops2use):
                fp_crop = f"{self.rgbflst[int(idx)]}__{i}.npy"
                feat_crop = np.load(fp_crop).astype(np.float32)  ## (t, dfrgb)
                log.debug(f'crop[{i}] {osp.basename(fp_crop)}: {feat_crop.shape} {feat_crop.dtype}')                
                rgb_feats.append(feat_crop)

            rgb_feats = np.array(rgb_feats)  ## (ncrops, t, dfrgb)
            log.debug(f'f[{idx}]: {type(rgb_feats)} {rgb_feats.shape} {rgb_feats.dtype}') 
        else: 
            rgb_fp = f"{self.rgbflst[int(idx)]}.npy"    
            rgb_feats = np.load(rgb_fp).astype(np.float32) ## (t, dfrgb)
            log.debug(f'vid[{idx}][RGB] {rgb_feats.shape} {rgb_feats.dtype}  {osp.basename(rgb_fp)} ')
        
        ## AUD
        if self.audflst:
            aud_fp = f"{self.audflst[int(idx)]}.npy"
            aud_feats = np.load(aud_fp).astype(np.float32) ## (t_aud , dfaud)
            log.debug(f'vid[{idx}] AUD {osp.basename(aud_fp)}: {aud_feats.shape} {aud_feats.dtype}')
            
            aud_feats = self._match_audio_length(aud_feats, rgb_feats.shape[-2])

            ## MIX
            if self.crops2use:
                aud_feats = np.repeat(aud_feats[np.newaxis, :, :], self.crops2use, axis=0)  # (ncrops, t, dfaud)
                feats = np.concatenate((rgb_feats, aud_feats), axis=2) # (ncrops, t, dfrgb+dfaud)
                log.debug(f'vid[{idx}] MIX (w/nc): {feats.shape} {feats.dtype}')
            else:
                feats = np.concatenate((rgb_feats, aud_feats), axis=1) # (t, dfrgb+dfaud)
                log.debug(f'vid[{idx}] MIX: {feats.shape} {feats.dtype}')
            
            return feats
        
        else: return rgb_feats
        

    def _match_audio_length(self, aud_feats, t_rgb):
        """
        Matches the length of audio features to the RGB features using linear interpolation.
        Args:
            aud_feats (np.ndarray): Audio features with shape (t_aud, dfaud).
            rgb_shape (tuple): Shape of the RGB features, either (ncrops, t, dfrgb) or (t, dfrgb).
        Returns:
            np.ndarray: Audio features with length matched to RGB, shape (t, dfaud).
        """
        t_aud = aud_feats.shape[-2]
        df_aud = aud_feats.shape[-1]  

        if t_rgb == t_aud: return aud_feats
        elif 1 <= (t_rgb - t_aud) <= 2:  # Adjust only if RGB is 1 or 2 elements longer
            new_aud_feats = np.zeros((t_rgb, df_aud)).astype(np.float32)
            idxs = np.linspace(0, t_aud - 1, t_rgb, dtype=np.int32) 
            
            idxs2 = np.round(np.arange(t_rgb) * (t_aud) / (t_rgb)).astype(np.int32)
            
            repeat_counts = np.ones(t_aud, dtype=int)
            ## Distribute the required repetitions at the end of the array
            for i in range(t_rgb - t_aud):
                repeat_counts[-(i % t_aud) - 1] += 1
            idxs3 = np.repeat(np.arange(t_aud), repeat_counts)
            
            ## eg t_aud 10 / t_rgb 12
            log.debug(f'idxs {idxs.shape} \n {idxs}') ##    [0 0 1 2 3 4 4 5 6 7 8 9]
            log.debug(f'idxs2 {idxs2.shape} \n {idxs2}') ## [0 1 2 2 3 4 5 6 7 8 8 9]
            log.debug(f'idxs3 {idxs3.shape} \n {idxs3}') ## [0 1 2 3 4 5 6 7 8 8 9 9]

            new_aud_feats = aud_feats[idxs, :]
            
            log.debug(f'Matched aud {aud_feats.shape[0]} w/ rgb {t_rgb} -> {new_aud_feats.shape}')
            # (Optional) Comparison check (remove if not needed)
            # ---------------------------------------------------
            # new_aud_feats_interp = np.zeros((t_rgb, df_aud))
            # x = np.linspace(0, aud_feats.shape[0] - 1, num=aud_feats.shape[0])
            # xp = np.linspace(0, aud_feats.shape[0] - 1, num=t_rgb)
            # for f in range(df_aud):
            #     new_aud_feats_interp[:, f] = np.interp(xp, x, aud_feats[:, f])
            # if np.allclose(new_aud_feats, new_aud_feats_interp):
            #     log.debug(f'Segmentation check successful.')
            # else:
            #     log.error(f'Segmentation mismatch!')
            # ---------------------------------------------------
            return new_aud_feats
        elif 1 <= (t_aud - t_rgb) <= 2: return aud_feats[:t_rgb]
        else: raise ValueError(f'MISMATCH: Incompatible lengths - RGB: {t_rgb}, Audio: {t_aud}')
        
    def __getitem__(self, idx):
        
        if self.in2mem:
            feats, (label, fn)= self.data[int(idx)]    
        else:   
            label, fn = self.get_label(idx)
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
    def __init__(self, ds, cfg_lbls):
        self.cfg_lbls = cfg_lbls
        self.encod = {
            'ucf': self.ucf_encod,
            'xdv': self.xdv_encod
        }.get(ds)
        
    def ucf_encod(self, fn):
        aux = [fn[:-3]]
        ##aux = [ osp.basename(fn) ]
        #aux = [fn[:fn.find("_x264")-3]]
        
        idxs = [self.cfg_lbls.id.index(a) for a in aux]
        aux = [self.cfg_lbls.info[i] for i in idxs]
            
        return aux
    
    def xdv_encod(self, fn):
        assert 'label_' in fn 
        ## norm
        if self.cfg_lbls.id[0] in fn: aux = [self.cfg_lbls.info[0]]
        else:
            aux = fn[fn.find('label_')+len('label_'):] ## B2-G-0 or B2-G-0__0
            aux = aux.split("__")[0]
            aux = [ a for a in aux.split('-') if a != '0'] ## ['B2', 'G']
            idxs = [self.cfg_lbls.id.index(a) for a in aux] ## [2, 6]
            aux = [self.cfg_lbls.info[i] for i in idxs] ## ['B2.SHOOT', 'G.EXPLOS'] pass as tuple
            #log.debug(aux)
        return aux