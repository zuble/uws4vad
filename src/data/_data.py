import torch
import numpy as np, random

import glob , os, os.path as osp, math, time
from collections import OrderedDict

from src.utils import mp4_rgb_info, logger
log = logger.get_log(__name__)


###########
## DEBUGGER
def run_dl(dl):
    """
    """
    try:
        for ii in range(0,2): ## 2 epos
            tic = time.time()
            for b_idx, data in enumerate(dl):
                #if not b_idx: log.info(f"{type(data)} {len(data)} {data}") 
                if len(data[0]) == 2:
                    (nfeat, nlabel), (afeat, alabel) = data
                    log.debug(f'B[{b_idx+1}] (seg) NORM {nfeat.shape=} {nfeat.dtype} | {nlabel.shape=} {nlabel[0]} {nlabel.dtype}')
                    log.debug(f'B[{b_idx+1}] (seg) ABNORM {afeat.shape=} {afeat.dtype} | {alabel.shape=} {alabel[0]} {alabel.dtype}')
                else:
                    cfeat, label = data
                    log.debug(f'B[{b_idx+1}] (seq) {cfeat.shape=} {cfeat.dtype} | {label.shape=} {label.dtype}')
            log.debug(f'[{ii}] time to load {b_idx+1} batches {time.time()-tic}')
    except ValueError as e:
        log.error(f'Error unpacking variables in batch {b_idx+1}: {e}')


def debug_cfg_data(cfg_data):
    """
    """
    ID_DL = cfg_data.id
    cfg_dsinf = cfg_data.ds.info
    cfg_trnsfrm = cfg_data.trnsfrm
    
    #log.debug("DEBUG RGB")
    cfg_frgb = cfg_data.ds.frgb
    ID_RGB = cfg_frgb.id
    assert cfg_dsinf.id in cfg_frgb.ds  
    
    ## TRAIN TEST FOLDERS
    root_rgb = cfg_dsinf.froot+"/RGB/"+ID_RGB
    assert osp.exists(root_rgb), f"{root_rgb} does not exist"
    assert osp.exists(f"{root_rgb}/TRAIN")
    assert osp.exists(f"{root_rgb}/TEST")
    #log.debug(f"[{ID_DL}] {root_rgb} w/ TRAIN + TEST folders")
    
    
    fpaths_train = glob.glob(f"{root_rgb}/TRAIN/*.npy")
    fpaths_train.sort()
    #log.debug(f"[{ID_DL}] found {len(fpaths_train)} .npy files")
    
    fpaths_test = glob.glob(f"{root_rgb}/TEST/*.npy")
    fpaths_test.sort()
    #log.debug(f"[{ID_DL}] found {len(fpaths_test)} .npy files")
    
    
    ## NUMBER OF .npy FEATS NCROPS ASSERT
    if cfg_frgb.ncrops:
        
        if cfg_trnsfrm.train.crops2use == 0:
            log.error(f"[{ID_DL}] wrong {cfg_trnsfrm.train.crops2use=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception 
        if cfg_trnsfrm.test.crops2use == 0:
            log.error(f"[{ID_DL}] wrong {cfg_trnsfrm.test.crops2use=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception
        
        assert len(fpaths_train) == (cfg_dsinf.train.normal + cfg_dsinf.train.abnormal) * cfg_frgb.ncrops
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg_dsinf.froot}/RGB/TRAIN/{ID_RGB}")
        assert len(fpaths_test) == (cfg_dsinf.test.normal + cfg_dsinf.test.abnormal) * cfg_frgb.ncrops    
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg_dsinf.froot}/RGB/TEST/{ID_RGB}")
        
    else:
        if cfg_trnsfrm.train.crops2use > 0:
            log.error(f"[{ID_DL}] wrong {cfg_trnsfrm.train.crops2use=} while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception 
        if cfg_trnsfrm.test.crops2use > 0:
            log.error(f"[{ID_DL}] wrong {cfg_trnsfrm.test.crops2use=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception
        
        if cfg_trnsfrm.train.cropasvideo: 
            log.error(f"[{ID_DL}] {cfg_frgb.ncrops=} while cropasvideo is True")
            raise Exception 
        
        assert len(fpaths_train) == (cfg_dsinf.train.normal + cfg_dsinf.train.abnormal)
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg_dsinf.froot}/RGB/TRAIN/{ID_RGB}")
        
        assert len(fpaths_test) == (cfg_dsinf.test.normal + cfg_dsinf.test.abnormal)
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg_dsinf.froot}/RGB/TEST/{ID_RGB}")
    
    
    ## FEAT LEN ASSERT
    tmp = np.load(fpaths_train[random.randint(0,len(fpaths_train))])
    assert tmp.shape[1] == cfg_frgb.dfeat
    #log.debug(f"[{ID_DL}] {cfg_frgb.dfeat=} match train .npy files")
    
    tmp = np.load(fpaths_test[random.randint(0,len(fpaths_test))])
    assert tmp.shape[1] == cfg_frgb.dfeat
    #log.debug(f"[{ID_DL}] {cfg_frgb.dfeat=} match test .npy files")
    
    
    ## FEATUREATHLISTFINDER
    rgbfplf = FeaturePathListFinder(cfg_data, 'train', 'rgb')
    rgbfl = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    #log.debug(f'TRAIN: RGB {len(rgbfl)} feats')
    
    rgbfplf = FeaturePathListFinder(cfg_data, 'test', 'rgb')
    rgbfl = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    #log.debug(f'TEST: RGB {len(rgbfl)} feats')
    
    
    if cfg_data.ds.get("faud"):
        raise NotImplementedError
    
    
    
##############
## XDV/UCF DS
def get_testxdv_info(txt_path):
    txt = open(txt_path,'r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    for vidx in range(len(video_list)):
        log.info(video_list[vidx])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[vidx])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[vidx][nota_i])
                dif_aux = aux2-int(video_list[vidx][nota_i-1])
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
        log.info(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[vidx][-1]),'max anom frame\n')
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    log.info("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")

def get_xdv_stats():
    folders = {"train":"/raid/DATASETS/anomaly/XD_Violence/training_copy", "test":"/raid/DATASETS/anomaly/XD_Violence/testing_copy"}
    for key, folder in folders.items():
        paths = glob.glob(f"{folder}/*.mp4")
        total = [0,0] ## normal , anom
        for p in paths:
            dur, tframes, fps = mp4_rgb_info(p)
            if "label_A" in p: total[0] += tframes
            else: total[1] += tframes
            
        log.info(f"XDV {key} STATS")
        log.info(f"{len(paths)} videos")
        log.info(f"TOTAL NORMAL FRAMES: {total[0]} {(total[0]/(total[0]+total[1]))*100:2f}%")
        log.info(f"TOTAL ANOMALY FRAMES: {total[1]} {(total[1]/(total[0]+total[1]))*100:2f}% \n\n")

def get_ucf_stats():
    folders = {"train":"/raid/DATASETS/anomaly/UCF_Crimes/DS/train", "test":"/raid/DATASETS/anomaly/UCF_Crimes/DS/test"}
    for key, folder in folders.items():
        paths = glob.glob(f"{folder}/*.mp4")
        total = [0,0] ## normal , anom
        for p in paths:
            dur, tframes, fps = mp4_rgb_info(p)
            if "Normal" in p: total[0] += tframes
            else: total[1] += tframes
            
        log.info(f"UCF {key} STATS")
        log.info(f"{len(paths)} videos")
        log.info(f"TOTAL NORMAL FRAMES: {total[0]} {(total[0]/(total[0]+total[1]))*100:2f}%")
        log.info(f"TOTAL ANOMALY FRAMES: {total[1]} {(total[1]/(total[0]+total[1]))*100:2f}%\n\n")




####################
## PATHS AND SUCH
class FeaturePathListFinder:
    """
        From cfg_data.ds.info.froot/cfg_feat.id/ finds a folder with mode in it (train / test)
        then based on cfg_data procedes to filter the features paths
        so it retrieves accurate features list to use
        used in data/get_trainloader && data/get_testloader
    """
    def __init__(self, cfg_data, mode:str, modality:str):
        self.listANOM , self.listNORM = [], []
        
        cfg_dsinf = cfg_data.ds.info
        cfg_trnsfrm = cfg_data.trnsfrm.get(mode)
        cfg_feat = cfg_data.ds.get(f"f{modality}")

        mode = mode.upper()
        modality = modality.upper()
        
        ID_FEAT = cfg_feat.id
        fpath = cfg_dsinf.froot+'/'+modality+"/"+ID_FEAT+"/"+mode
        flist = glob.glob(fpath + '/*.npy')
        if not len(flist): 
            raise Exception (f'{fpath} has NADA')
        flist.sort()
        log.debug(f"feat flist pre-filt in {mode} {modality} : {len(flist)}")
        
        #################        
        ## if cropasvideo
        ##      if train, 
        ##          if cfg_data.ds.frgb.ncrops == to the amount of crops in the flist 
        ##              all features will treatead as a video so leave all fn in list
        ##          if != load only correspondant crop based cfg_data.trnsfrm.train.crops2use , 
        ##              eg if cfg_data.ds.frgb.ncrops == 1 and the flist contains 5 crops for each vid, form flist with flist excluding the __1.npy __2.npy __3.npy __4.npy crops for each video 
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## elif cfg_data,DATA.ncrops is set means that features files have crops even if only 1 is used
        ##      if train, take unique basename from crop fns -> feed each crop feat -> mean scores over crop dimension
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## else means that all files are a feature of video full view

        if modality == 'RGB' and mode == 'TRAIN':

            if cfg_trnsfrm.get("cropasvideo"):
                
                if cfg_trnsfrm.crops2use == cfg_feat.ncrops: ## get full list w/o ".npy"
                    flist = [f[:-4] for f in flist] 
                else: ## get only rigth crop idx
                    flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                    flist = [f"{f}__{i}" for f in flist for i in range(cfg_trnsfrm.train.crops2use)]
                    
            elif cfg_trnsfrm.crops2use: ## >= 1
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                
        elif modality == 'RGB' and mode == 'TEST': 
            
            if cfg_trnsfrm.crops2use == 1: ## == 1
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                
            else: ## ds w/ 1 rgbf per video
                flist = [f[:-4] for f in flist]
        
        ## aud 1 .npy per vid
        else: flist = [f[:-4] for f in flist]


        log.debug(f"feat flist pos-filt in {mode} {modality} : {len(flist)}")
        #log.warning(f"{flist}")
        
        ######
        ## filters into anom and norm w/ listNORM and listANOM
        ## filters for watching purposes w/ fn_label_dict
        self.fn_label_dict = {lbl: [] for lbl in cfg_dsinf.lbls_info[:-1]}
        fist = list(self.fn_label_dict.keys())[0]  ## 000.NORM
        last = list(self.fn_label_dict.keys())[-1] ## 111.ANOM

        for fp in flist:
            fn = osp.basename(fp)
            
            ## v=Z12t5h2mBJc__#1_label_B1-0-0
            if 'label_' in fn: xearc = fn[fn.find('label_'):]
            else: xearc = fn
            
            #log.debug(f'{fp} {xearc} ')
            
            if cfg_dsinf.lbls[0] in xearc: ## Assault018
                self.listNORM.append(fp)
                self.fn_label_dict[fist].append(fn)
                #log.debug(f'{xearc} {fn}')
            else:
                self.listANOM.append(fp)
                self.fn_label_dict[last].append(fn)

                for key in self.fn_label_dict.keys():
                    if key in xearc: 
                        self.fn_label_dict[key].append(fn)
                        #log.debug(f'{key} {fn}')
                        
        #for label, lst in self.fn_label_dict.items(): 
        #    log.debug(f'[{label}]: {len(self.fn_label_dict[label])}  ') ##{self.fn_label_dict[label]}
                
                
    def get(self, mode, watch_list=[]):
        if mode == 'ANOM': l = self.listANOM
        elif mode == 'NORM': l = self.listNORM
        elif mode == 'watch': 
            l = []
            for lbl2wtch in watch_list:
                ## find the labels that match the ones provided which need to be the nummberss prior to dot/.
                lbl2wtch = [lbl for lbl in list(self.fn_label_dict.keys()) if lbl.split('.')[0] == lbl2wtch][0]
                log.debug(f'{lbl2wtch} {len(self.fn_label_dict[lbl2wtch])}')
                l.extend(self.fn_label_dict[lbl2wtch])
        else: log.error(f'{mode} not found')
        return l
    
