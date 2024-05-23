import torch
import numpy as np

import glob , os, os.path as osp, math, time
from collections import OrderedDict

from .testdl import init as init1
from .traindl import init as init2

from utils import mp4_rgb_info, LoggerManager

log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)
    init1(log); init2(log)

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
                    log.info(f'B[{b_idx+1}] (seg) NORM {nfeat.shape=} {nfeat.dtype} | {nlabel.shape=} {nlabel[0]} {nlabel.dtype}')
                    log.info(f'B[{b_idx+1}] (seg) ABNORM {afeat.shape=} {afeat.dtype} | {alabel.shape=} {alabel[0]} {alabel.dtype}')
                else:
                    cfeat, label = data
                    log.info(f'B[{b_idx+1}] (seq) {cfeat.shape=} {cfeat.dtype} | {label.shape=} {label.dtype}')
            log.info(f'[{ii}] time to load {b_idx+1} batches {time.time()-tic}')
    except ValueError as e:
        log.error(f'Error unpacking variables in batch {b_idx+1}: {e}')


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
        From cfg_ds.FROOT/featdirname/ finds a folder with mode in it (train / test)
        then based on cfg procedes to filter the features paths
        so it retrieves accurate features list to use
        used in data/get_trainloader && data/get_testloader
    """
    def __init__(self, cfg, mode:str, modality:str, featdirname:str, cfg_ds, cfg_feat):
        self.listANOM , self.listNORM = [], []
        
        #if not cfg_ds:
        #    if mode in 'train': cfg_ds = getattr(cfg.DS, cfg.TRAIN.DS)
        #    ## test mode is used by vldt/Validate on 2 scenarios:
        #    ## train for validation and test
        #    ## it must be specified in order to get the right cfg_ds
        #    ## as train and test might have different ds to operate on
        #    else: raise Exception("cfg_ds is None with mode in test")
        
        fpath = ''
        for root, dirs, _ in os.walk(cfg_ds.FROOT):
            if featdirname == osp.basename(root):
                for d in dirs:
                    if mode in d:
                        #log.info(f'{d}')
                        fpath = osp.join(root, d)
                        log.info(f'{mode} features path {fpath}')
                        break        
        if not fpath: 
            raise Exception (f'{featdirname} or {featdirname}/{mode} not found in {cfg_ds.FROOT=}')

        flist = glob.glob(fpath + '/**.npy')
        flist.sort()
        log.debug(f"feat flist pre-filt in {mode} {modality} : {len(flist)}")
        
        #################        
        ## if cropasvideo
        ##      if train, 
        ##          if cfg.DATA.RGB.NCROPS == to the amount of crops in the flist 
        ##              all features will treatead as a video so leave all fn in list
        ##          if != load only correspondant crop, eg if cfg.DATA.RGB.NCROPS == 1 and the flist contains 5 crops for each vid, form flist with flist excluding the __1.npy __2.npy __3.npy __4.npy crops for each video 
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## elif cfg,DATA.NCROPS is set means that features files have crops even if only 1 is used
        ##      if train, take unique basename from crop fns -> feed each crop feat -> mean scores over crop dimension
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## else means that all files are a feature of video full view

        if modality.upper() == 'RGB':
            
            ## assert cases when there no crops
            ## not ideal, but both train/test depend on it
            ## mainly for TrainFrmt.reshape_in and NetPstFwd.reshape_*, although tackle that edges
            if cfg_feat.NCROPS == 0:
                c2u = getattr( getattr(cfg, mode.upper()), 'CROPS2USE')
                if c2u  != 0:
                    log.warning(f"cfg_f{modality.lower()}.{featdirname} has no crops, while cfg.{mode.upper()}.CROPS2USE is {c2u}")
                    log.warning(f"overriding cfg.{mode.upper()}.CROPS2USE to 0")
                    cfg.merge_from_list([f"{mode.upper()}.CROPS2USE", 0])
            
            
            if cfg.TRAIN.CROPASVIDEO and mode in 'train':
                
                if cfg.TRAIN.CROPS2USE == cfg_feat.NCROPS:
                    flist = [f[:-4] for f in flist]
                else:
                    flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                    flist = [f"{f}__{i}" for f in flist for i in range(cfg.TRAIN.CROPS2USE)]
                    
                ### Get the unique video identifiers from the first cfg.DATA.RGB.NCROPS * 2 files
                #unique_video_ids = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist[:cfg.DATA.RGB.NCROPS * 2]]))
                ### if len is 2: ncrops corresponds to cfg.DATA.RGB.NCROPS select all
                #if len(unique_video_ids) == 2: flist = [f[:-4] for f in flist]
                ### selects only the frist cfg.DATA.RGB.NCROPS crop files
                #else:
                #    flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                #    flist = [f"{f}__{i}" for f in flist for i in range(cfg.DATA.RGB.NCROPS)]
            
            ## cfg.DATA.CROPASVIDEO is False or the mode is "test"
            ## check for use of crops
            elif mode in 'test' and cfg.TEST.CROPS2USE:
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                
            elif mode in 'train' and cfg.TRAIN.CROPS2USE:
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
            
            ## ds w/ 1 rgbf per video
            else: flist = [f[:-4] for f in flist]
        
        ## aud 1 .npy per vid
        else: flist = [f[:-4] for f in flist]


        log.debug(f"feat flist pos-filt in {mode} {modality} : {len(flist)}")
        #log.warning(f"{flist}")
        
        ######
        ## filters into anom and norm w/ listNORM and listANOM
        ## filters for watching purposes w/ fn_label_dict
        self.fn_label_dict = {lbl: [] for lbl in cfg_ds.LBLS_INFO[:-1]}
        fist = list(self.fn_label_dict.keys())[0]  ## 000.NORM
        last = list(self.fn_label_dict.keys())[-1] ## 111.ANOM

        for fp in flist:
            fn = osp.basename(fp)
            
            ## v=Z12t5h2mBJc__#1_label_B1-0-0
            if 'label_' in fn: xearc = fn[fn.find('label_'):]
            else: xearc = fn
            
            #log.debug(f'{fp} {xearc} ')
            
            if cfg_ds.LBLS[0] in xearc:
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
    
