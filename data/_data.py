import torch
import numpy as np

import glob , os, os.path as osp , math , time

from .testdl import init as init1
from .traindl import init as init2

from utils import FeaturePathListFinder, mp4_rgb_info, Visualizer, LoggerManager

log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)
    init1(log); init2(log)



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


###############################
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

