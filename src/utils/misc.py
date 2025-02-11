import os.path as osp
import time
import pickle

import cv2

from src.utils.logger import get_log
log = get_log(__name__)


def hh_mm_ss(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def mp4_rgb_info(vpath):
    if osp.exists(vpath):
        vv = cv2.VideoCapture(vpath)
        fps = vv.get(cv2.CAP_PROP_FPS)
        tframes = vv.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = vv.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        vv.release()
        cv2.destroyAllWindows()
        return dur,int(tframes),fps
    else: raise Exception(f"{vpath} no exist")


######################
## vldt/watch_info .pkl i/o
def load_pkl(path, wut):
    if wut == 'watch':
        p = osp.join(path,'watch_info.pkl')
        if not osp.exists(p):
            raise Exception(f"none {p} -> run once with cfg.TEST.WATCH.SAVEPKL: true / .FROMPKL: false")
        with open(p, 'rb') as f: data = pickle.load(f)
        return data
    
    elif wut == 'vldt':
        p = osp.join(path,'vldt_info.pkl')
        if not osp.exists(p):
            raise Exception(f"none {p} -> run once with cfg.TEST.VLDT.SAVEPKL: true / .FROMPKL: false")
        with open(p, 'rb') as f: data = pickle.load(f)
        return data 
    else: raise Exception(f"wut must be watch or vldt")

def save_pkl(path,data,wut):
    if wut == 'watch':
        p = osp.join(path,'watch_info.pkl')
        with open(p, 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    elif wut == 'vldt':
        p = osp.join(path,'vldt_info.pkl')
        with open(p, 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else: raise Exception(f"wut must be watch or vldt")
