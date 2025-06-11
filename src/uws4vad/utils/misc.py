import os.path as osp
import time
import cv2

from uws4vad.utils.logger import get_log
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


