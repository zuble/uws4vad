import numpy
import torch
from torch.utils.data import DataLoader, Dataset

import decord
from decord import AudioReader, cpu
decord.bridge.set_bridge('torch')
import cv2

import os, os.path as osp, glob, time, random
from src.fe.utils import print_acodec_from_mp4
from src.utils import get_log
log = get_log(__name__)



def get_audloader(cfg, cfg_model, vpaths):
    cfg_loader = cfg.dataloader.test
    return DataLoader(  
                    AudioDS(cfg, cfg_model, vpaths), 
                    batch_size=1, 
                    shuffle=False,
                    num_workers=0,#cfg_loader.nworkers, 
                    pin_memory=False, 
                    )

class AudioDS(Dataset):
    def __init__(self, cfg, cfg_model, vpaths):
        self.cfg = cfg
        self.cfg_model = cfg_model
        self.sr = cfg_model.sr
        
        self.vpaths = vpaths
        assert len(self.vpaths) > 0, "No video found in the provided paths"
        log.info(f"{len(self.vpaths)=}")
        
        self.records = [AudioRecord(vp, cfg_model) for vp in self.vpaths]
        log.info(f"{len(self.records)=}")
    
    def __getitem__(self, idx):
        #log.debug(f"{self.vpaths[idx]}")
        arec = self.records[idx]
        #aud = AudioReader(self.vpaths[idx], ctx=cpu(0), sample_rate=self.sr, mono=True)
        #log.info(f'{aud.shape} {type(aud[0:-1])}')    
        return arec.aud, arec.vname, arec.fps
    
    def __len__(self):
        return len(self.vpaths)


class AudioRecord:
    def __init__(self, vpath, cfg_model):
        fstep = 1
        clip_len = cfg_model.clip_len
        
        ###############
        vid2 = cv2.VideoCapture(vpath)
        if not vid2.isOpened(): raise ValueError(f"Failed to open video {vpath}")
        self.fps = int(vid2.get(cv2.CAP_PROP_FPS))
        vid_len = int(vid2.get(cv2.CAP_PROP_FRAME_COUNT))
        vid2.release()
        #if self.cfg.data.get("fps"): assert self.cfg.data.fps == arec.fps, f"fps mismatch {self.cfg.data.fps} != {arec.fps}"
        sr_og = print_acodec_from_mp4([vpath],only_sr=True)
        log.debug(f'{osp.splitext(osp.basename(vpath))[0]}  {str(self.fps)} fps | {str(vid_len)} frames | {sr_og} Hz')
        ##############

        ## reuses rgb func to gen same idxs 
        ## and drop same amnout of samples as frames
        #vidxs = list(range(0, vid_len, fstep))
        #if vid_len < clip_len: 
        #    raise NotImplementedError
        #    #r = clip_len // vid_len + 1
        #    #vidxs = (vidxs * r)[:clip_len]
        #
        #frames_yield = int(len(vidxs) / clip_len) * clip_len
        #secs_yield = frames_yield / self.fps
        #
        #samples2yield = int(secs_yield * sr_og)
        #log.debug(f'{frames_yield=}, {secs_yield=} {samples2yield=}')
        
        #aud = AudioReader(vpath, ctx=cpu(0), sample_rate=-1, mono=True)
        #self.aud = aud[0, :samples2yield]
        
        
        ## assume possible clip_len-1 frames difference betwen frgb are not impactful        
        aud = AudioReader(vpath, ctx=cpu(0), sample_rate=cfg_model.sr, mono=True)
        log.debug(f'ar: {aud.shape} {type(aud[0:-1])}')

        self.aud = aud[0:-1]
        self.vname = osp.splitext(osp.basename(vpath))[0]        