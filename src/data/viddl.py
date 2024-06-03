import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, Resize, TenCrop, FiveCrop, CenterCrop, ToTensor, Normalize, ToPILImage, Lambda
from torchvision.transforms.functional import to_pil_image
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import decord
from decord import VideoReader
decord.bridge.set_bridge('torch')
import cv2


import os, os.path as osp, glob, time, random

from src.utils import get_log
log = get_log(__name__)



def get_videoloader(cfg, trnsfrm = None):
    cfg_loader = cfg.dataloader.test
    return DataLoader(  
                    VideoDS(cfg), 
                    batch_size=cfg_loader.bs , 
                    shuffle=False, #cfg_loader.shuffle, 
                    num_workers=0, #cfg_loader.nworkers, 
                    prefetch_factor=cfg_loader.pftch_fctr,
                    pin_memory=cfg_loader.pinmem, 
                    persistent_workers=cfg_loader.prstwrk 
                    )


class VideoRecord:
    def __init__(self, vpath, cfg_trnsfrm):
        self.vpath = vpath
        self.vname = osp.splitext(osp.basename(vpath))[0]
        log.error(vpath)
        
        ###############
        vid2 = cv2.VideoCapture(vpath)
        if not vid2.isOpened(): raise ValueError(f"Failed to open video {vpath}")
        self.fps = int(vid2.get(cv2.CAP_PROP_FPS))
        self.tframes = int(vid2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w = vid2.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid2.release()
        log.info(f'{self.vname}  {str(self.fps)} fps | {str(self.tframes)} frames {self.h}*{self.w}')
        ##############
        
        #if self.cfg_trnsfrm.aug == 'v0': vid = VideoReader(vpath)
        #elif self.cfg_trnsfrm.aug == 'v1': vid = VideoReader(vpath, width=self.cfg_trnsfrm.new_width, height=self.cfg_trnsfrm.new_height)
        self.vid = VideoReader(vpath)
        self.vid_len = len(self.vid)
        assert self.vid_len == self.tframes, f'{self.vid_len} != {self.tframes}'
        
        self.frame_step = cfg_trnsfrm.frame_step
        self.clip_len = cfg_trnsfrm.clip_len
        if cfg_trnsfrm.id == 'clip': self.clip_sel = cfg_trnsfrm.clip_sel
        else: self.clip_sel = None
        
        self.get_vidxs()
    
    def get_vidxs(self):
        
        vidxs = list(range(0, self.vid_len, self.frame_step)) ## grab vidxs by frame step
        if self.vid_len < self.clip_len: 
            r = self.clip_len // self.vid_len + 1
            vidxs = (vidxs * r)[:self.clip_len]
        
        nclips = int(len(vidxs) / self.clip_len)
        log.info(f'yielding {nclips*self.clip_len} frames in {nclips} clips  w/ {self.clip_len} out of {len(vidxs)} = ({str(self.vid_len)} frames/{self.frame_step} fstep)')
        if nclips * self.clip_len < len(vidxs):
            log.info(f'discarting {len(vidxs) - (nclips * self.clip_len)} ')
        
        self.cidxs_bat = []
        for k in range(nclips):
            
            cidxs = vidxs[k * self.clip_len : (k+1) * self.clip_len]
            log.error(f"[{k}]  pre {cidxs}")
            
            if self.clip_sel == "rnd": ## grab random from each clip
                tmp = [cidxs[ random.randint(0,len(cidxs)-1) ]]
            elif self.clip_sel == 'mid': ## grab middle frame from each clip
                tmp = [cidxs[ (len(cidxs) // 2)-1 ]] 
            else: ## cnn based need full clip as inpt
                tmp = cidxs 
            
            self.cidxs_bat.append( tmp )
            log.error(f"[{k}]  pst {tmp}")            


class VideoDS(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data.info
        self.cfg_trnsfrm = cfg.data.trnsfrm
        
        ####################
        self.vpaths = glob.glob(self.cfg_dsinf.vroot + "/TEST/*.mp4")[:4]
        self.vpaths.sort()
        log.info( self.vpaths )
        
        self.vrecords = [VideoRecord(vp, self.cfg_trnsfrm) for vp in self.vpaths]
        
        self.get_transform()
        
    def _convert_image_to_rgb(self, images):
        return image.convert("RGB")
    
    def get_transform(self):
        #mean = (0.48145466, 0.4578275, 0.40821073)
        #std = (0.26862954, 0.26130258, 0.27577711)
    
        if self.cfg_trnsfrm.ncrops == 10:
            raise NotImplementedError
            self.trnsfrm = Compose([
                #ToPILImage("RGB"),
                Resize(self.cfg_trnsfrm.inpt_size, interpolation=BICUBIC),
                TenCrop(self.cfg_trnsfrm.inpt_size),  # This will create 5 crops of the image at each corner and the center
                Lambda(lambda crops: torch.stack([ToTensor()(self._convert_image_to_rgb(crop)) for crop in crops])), # Convert each crop to tensor
                Lambda(lambda tensors: torch.stack([Normalize(self.cfg_trnsfrm.mean, self.cfg_trnsfrm.std)(t) for t in tensors])),
            ])
        elif self.cfg_trnsfrm.ncrops == 5:
            raise NotImplementedError
            self.trnsfrm = Compose([
                #ToPILImage("RGB"),
                Resize(self.cfg_trnsfrm.inpt_size, interpolation=BICUBIC),
                FiveCrop(self.cfg_trnsfrm.inpt_size),  # This will create 5 crops of the image at each corner and the center
                #Lambda(lambda crops: torch.stack([ToTensor()(self._convert_image_to_rgb(crop)) for crop in crops])), # Convert each crop to tensor
                Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])), # Convert each crop to tensor
                Lambda(lambda tensors: torch.stack([Normalize(self.cfg_trnsfrm.mean, self.cfg_trnsfrm.std)(t) for t in tensors])),
            ])
        else:
            self.trnsfrm = Compose([
                #ToPILImage("RGB"), 
                Resize(self.cfg_trnsfrm.inpt_size, interpolation=BICUBIC),
                CenterCrop(self.cfg_trnsfrm.inpt_size),
                #self._convert_image_to_rgb,
                ToTensor(),
                Normalize(self.cfg_trnsfrm.mean, self.cfg_trnsfrm.std),
            ])
        log.warning(self.trnsfrm)
    
    
    def _get(self, vrec):
        
        clips = []
        for k, cidx in enumerate(vrec.cidxs_bat):
            
            log.warning(f"{k}  {cidx}")
            
            if len(cidx) == 1: ## means  that clip only has 1 frame idx
                frames = vrec.vid[ cidx[0] ].permute(2,0,1) ## (H, W, 3) -> c * h * w - expected per ToPILImage
                log.error(f'dcord[{k}] {frames.shape} {frames[0].dtype} {type(frames)}')
                
                frames = to_pil_image(frames, 'RGB') ##
                log.error(f'pil[{k}] {type(frames)}')
                
                frames = self.trnsfrm(frames) 
                #frames = self.prepfx(frames)
                log.error(f'trnsfrm[{k}] {frames.shape} {frames[0].dtype} {type(frames)}')
                
                #if cfg.viewer: view_crops_trnsf(frames,self.cfg_trnsfrm.ncrops)
                clips.append(frames)
                
            else: ## normal input to cnn
                frames = vrec.vid.get_batch(cidx) ## (new_length, H, W, 3)
                log.error(f'dcord[{k}] {frames.shape} {frames[0].dtype} {type(frames)}')
                ## not tested yet.....
                
        return clips
    
    def __getitem__(self, idx):
        log.error( self.vpaths[idx] )
        
        vrec = self.vrecords[idx]
        clips = self._get(vrec)
        log.error(f"{len(clips)}")
        
        return clips, vrec.vname
    
    def __len__(self):
        return len(self.vpaths)