import clip
import torch

import os, os.path as osp, glob, time, random


from src.data import get_videoloader
from src.utils import get_log
log = get_log(__name__)


class FeatExtract():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data.ds.info
        self.cfg_trnsfrm = cfg.data.trnsfrm

        self.model_dir = cfg.path.fext.models_dir
        self.out_dir = cfg.path.fext.out_dir
        self.get_backbone()
        
        self.DL = get_videoloader(cfg)
        
        self.start()

    def get_backbone(self):

        if self.cfg.model.id != "clip": raise NotImplementedError

        models = clip.available_models()
        log.info(models)

        if self.cfg.model.version not in models: raise Exception

        model_fn = self.cfg.model.version.replace("/", "-")+".pt"
        model_path = self.model_dir + model_fn
        log.info(model_path)
        if os.path.isfile(model_path):
            self.model, self.prepfx = clip.load(model_path)  
        else:
            self.model, self.prepfx = clip.load(self.cfg.model.version, device=self.cfg.dvc , download_root=self.model_dir)
        
        log.info(f"{self.model}   {self.model.visual.input_resolution}")
        assert self.cfg_trnsfrm.inpt_size == self.model.visual.input_resolution
    
    
    def start(self):
        
        extracted = 0; start_time = time.time()
        for vidx, data in enumerate(self.DL):
            clips = data[0]; vn = data[1]
            
            log.info(f'{vidx} {vn}')
            
            ## checks if video is already processed
            #out_npys = [osp.join(self.out_dir, f'{vn}' + (f'__{i}' if self.cfg_trnsfrm.ncrops > 1 else '') + '.npy') for i in range(self.cfg_trnsfrm.ncrops)]
            #log.info(out_npys)
            #if all([osp.exists(out_npy) for out_npy in out_npys]):
            #    log.info("All feature files for video " + vn + " already created.")
            #    continue
            
            feats = []
            for i, clip in enumerate(clips):
                #clip = clip.unsqueeze(0).to(self.cfg.dvc)
                clip = clip.to(self.cfg.dvc)
                log.error(f'fwd[{i}] {clip.shape} {clip.dtype} {type(clip)}')
                
                with torch.no_grad():
                    feat = self.model.encode_image(clip)
                    feats.extend(feat)
                    log.error(f'fwd[{i}] {clip.shape} -> {feat.shape}')

            feats = torch.stack(feats, axis=0)
            log.error(f'fwd[{i}] {feats.shape}')

            break

    