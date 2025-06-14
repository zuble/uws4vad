import torch
import numpy as np
import timm
from timm.data import resolve_data_config, create_transform

from torchvision.transforms import Compose, Resize, TenCrop, FiveCrop, CenterCrop, ToTensor, Normalize, ToPILImage, Lambda
from PIL import Image

import os, os.path as osp, glob, time, random, hydra, tabulate
from hydra.utils import instantiate as instantiate

from uws4vad.fe.dlvid import get_vidloader
from uws4vad.utils import get_log
log = get_log(__name__)


def get_vid_model(cfg):
    model_dir = cfg.path.fext.models_dir
    cfg_model = cfg.data.frgb

    def print_timm(model_names):
        n_cols=4
        chunked = [model_names[i:i + n_cols] for i in range(0, len(model_names), n_cols)]
        log.info(f"\n {tabulate(chunked, tablefmt='fancy_grid')}")
    
    if cfg_model.id == "timm":
        # https://github.com/huggingface/pytorch-image-models/discussions/2069
        log.info("timm model")

        if cfg_model.vrs:
            all_models = timm.list_models(pretrained=True)
            if cfg_model.vrs in all_models:
                model = timm.create_model(cfg_model.vrs, pretrained=True, num_classes=0) 
            else:
                filtered = timm.list_models(cfg_model.vrs, pretrained=True)
                log.error(f"'{cfg_model.vrs}' invalid. Options:")
                print_timm(filtered)
                raise ValueError(f"Set cfg_model.vrs to valid model")
        else:
            print_timm(timm.list_models(pretrained=True))
            raise ValueError("cfg_model.vrs required")

        # if cfg_model.vrs: 
        #     model_names = timm.list_models(filter=cfg_model.vrs, pretrained=True)
        #     if len (model_names) == 1: pass
        #     elif not model_names: error=f"Model '{cfg_model.vrs}' not found"; print_timm(timm.list_models(pretrained=True))
        #     else: error=f"Fill '{cfg_model.vrs=}'"; print_timm(model_names)
        #     raise ValueError(error)
        # else: print_timm(timm.list_models(pretrained=True));raise ValueError(f"Fill '{cfg_model.vrs=}'")

        ## TODO add cfg parameter to model & direct load from saved weights
        # model = timm.create_model(f"{cfg_model.id}/{cfg_model.vrs}", checkpoint_path=)
        torch.hub.set_dir(osp.join(cfg.path.data_dir,"fe/rgb_timm"))        
        model = timm.create_model(f"{cfg_model.id}/{cfg_model.vrs}", pretrained=True, num_classes=0)
        #print(dir(model))
        model.to(cfg.dvc)
        model.eval()
        
        cfg_trnsfrm = resolve_data_config(model.pretrained_cfg)
        trnsfrm = Compose([
            lambda arr: Image.fromarray(arr, mode='RGB'), 
            create_transform(**cfg_trnsfrm),
            #lambda x: x.unsqueeze(0)
        ])
        log.debug(f"timm {cfg_trnsfrm=} \n{trnsfrm=}")

        frame = trnsfrm(torch.randn(224, 224, 3).numpy()).unsqueeze(0)
        log.debug(f'trnsfrm {frame.shape} {frame.dtype} {type(frame)}')
        
    elif cfg_model.id == "clipai":
        import clip
        log.info("OAICLIP model")
        models = clip.available_models()
        for m in models: log.info(f"{m}")
        version = {
            'vitb16': "ViT-B/16",
            'vitb32': "ViT-B/32"
        }.get(cfg_model.vrs)
        if version not in models: raise Exception
        
        model_path = model_dir + version.replace("/", "-")+".pt"
        if not os.path.isfile(model_path): 
            log.warning(f"{model_path} not found, downloading {version}")
            model_path = version
        else: log.info(f"loading from {model_path}")
        
        _model, prepfx = clip.load(model_path, device=cfg.dvc, download_root=model_dir)  
        _model.eval()
        model = _model.encode_image
    
        assert cfg_model.inpt_size == model.visual.input_resolution, f"{cfg_model.inpt_size} != {model.visual.input_resolution}"

        #https://pc-pillow.readthedocs.io/en/latest/Image_class/Image_fromarray.html
        trnsfrm = Compose([
            lambda arr: Image.fromarray(arr),
            Resize(cfg_model.inpt_size, interpolation=Image.BICUBIC),
            CenterCrop(cfg_model.inpt_size),
            lambda image: image.convert('RGB'),
            ToTensor(),
            Normalize(cfg_model.mean, cfg_model.std),
        ])
        log.error(f"{trnsfrm=} {prepfx=}")
        
        frame = transfrm(torch.randn(3, 224, 224).numpy()).unsqueeze(0)
        log.debug(f'trnsfrm {frame.shape} {frame.dtype} {type(frame)}')
        
        #model.encode_text
    
    else: raise NotImplementedError
    
    ############
    ## PROFILING
    # TODO: the same is done in the end of network creation (model/net/_builder.py)
    if cfg.dryfwd:
        out = model(frame)       
        log.warning(f"DRY RUN: {frame.shape} -> {out.shape} {out.dtype}")
    
    if cfg.profile:
        from uws4vad.utils import prof
        bs = 2; t = 16; clip = []
        for i in range(0, t): clip.extend(frame)   
        clip = torch.stack(clip,dim=0)
        clips = clip.repeat(bs, 1, 1, 1, 1)  ## bs,t,c,h,w        
        log.warning(f"PROFILING: Simulating {bs}x{t}frames {clips.shape=}")
        prof(model, inpt_data=clips)    
    
    if cfg.summary: 
        from uws4vad.utils import info, flops
        flops(model, inpt_data=frame)
        #info(model, inpt_size=None, inpt_data=frame )
    
    return model, trnsfrm, cfg_model


## ----------------------- ##
def get_aud_model(cfg):
    cfg_model = cfg.data.faud
    log.info(f"{cfg_model=}")

    if cfg_model.id == 'efat':
    
        model = instantiate(cfg.data.faud.model)#, _convert_="partial", _partial_=True)
        model.to(cfg.dvc)
        model.net.eval() 
        
        assert int(cfg_model.hop * cfg_model.sr / 1000) == model.timestamp_hop
        
        #from uws4vad.fe.hear_mn import mn10_all_b as mn10_MB
        #from uws4vad.fe.hear_mn import mn10_all_b_all_se as mn10_MB_MSE
        #if cfg_model.vrs == '10_MB':
        #    model = mn10_MB.load_model()
        #elif cfg_model.vrs == '10_MB_MSE':
        #    model = mn10_MB_MSE.load_model()
        #else: raise NotImplementedError
        
        if cfg.dry_run:
            vids=1; secs=32; sr=cfg_model.sr; fps=cfg.data.fps
            ## n_sounds * n_samples
            audio = torch.ones((vids, int(sr * secs)))
            ## (vids, t, d) // (vids, t)
            embeds, timestamps = model.get_timestamp_embeddings(audio) 
            log.info(f"{embeds.shape=} {timestamps.shape=} {embeds}")
            ## timestamps = [model.timestamp_hop*0, model.timestamp_hop*1, .., secs*1000]
    else: raise NotImplementedError
        
    return model, cfg_model

#################################
#if self.cfg_trnsfrm.ncrops == 10:
#    raise NotImplementedError
#    self.trnsfrm = Compose([
#        #ToPILImage("RGB"),
#        Resize(self.cfg_model.inpt_size, interpolation=BICUBIC),
#        TenCrop(self.cfg_model.inpt_size),  # This will create 5 crops of the image at each corner and the center
#        Lambda(lambda crops: torch.stack([ToTensor()(self._convert_image_to_rgb(crop)) for crop in crops])), # Convert each crop to tensor
#        Lambda(lambda tensors: torch.stack([Normalize(self.cfg_model.mean, self.cfg_model.std)(t) for t in tensors])),
#    ])
#elif self.cfg_trnsfrm.ncrops == 5:
#    raise NotImplementedError
#    self.trnsfrm = Compose([
#        #ToPILImage("RGB"),
#        Resize(self.cfg_model.inpt_size, interpolation=BICUBIC),
#        FiveCrop(self.cfg_model.inpt_size),  # This will create 5 crops of the image at each corner and the center
#        #Lambda(lambda crops: torch.stack([ToTensor()(self._convert_image_to_rgb(crop)) for crop in crops])), # Convert each crop to tensor
#        Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])), # Convert each crop to tensor
#        Lambda(lambda tensors: torch.stack([Normalize(self.cfg_model.mean, self.cfg_model.std)(t) for t in tensors])),
#    ])
#else:
#    self.trnsfrm = Compose([
#        #ToPILImage("RGB"), 
#        Resize(self.cfg_model.inpt_size, interpolation=BICUBIC),
#        CenterCrop(self.cfg_model.inpt_size),
#        #self._convert_image_to_rgb,
#        ToTensor(),
#        Normalize(self.cfg_model.mean, self.cfg_model.std),
#    ])
