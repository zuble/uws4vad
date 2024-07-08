import torch
import numpy as np
import timm
from timm.data import resolve_data_config, create_transform

from torchvision.transforms import Compose, Resize, TenCrop, FiveCrop, CenterCrop, ToTensor, Normalize, ToPILImage, Lambda
from PIL import Image

import os, os.path as osp, glob, time, random
import hydra
from hydra.utils import instantiate as instantiate

from src.fe.dlvid import get_vidloader
from src.utils import get_log
log = get_log(__name__)



def get_vid_model(cfg):
    model_dir = cfg.path.fext.models_dir
    cfg_model = cfg.data.frgb
    
    if cfg_model.id == "timm":
        # https://github.com/huggingface/pytorch-image-models/discussions/2069
        
        log.info("timm model")
        model_names = timm.list_models('*clip*', pretrained=True)
        for m in model_names:
            log.info(f"{m}")
        if cfg_model.vrs not in model_names: raise Exception
        
        model = timm.create_model(f"{cfg_model.id}/{cfg_model.vrs}", pretrained=True, num_classes=0)
        model.to(cfg.dvc)
        model.eval()
        
        cfg_trnsfrm = resolve_data_config(model.pretrained_cfg)
        trnsfrm = Compose([
            lambda arr: Image.fromarray(arr, mode='RGB'), 
            create_transform(**cfg_trnsfrm),
            #lambda x: x.unsqueeze(0)
        ])
        log.debug(f"timm {cfg_trnsfrm=} \n{trnsfrm=}")
        if cfg.dr: log.warning(model(trnsfrm(torch.randn(224, 224, 3).numpy()).unsqueeze(0)).shape)
        
        return model, trnsfrm, cfg_model
        
    
    elif cfg_model.id == "clip":
        import clip
        log.info("OAICLIP model")
        models = clip.available_models()
        for m in models:
            log.info(f"{m}")
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
        
        model, prepfx = clip.load(model_path, device=cfg.dvc, download_root=model_dir)  
        model.eval()
        
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
        
        return model.encode_image, trnsfrm, cfg_model
        #model.encode_text
        




## ----------------------- ##
def get_aud_model(cfg):
    cfg_model = cfg.data.faud
    log.info(f"{cfg_model=}")

    if cfg_model.id == 'efat':
    
        model = instantiate(cfg.data.faud.model)#, _convert_="partial", _partial_=True)
        model.to(cfg.dvc)
        model.net.eval() 
        
        assert int(cfg_model.hop * cfg_model.sr / 1000) == model.timestamp_hop
        
        #from src.fe.hear_mn import mn10_all_b as mn10_MB
        #from src.fe.hear_mn import mn10_all_b_all_se as mn10_MB_MSE
        #if cfg_model.vrs == '10_MB':
        #    model = mn10_MB.load_model()
        #elif cfg_model.vrs == '10_MB_MSE':
        #    model = mn10_MB_MSE.load_model()
        #else: raise NotImplementedError
        
        if cfg.dr:
            vids=1; secs=32; sr=cfg_model.sr; fps=cfg.data.fps
            ## n_sounds * n_samples
            audio = torch.ones((vids, int(sr * secs)))
            ## (vids, t, d) // (vids, t)
            embeds, timestamps = model.get_timestamp_embeddings(audio) 
            log.info(f"{embeds.shape=} {timestamps.shape=} {embeds}")
            ## timestamps = [model.timestamp_hop*0, model.timestamp_hop*1, .., secs*1000]
            
            ## asert feats so each element corresponds to same fraction as clip_len on rgb feats
            clip_dur = cfg.data.faud.clip_len / fps
            samples_per_clip = clip_dur * sr
            timestamps_per_clip = int(samples_per_clip / model.timestamp_hop)
            log.info(f"{timestamps_per_clip=}")
            
            clip_starts = range(0, timestamps.shape[-1], timestamps_per_clip)
            tmp = [embeds[0][s:s+timestamps_per_clip].mean(dim=0) for s in clip_starts]
            sl_embeds = torch.stack(tmp)
            log.info(f"{sl_embeds.shape=} {sl_embeds}") 
    
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