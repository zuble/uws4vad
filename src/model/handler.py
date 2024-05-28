import os, os.path as osp
from collections import OrderedDict
import copy
import math

import torch
import torch.nn
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils import get_log 
log = get_log(__name__)

def weight_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def count_parms(net):
    t = sum(p.numel() for p in net.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')
    
#log.info(f"Net structure: {net}\n\n") if 'blck' in cfg.NET.LOG_INFO else None 
#if 'prmt' in cfg.NET.LOG_INFO:
#    for name, param in net.named_parameters():
#        log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def save(cfg, net):
    ## https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    ## https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
    PATH = osp.join(cfg.EXPERIMENTPATH,f'{cfg.NET.NAME}')

    try: 
        ## net arch&params
        p = f'{PATH}_{cfg.SEED}.net.pt'
        torch.save(net, p)
        log.info(f'µµµ SAVED full netowrk {cfg.NET.NAME}.pt in \n{p}')
    except Exception as e: log.error(f'{cfg.NET.NAME}.export: {e}')  
    
    try:    
        ## net params / state_dict (needs same net def) 
        p =  f'{PATH}_{cfg.SEED}.dict.pt'
        torch.save(net.state_dict(), p)
        log.info(f'µµµ SAVED state_dict of {cfg.NET.NAME}.pt in \n{cfg.EXPERIMENTPATH}')
    except Exception as e: log.error(f'{cfg.NET.NAME}-state_dict.save: {e}')  
    
def get_ldnet(cfg, dvc):
    
    if not cfg.TEST.LOADFROM: raise Exception (f'cfg.TEST.LOADFROM must be set')
    
    load_dict = parse_ptfn(cfg.TEST.LOADFROM)
    
    PATH = osp.join(cfg.EXPERIMENTPATH, f"{cfg.TEST.LOADFROM}.pt")
    log.info(f"loading {load_dict['mode']} from {PATH}")
    
    if load_dict['mode'] == 'net': 
        net = torch.load(PATH)
    elif load_dict['mode'] == 'dict':
        net = get_net(cfg, dvc)
        net.load_state_dict( torch.load(PATH) )
    else: raise Exception(f'{cfg.TEST.LOADFROM = } and should be (...).["net","dict"]')
    
    return net, load_dict['seed']


class ModelHandler:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def save_net(self, net, trn_inf, save_file=True):
        
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.to("cpu")
        if save_file:
            fname = f"{cfg.seed}-{trn_inf['epo']}_{trn_inf['step']}.pt"
            path = osp.join(self.cfg.log.chkpt_dir, fname)
            torch.save(state_dict, save_path)
            log.info("Saved network checkpoint to: %s" % save_path)
        return state_dict
    
    def load_network(self, net_arch, model_state=None):
        
        if model_state is None:
            model_state = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.cfg.dvc),
            )
        model_clean_state = OrderedDict()  # remove unnecessary 'module.'
        for k, v in model_state.items():
            if k.startswith("module."):
                model_clean_state[k[7:]] = v
            else:
                model_clean_state[k] = v

        net_arch.load_state_dict(model_clean_state, strict=self.cfg.load.strict_load)
        log.info(
            "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path
        )
        return net_arch
        
    def save_training_state(self, net, optima):
        tmp = f"{cfg.seed}-{trn_inf['epo']}_{trn_inf['step']}.state"
        save_path = osp.join(self.cfg_path.work_dir, tmp)
        
        net_state_dict = self.save_net(False)
        state = {
            "model": net_state_dict,
            "optima": optima.state_dict(),
            "step": trn_inf['step'],
            "epo": trn_inf['epo'],
        }
        torch.save(state, save_path)
        log.info("Saved training state to: %s" % save_path)

    def load_training_state(self, net_arch, optima, trn_inf):
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.cfg.dvc),
        )
        
        net_loaded = self.load_network(net_arch, state=resume_state["model"])
        optima.load_state_dict(resume_state["optima"])
        trn_inf["step"] = resume_state["step"]
        trn_inf["epo"] = resume_state["epo"]
        
        log.info(
            "Resuming from training state: %s" % self.cfg.load.resume_state_path
        )
        return net_loaded, optima 
