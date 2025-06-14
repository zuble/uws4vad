import torch
from multiprocessing import cpu_count
import os.path as osp
from typing import Any, Dict, Callable
from functools import wraps
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose

from uws4vad.utils.logger import get_log
log = get_log(__name__)

## hydra cfg resolvers
## decorator at main
def dyn_dvc(dvc):
    if dvc != -1 and torch.cuda.is_available():
        tmp = torch.cuda.device_count()
        if dvc >= tmp-1: 
            dvc = tmp-1
            #log.warning(f"{cfg.dvc=} , torch acess to {dvc} gpu")
        return f'cuda:{dvc}' #torch.device()
    else: return 'cpu'
    #if cfg.gpuseton:
    #    gpus = subprocess.check_output(['nvidia-smi'], text=True).count('%')
    #    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(gpus))) ## '0,...,gpus-1'
    #log.info(f"{cfg.dvc=} , torch acess to {torch.cuda.device_count()} gpu")

def dyn_nworkers(nwkrs):
    if nwkrs > 0: return nwkrs
    elif nwkrs == 0: return int(cpu_count() / 4)
    elif nwkrs == -1: return cpu_count()


def dyn_vldt(epochs,x):
    if x == 0 or x > epochs: return 1
    else: return epochs // x

def dyn_vldtmtrc(ds, mtrc):
    if not mtrc:
        if ds == 'ucf': return 'AUC_ROC'
        elif ds == 'xdv': return 'AUC_PR'
    return mtrc    

def dyn_vldtfwd(netid, chuksiz):
    if netid == 'anm': return chuksiz
    else: return None


def dyn_fencrops(ds_id, nc_ucf, nc_xdv):
    return nc_ucf if ds_id == 'ucf' else nc_xdv
    
def dyn_crops2use(frgb_ncrops, crops2use_stg):
    if frgb_ncrops == 0: crops2use = 0
    elif crops2use_stg == -1:
        crops2use = frgb_ncrops  # Use the maximum number of crops
    elif 1 <= crops2use_stg <= frgb_ncrops:
        crops2use = crops2use_stg  # Use the specified number of crops
    else: raise ValueError(f"Invalid crops2use setting: {crops2use_stg}. Must be between 1 and {frgb_ncrops} (or -1 for maximum).")
    return crops2use

#####
def dyn_per_ds(ds,option_ucf,option_xdv):
    if ds == 'ucf': return option_ucf
    elif ds == 'xdv': return option_xdv
def dyn_oneorboth(idd,vrs):
    if not vrs: return f"uws4vad.model.net.{idd}.Network"
    else: return f"uws4vad.model.net.{idd}.{vrs}" 
def dyn_join(seperator, *args):
    # Join non-empty arguments with seperator
    return seperator.join(arg for arg in args if arg)
#####  


def dyn_retatt(watch_frm, intest):
    if not intest: return False
    elif 'attws' not in watch_frm: return False
    else: return True
    
def dyn_vadtaskname(ds,dtproc):
    tmp = f"{ds.frgb.id.lower()}"
    if dtproc.crops2use.train: 
        tmp += f"{str(dtproc.crops2use.train)}"
    if ds.get("faud"): 
        tmp += f"-{ds.faud.id.lower()}"
    return tmp

#def dyn_fetaskname(cfg_faud,cfg_frgb):
#    if cfg_frb is None:
#        
#    tmp = f"{ds.id}_{ds.frgb.id.lower()}"
#    if dtproc.train.crops2use: 
#        tmp += f"-{str(dtproc.train.crops2use)}"
#    if ds.get("aud"): 
#        tmp += f"_{ds.aud.id.lower()}"
#    return tmp

def dyn_dataroot(*paths):
    valid_paths = [p for p in paths if osp.isdir(p)]
    if len(valid_paths) > 1:
        raise ValueError(f"Multiple valid directories found: {valid_paths}")
    return valid_paths[0] if valid_paths else None


def reg_custom_resolvers(version_base: str, config_path: str, config_name: str) -> Callable:
    ## Initialize the Global Hydra if not already
    with initialize_config_dir(version_base=version_base, config_dir=config_path):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=[])
    GlobalHydra.instance().clear()

    new_res = {
        'dyn_nworkers': dyn_nworkers,
        'dyn_dvc': dyn_dvc,
        'dyn_vldt': dyn_vldt,
        'dyn_vldtmtrc': dyn_vldtmtrc,
        'dyn_crops2use': dyn_crops2use,
        'dyn_oneorboth': dyn_oneorboth,
        'dyn_join': dyn_join,
        'dyn_vldtfwd': dyn_vldtfwd,
        'dyn_retatt': dyn_retatt,
        'dyn_vadtaskname': dyn_vadtaskname,
        'dyn_fencrops': dyn_fencrops,
        'dyn_dataroot': dyn_dataroot,
        'dyn_per_ds':dyn_per_ds
    }
    for resolver, function in new_res.items():
        OmegaConf.register_new_resolver(resolver, function)
    
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)
        return wrapper

    return decorator
