import torch
from multiprocessing import cpu_count

from typing import Any, Dict, Callable
from functools import wraps
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir, compose



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

def dyn_workers(nwkrs):
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

def dyn_crops2use(rgbncrops, mode, crops2use):
    if rgbncrops == 0: return 0
    
    if mode == 'test': return 1 ## has crops but use center
        
    else:
        if crops2use == -1: return rgbncrops
        elif 0 < crops2use <= rgbncrops: return crops2use
        else: return 1

def dyn_mainet(idd,vrs):
    if not vrs: return f"src.model.net.{idd}.Network"
    else: return f"src.model.net.{idd}.{vrs}"    

def dyn_vldtfwd(netid, chuksiz):
    if netid == 'attnmil': return chuksiz
    else: return None

def dyn_name(x,y): 
    if y: return f"{x}_{y}"
    else: return x
def dyn_taskname(ds,rgb,aud,crops2use):
    tmp = f"{ds}_{rgb.lower()}"
    if crops2use: tmp =+ f"-{crops2use}"
    if aud: 
        tmp =+ f"_{aud.lower()}"
    return tmp

def dyn_retatt(watch_frm, intest):
    if not intest: return False
    elif 'attws' not in watch_frm: return False
    else: return True

def reg_custom_resolvers(version_base: str, config_path: str, config_name: str) -> Callable:
    ## Initialize the Global Hydra if not already
    with initialize_config_dir(version_base=version_base, config_dir=config_path):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=[])
    GlobalHydra.instance().clear()

    new_res = {
        'dyn_nworkers': dyn_workers,
        'dyn_dvc': dyn_dvc,
        'dyn_vldt': dyn_vldt,
        'dyn_vldtmtrc': dyn_vldtmtrc,
        'dyn_crops2use': dyn_crops2use,
        'dyn_mainet': dyn_mainet,
        'dyn_name': dyn_name,
        'dyn_vldtfwd': dyn_vldtfwd,
        'dyn_retatt': dyn_retatt,
        'dyn_taskname': dyn_taskname
    }
    for resolver, function in new_res.items():
        OmegaConf.register_new_resolver(resolver, function)
    
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)
        return wrapper

    return decorator
