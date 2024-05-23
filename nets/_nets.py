import torch
import torch.nn as nn
import torch.nn.init as tc_init
import math, os.path as osp , glob , warnings
import importlib, pkgutil, inspect

#from .layers import *
from utils import LoggerManager, parse_ptfn


## 1 network definition per file
## main module named as Network
## if not cfg_net.VERSION must have the exact name as chosen
## glossary of terms
##  sls -> either represents segment-level scores or snippet-level scores (train/infer)
##  vls -> video-label scores
## 

log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)
    
    ## Dynamically import and initialize modules in the 'layers' subdirectory
    layers_dir = importlib.util.find_spec('nets.layers').submodule_search_locations[0]
    layers_modules = pkgutil.iter_modules([layers_dir])
    for _, module_name, _ in layers_modules:
        layer_module = importlib.import_module(f".layers.{module_name}", package='nets')
        if hasattr(layer_module, 'init'):
            print("NETLAYERS",layer_module)
            layer_module.init(log)
            

def weight_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def get_net(cfg, cfg_frgb, cfg_faud, dvc, wght_init=None, mode='train'):

    ## .tmp.py purposes when the dataloader dont run, therefore dont populate those values
    #if not cfg.DATA.RGB.NFEATURES: 
    #    log.error("exp .tmp.yml ??")
    #    log.warning("using cfg.DATA.RGB/AUD.NFEATURES to be 1024/128")
    #    cfg.merge_from_list(['DATA.RGB.NFEATURES', 1024, 'DATA.AUD.NFEATURES', 128]) 


    ## network / classifier cfg 
    cfg_net = getattr(cfg.NET, cfg.NET.NAME.upper(), None)
    if cfg_net: 
        cfg_cls = getattr(cfg.NET.CLS, cfg_net.CLS_VRS.upper(), None)
        if cfg_cls is None: log.warning(f"no classifier cfg in use")
    else: 
        cfg_cls = None
        log.error("exp .tmp.yml ?? no network cfg in use")
        
    
    '''
        cfg.merge_from_list(["NET.NAME", f"{cfg.NET.NAME}_{cfg_net.VERSION}"])
    '''
    
    ## every network main module must be named as Network
    net_mdl = importlib.import_module(f".{cfg.NET.NAME.lower()}", package="nets")
    #net_classes = inspect.getmembers(net_mdl, inspect.isclass)
    
    if getattr( cfg_net, 'VERSION', None): 
        class_name = cfg_net.VERSION
    else: class_name = "Network"

    net_class = getattr(net_mdl, class_name, None)
    if net_class == None: 
        raise Exception(f"Class {class_name} not found in module {cfg.NET.NAME.lower()}.py") 
    
    ## all posible values that network may contain
    kwargs_mapping = {
        'rgbnf': cfg_frgb.NFEATS,
        'audnf': getattr( cfg_faud, 'NFEATS', 0), ## !!
        'dvc': dvc,
        'cfg_net': cfg_net,
        'cfg_cls': cfg_cls
    }
    
    net_kwargs = {}
    # Get the constructor parameters of the network class
    constructor_params = inspect.signature(net_class.__init__).parameters
    for param_name, param in constructor_params.items():
        if param_name == 'self':
            continue

        if param_name in kwargs_mapping:
            # Get the corresponding variable from the configuration
            var = kwargs_mapping[param_name]
            log.info(f"{var} {param_name}")
            net_kwargs[param_name] = var
        else:
            raise Exception(f"Unexpected keyword argument '{param_name}' in {net_name}")

    net = net_class(**net_kwargs)
    
    log.info(f"NETWORK {cfg.NET.NAME} created")
    log.info('\ncfg_net\n{}\ncfg_cls\n{}'.format(cfg_net.dump(),cfg_cls.dump()) )
    
    
    ## weight init
    if wght_init is not None:
        if wght_init.lower() == 'xavier':
            net.apply(weight_init)
            log.info(f"{cfg.NET.NAME} Conv/Linear -> weights~xavier , bias~0")
    
    log.info(f"Net structure: {net}\n\n") if 'blck' in cfg.NET.LOG_INFO else None 
    if 'prmt' in cfg.NET.LOG_INFO:
        for name, param in net.named_parameters():
            log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    
    ## NetPstFwd
    net_mdl.init(log)
    net_pst_fwd = net_mdl.NetPstFwd(
        bs=cfg.TRAIN.BS, 
        ncrops=getattr( getattr(cfg, mode.upper()), 'CROPS2USE'),
        dvc=dvc
        )
        
    return net, net_pst_fwd


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


def convin2onnx(cfg):
    return 


def count_parms(net):
    t = sum(p.numel() for p in net.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')
