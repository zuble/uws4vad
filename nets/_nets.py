import torch
import torch.nn as nn
import torch.nn.init as tc_init
import math, os.path as osp , glob , warnings
import importlib, pkgutil

#from .layers import *
from utils import LoggerManager


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


def get_net(cfg, dvc, wght_init=None):

    ## .tmp.py purposes when the dataloader dont run, therefore dont populate those values
    if not cfg.DATA.RGB.NFEATURES: 
        log.error("exp .tmp.yml ??")
        log.warning("using cfg.DATA.RGB/AUD.NFEATURES to be 1024/128")
        cfg.DATA.RGB.NFEATURES = 1024
        cfg.DATA.AUD.NFEATURES = 128
    
    cfg_net = getattr(cfg.NET, cfg.NET.NAME.upper())
    cfg_cls = getattr(cfg.NET.CLS, cfg_net.CLS_VRS.upper(), None)
    
    
    if 'attnomil' == cfg.NET.NAME.lower():
        net_mdl = importlib.import_module(".attnomil", package="nets")    
        
        if cfg_net.VERSION == "VCls":
            net = net_mdl.VCls(cfg.DATA.RGB.NFEATURES, cfg_net, cfg_cls)
            #net = net_mdl.VCls()
            
        elif cfg_net.VERSION == "SAVCls":
            net = net_mdl.SAVCls(cfg.DATA.RGB.NFEATURES, cfg_net)
            
        elif cfg_net.VERSION == "SAVCls_lstm":
            if not cfg_net.LSTM_DIM: raise Exception(f"set cfg.NET.ATTNOMIL.LSTM_DIM")
            net = net_mdl.SAVCls_lstm()
            
        elif cfg_net.VERSION == "LSTMCls":
            if not cfg_net.LSTM_DIM: log.error(f"set cfg.NET.ATTNOMIL.LSTM_DIM")
            net = net_mdl.LSTMCls(lstm_dim = cfg_net.LSTM_DIM)
            
        else: raise Exception(f'no version {cfg_net.VERSION} in ATNOMIL')
        
        cfg.merge_from_list(["NET.NAME", f"{cfg.NET.NAME}_{cfg_net.VERSION}"])

    
    elif 'cmala' == cfg.NET.NAME.lower(): 
        #cmala.init(log)
        net_mdl = importlib.import_module(".cmala", package="nets")    
        net = net_mdl.CMA(
            rgbnf=cfg.DATA.RGB.NFEATURES, 
            audnf=cfg.DATA.AUD.NFEATURES, 
            dvc=dvc,
            cfg_net=cfg_net,
            cfg_cls=cfg_cls
            )
    
        
    elif 'dtr' == cfg.NET.NAME.lower():
        #dtr.init(log)
        net_mdl = importlib.import_module(".dtr", package="nets")            
        net = net_mdl.dtr(  
            rgbnf=cfg.DATA.RGB.NFEATURES, 
            audnf=cfg.DATA.AUD.NFEATURES, 
            cfg_net=cfg_net,
            cfg_cls=cfg_cls
            )
    
        
    elif 'mindspore' == cfg.NET.NAME.lower():
        #mindspore.init(log)
        net_mdl = importlib.import_module(".mindspore", package="nets")    
        net = net_mdl.MindSpore(  
            rgbnf=cfg.DATA.RGB.NFEATURES, 
            dvc=dvc,
            cfg_net=cfg_net,
            cfg_cls=cfg_cls
            )
    
        
    elif 'rtfm' == cfg.NET.NAME.lower():
        net_mdl = importlib.import_module(".rtfm", package="nets")            
        net = net_mdl.RTFM(  
            rgbnf=cfg.DATA.RGB.NFEATURES, 
            cfg_net=cfg_net,
            cfg_cls=cfg_cls
            )
    
        
    else: raise Exception(f'no net name {cfg.NET.NAME}')
    
    
    log.info(f"NETWORK {cfg.NET.NAME} created")
    log.info('\ncfg_net\n{}\ncfg_cls\n{}'.format(cfg_net.dump(),cfg_cls.dump()) )
    if wght_init.lower() == 'xavier':
        net.apply(weight_init)
        log.info(f"{cfg.NET.NAME} Conv/Linear -> weights~xavier , bias~0")
    
    log.info(f"Net structure: {net}\n\n") if 'blck' in cfg.NET.LOG_INFO else None 
    if 'prmt' in cfg.NET.LOG_INFO:
        for name, param in net.named_parameters():
            log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    ## NetPstFwd
    ## as cfg.DATA.RGB.NCROPS guides the traindl init
    ## assure that if no crops in use, NetPstFwd.ncrops is 1
    ## changing is dangerous
    if cfg.DATA.RGB.NCROPS == 0: nc = 1; log.warning(f"NetPstFwd.ncrops set as 1 ({cfg.DATA.RGB.NCROPS=})")
    else: nc = cfg.DATA.RGB.NCROPS
    
    net_mdl.init(log)
    net_pst_fwd = net_mdl.NetPstFwd(
        bs=cfg.TRAIN.BS, 
        ncrops=nc,
        dvc=dvc
        )
        
    return net, net_pst_fwd


def get_ldnet(cfg, dvc):
    
    if not cfg.TEST.LOADFROM: raise Exception (f'cfg.TEST.LOADFROM must be set')
    
    load_mode = parse_ptfn(cfg.TEST.LOADFROM)['load_mode']
    
    PATH = osp.join(cfg.EXPERIMENTPATH, f"{cfg.TEST.LOADFROM}.pt")
    log.info(f'loading {load_mode} from {PATH}')
    
    if load_mode == 'net': 
        net = torch.load(PATH)
    elif load_mode == 'dict':
        net = get_net(cfg, dvc)
        net.load_state_dict( torch.load(PATH) )
    else: raise Exception(f'{cfg.TEST.LOADFROM = } and should be (...).["net","dict"]')
    
    return net


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