import torch
import torch.nn as nn
import torch.nn.init as tc_init

from hydra.utils import instantiate as instantiate
from omegaconf import OmegaConf, DictConfig

from src.utils import prof, flops, info, get_log
log = get_log(__name__)


def wght_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        log.debug(f"WeigthInnit: {m}")
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def dry_run(net, cfg, dfeat, inferator=None,):
    log.debug("DBG DRY FWD")
    nc = 1 if cfg.dataproc.crops2use.train == 0 else cfg.dataproc.crops2use.train 
    bs = cfg.dataload.bs
    t = cfg.dataproc.seg.len
    feat = torch.randn( (nc*bs, t, sum(dfeat) ))
    ndata = net(feat)
    for k, v in ndata.items(): 
        try: log.info(f"{k}: {list(v.shape)}")
        except: log.info(f"{k}: {len(v)}")
    if inferator:
        sls = inferator(ndata)
        log.info(f"sls {list(sls.shape)}")
    
def build_net(cfg):
    ## ARCH
    ## in Network.innit sum or ..
    dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
    log.info(f"{dfeat=}")
    
    
    ## TODO
    ## a robust builder in order to lock a Network design 
    ## having multiple variable networks,
    ## without relying on instantiate in individuals Network classes 
    ## still delving into this, for now take care of cls here only
    ## if need do it under the Network class

    log.info(f"{cfg.net.main=}")
    #cfg_net_main = cfg.net.main
    ## this enables to have all parameters in cfg under _target_
    ## while having only one parameter in _init_ method of target class
    cfg_net_main = DictConfig({
                        '_target_': cfg.net.main._target_,
                        '_cfg': {
                            **{k: v for k, v in cfg.net.main.items() if k != '_target_'}
                        }
                    })
    #cfg_net_main = OmegaConf.resolve(cfg_net_main)
    #log.error(cfg_net_main)
    
    ## if theres cls, instaneate it inside Network.innit
    ## otherwise import from layers.classifier or construct
    if cfg.net.get("cls"):
        log.debug(f"{cfg.net.cls=}")
        rgs = instantiate(cfg.net.cls, dfeat=dfeat)
        log.debug(f"{rgs=}")
        net = instantiate(cfg_net_main, dfeat=dfeat, rgs=rgs)
    else:
        net = instantiate(cfg_net_main, dfeat=dfeat)
    
    
    if cfg.net.wght_init == 'xavier0':
        net.apply(wght_init)
    else: raise NotImplementedError
    
    ## inferator
    cfg_infer = DictConfig({
                '_target_': cfg.net.infer._target_,
                '_cfg': {
                    **{k: v for k, v in cfg.net.infer.items() if k != '_target_'}
                }
            })
    ## tst flag set ncrops rigth
    pstfwd_utils = instantiate(cfg.model.pstfwd, tst=True)
    inferator = instantiate(cfg_infer, pfu=pstfwd_utils) 
    
    
    ## EXTRA
    #if cfg.get("debug"):
    #    for name, param in net.named_parameters():
    #        log.info(f"Layer: {name} | Size: {param.size()} | DVC: {param.device} Values : {param[:2]} \n")
    if cfg.net.summary: 
        from src.utils import info, flops
        sz = (cfg.dataload.bs, cfg.dataproc.seg.len, sum(dfeat))
        flops(net, inpt_size=sz)
        #info(model, inpt_size=sz, inpt_data=frame )
        log.info(f"\n\n{net=}\n")
        log.info(f"INFER\n\t{cfg_infer=}\n\t{inferator=}")
    
    if cfg.net.dryfwd: 
        dry_run(net, cfg, dfeat, inferator)
    
    if cfg.net.profile:
        nb = 2
        log.warning(f"PROFILING: {nb}x{cfg.dataload.bs}x{cfg.dataproc.seg.len}segments")
        prof(net, inpt_size=(nb,  cfg.dataload.bs, cfg.dataproc.seg.len, sum(dfeat)))
    
    return net, inferator