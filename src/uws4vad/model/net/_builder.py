import torch
import torch.nn as nn
import torch.nn.init as tc_init

from hydra.utils import instantiate as instantiate
from omegaconf import OmegaConf, DictConfig

from uws4vad.common.registry import registry
from uws4vad.utils import prof, flops, info, get_log
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

def dry_run(net, cfg, inferator=None,):
    log.debug("DBG DRY FWD")
    nc = 1 if cfg.dataproc.crops2use.train == 0 else cfg.dataproc.crops2use.train 
    bs = cfg.dataload.bs
    t = cfg.dataproc.seg.len
    feat = torch.randn( (nc*bs, t, net.dfeat ))
    ndata = net(feat)
    sms="DRY RUN:\n\t"
    for k, v in ndata.items(): 
        try: sms+=f"{k}: {list(v.shape)}\n\t"
        except: sms+=f"DR {k}: {len(v)}\n\t"
    if inferator:
        sls = inferator(ndata)
        sms+=f"sls {list(sls.shape)}"
    log.info(sms)
    
def build_net(cfg):
    log.debug(f"{cfg.net=}  \n\n {registry._registry['network']=}")
    
    ## ---------
    ## Network
    net_class = registry.get("network", cfg.net.id)
    if not net_class:
        raise ValueError(f"Network type {cfg.net.id} not registered.")
    
    net = net_class(cfg=cfg) # Instantiate netowrk with config
    
    if cfg.net.wght_init == 'xavier0':
        net.apply(wght_init)
    else: raise NotImplementedError
    
    
    ## ---------
    ## Inferator
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
        from uws4vad.utils import info, flops
        sz = (cfg.dataload.bs, cfg.dataproc.seg.len, net.dfeat)
        flops(net, inpt_size=sz)
        #info(model, inpt_size=sz, inpt_data=frame )
        log.info(f"\n\n{net=}\n")
        log.info(f"INFER\n\t{cfg_infer=}\n\t{inferator=}")
    
    if cfg.net.dryfwd: 
        dry_run(net, cfg, inferator)
    
    if cfg.net.profile:
        nb = 2
        log.warning(f"PROFILING: {nb}x{cfg.dataload.bs}x{cfg.dataproc.seg.len}segments")
        prof(net, inpt_size=(nb,  cfg.dataload.bs, cfg.dataproc.seg.len, net.dfeat))
    
    return net, inferator