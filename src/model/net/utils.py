import torch
import torch.nn as nn
import torch.nn.init as tc_init

from hydra.utils import instantiate as instantiate
from omegaconf import OmegaConf, DictConfig

from src.utils import get_log
log = get_log(__name__)


#from src.model.net._utils import model_stats
#df = model_stats(model, input_shape)
#print(df)
#if args.out_file:
#    df.to_html(args.out_file + '.html')
#    df.to_csv(args.out_file + '.csv')

def count_parms(net):
    t = sum(p.numel() for p in net.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')
    
    ## https://github.com/Swall0w/torchstat
    ## https://github.com/sovrasov/flops-counter.pytorch
    ## https://github.com/cool-xuan/x-hrnet/blob/main/tools/torchstat_utils.py
    #from torchstat import stat
    #stat(net, (3, 224, 224))
    
def wght_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        log.debug(f"WI: {m}")
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def dry_run(net, cfg, dfeat):
    log.debug("DBG DRY FWD")
    nc = 1 if cfg.dataproc.crops2use.train == 0 else cfg.dataproc.crops2use.train 
    bs = cfg.dataload.bs
    t = cfg.dataproc.seg.len
    feat = torch.randn( (nc*bs, t, sum(dfeat) ))
    _ = net(feat)


def build_net(cfg):
    ## ARCH
    ## in Network.innit sum or ..
    dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
    log.info(f"{dfeat=}")
    
    ## !!!!!!!!!!!!!!
    ## how to have a feature modulator variable network
    ## without relying on instantiate in individuals Network classes 
    ## while having a robust builder in order to lock a Network design
    ## still delving into this and its blocking me to experiment

    log.info(f"{cfg.net.main=}")
    #cfg_net_main = cfg.net.main
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
    else:
        rgs = None
    
    net = instantiate(cfg_net_main, dfeat=dfeat, rgs=rgs)
    
    if cfg.net.wght_init == 'xavier0':
        net.apply(wght_init)
    else: raise NotImplementedError
    
    ## EXTRA
    if cfg.model.dryfwd:  dry_run(net, cfg, dfeat)
    if cfg.xtra.get("net"):
        log.info(f"\n{net}\n")
        #for name, param in net.named_parameters():
        #    log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
    ## inferator
    cfg_infer = DictConfig({
                '_target_': cfg.net.infer._target_,
                '_cfg': {
                    **{k: v for k, v in cfg.net.infer.items() if k != '_target_'}
                }
            })
    pstfwd_utils = instantiate(cfg.model.pstfwd)
    inferator = instantiate(cfg_infer, pfu=pstfwd_utils) 
    log.info(f"INFER {cfg_infer=} {inferator=}")
    
    return net, inferator