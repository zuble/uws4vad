import torch
import torch.nn as nn
import torch.nn.init as tc_init

from hydra.utils import instantiate as instantiate

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
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def dry_run(net, cfg, dfeat):
    log.debug("DBG DRY FWD")
    nc = 1 if cfg.dataproc.crops2use.train == 0 else cfg.dataproc.crops2use.train 
    bs = cfg.dataload.train.bs
    t = cfg.dataproc.seg.len
    feat = torch.randn( (nc*bs, t, sum(dfeat) ))
    _ = net(feat)


def build_net(cfg):
    ## ARCH
    #log.info(cfg.model.net.arch)
    ## in Network.innit sum or ..
    dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
    log.debug(f"{dfeat}")
    
    ## if theres cls, instaneate it inside Network.innit
    ## otherwise import from layers.classifier or construct
    if cfg.model.net.get("cls"): 
        net = instantiate(cfg.model.net.main, dfeat=dfeat, _cls=cfg.model.net.cls, _recursive_=False)
    else:
        net = instantiate(cfg.model.net.main, dfeat=dfeat)
    
    ## PSTFWD   
    netpstfwd = instantiate(cfg.model.net.pstfwd)
    
    ## INIT
    if cfg.model.net.wght_init == 'xavier0':
        net.apply(wght_init)
    else: raise NotImplementedError
    
    ## EXTRA
    if cfg.model.dryfwd:  dry_run(net, cfg, dfeat)
    if cfg.xtra.get("net"):
        log.info(f"\n{net}\n")
        #for name, param in net.named_parameters():
        #    log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        log.info(netpstfwd)
    
    return net, netpstfwd