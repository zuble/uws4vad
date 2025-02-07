import torch
import torch.nn as nn
import torch.nn.init as tc_init

from hydra.utils import instantiate as instantiate
from omegaconf import OmegaConf, DictConfig

from src.utils import get_log
log = get_log(__name__)


def prof(model, inpt=None, inpt_size=None):
    
    from torch.profiler import profile, record_function, ProfilerActivity
    
    if inpt is None:
        assert inpt_size is not None
        inpt = torch.randn( inpt_size )       
    
    # Warmup runs (not profiled)
    #with torch.no_grad():
    #    for i in range(int(len(inpt)/2)):
    #        _ = model(inpt[i])
    #        #torch.cuda.synchronize()  # For accurate CUDA timing
    
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", #excludes time spent in children operator calls
            #sort_by="cpu_memory_usage",
            row_limit=-1))

    with profile(
        activities=[
            ProfilerActivity.CPU, 
            #ProfilerActivity.CUDA
            ],
        profile_memory=True, 
        #group_by_input_shape=True  #finer granularity of results and include operator input shapes
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=2,
        #     repeat=2),
        # on_trace_ready=trace_handler
        ) as prof:
            with record_function("infer"):
                for x in inpt:
                    _ = model(x)
            # for i in range(0, 2):
            #     _ = model(x)
            #     torch.cuda.synchronize()  # Sync CUDA ops
            #     prof.step()
    
    #print("CPU/GPU Time Analysis:")
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
    print("\nMemory Analysis:")
    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

def count_params(net):
    t = sum(p.numel() for p in model.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')   

    
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
    for k, v in ndata.items(): print(f"{k}: {list(v.shape)}")
    if inferator:
        sls = inferator(ndata)
        print(f"sls {list(sls.shape)}")
    
def build_net(cfg):
    ## ARCH
    ## in Network.innit sum or ..
    dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
    log.info(f"{dfeat=}")
    
    ## !!!!!!!!!!!!!!
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
        rgs = instantiate(cfg.net.cls, dfeat=dfeat).to(cfg.dvc)
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
    if cfg.net.info:
        from torchinfo import summary
        summary(net, 
            input_size=(cfg.dataload.bs, cfg.dataproc.seg.len, sum(dfeat)),
            col_names=["input_size","output_size", "num_params", "trainable", "mult_adds"],
            #verbose=2
        ) 
        log.info(f"\n\n{net=}\n")
        log.info(f"INFER\n\t{cfg_infer=}\n\t{inferator=}")
    
    if cfg.net.dryfwd: 
        dry_run(net, cfg, dfeat, inferator)
    
    if cfg.net.profile:
        prof(net, inpt_size=(32, cfg.dataproc.seg.len, sum(dfeat)) )
    
    return net, inferator