import torch
import torch.nn as nn
import torch.nn.init as tc_init

import os, os.path as osp, time, copy
from hydra.utils import instantiate as instantiate
from collections import OrderedDict
#from tqdm import tqdm

from src.utils import get_log, hh_mm_ss
log = get_log(__name__)


def count_parms(net):
    t = sum(p.numel() for p in net.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')
    

def wght_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)

def build_net(cfg):
    ## ARCH
    #log.info(cfg.model.net.arch)
    ## in Network.innit sum or ..
    dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
    log.debug(f"{dfeat}")
    
    ## if theres cls, instaneate it inside Network.innit
    ## otherwise import from layers.classifier or construct
    if cfg.model.net.get("cls"): 
        net = instantiate(cfg.model.net.main, dfeat=dfeat, _cls=cfg.model.net.cls, _recursive_=False).to(cfg.dvc)
    else:
        net = instantiate(cfg.model.net.main, dfeat=dfeat).to(cfg.dvc)
    
    if cfg.model.dryfwd:
        log.debug("DBG DRY FWD")
        nc = 1 if cfg.datatrnsfrm.train.crops2use == 0 else cfg.datatrnsfrm.train.crops2use 
        bs = cfg.dataloader.train.bs
        t = cfg.datatrnsfrm.train.len
        feat = torch.randn( (nc*bs, t, sum(dfeat) ))
        _ = net(feat)
    
    ## PSTFWD   
    netpstfwd = instantiate(cfg.model.net.pstfwd)
    
    ## INIT
    if cfg.model.net.wght_init == 'xavier0':
        net.apply(wght_init)
    else: raise NotImplementedError
    
    ## LOG STRUCT
    if cfg.xtra.get("net"):
        log.info(f"\n{net}\n")
        #for name, param in net.named_parameters():
        #    log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        log.info(netpstfwd)
        
    return net, netpstfwd


class ModelHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.id = f"{cfg.model.net.id}" ## if using dyn_name -> cfg.name can be used
        if cfg.model.net.vrs: self.id = self.id+f"_{cfg.model.net.vrs}"
        ## used later on test load to pre check if atual arch match state
        ## can fall into cases where eg cls used is diff !!??
        ## construct a + reliable id
        
        self.ckpt_path = cfg.load.ckpt_path
        ## only for train
        self.high_info = {
            'lbl2wtc': cfg.vldt.train.record_lbl,
            'mtrc2wtch': cfg.vldt.train.record_mtrc,
            'rec_val': 0.6500
        }
        self.high_state = {
            "net": None,
            "optima": None,
            "step": None,
            "epo": None
        }
        
    def optima_to(self, optima, dvc = 'cpu'):
        for param in optima.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(dvc)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(dvc)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(dvc)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(dvc)
                            
    def record(self, mtrc_info, net, optima, trn_inf):
        
        ## make modular by acpeting both subset and full metrcs
        ## and ssave high sate by pais
        ## sure there are some fx for this...
        tmp_res = mtrc_info[ self.high_info['lbl2wtc'] ][ self.high_info['mtrc2wtch'] ][0]
        if  tmp_res > self.high_info['rec_val']:
            ttic = time.time()
            ## deepcopy to avoid mess with runtime
            nett = copy.deepcopy(net).cpu()
            optimaa = copy.deepcopy(optima)
            self.optima_to(optimaa) 
        
            self.high_state = {
                "id": self.id,
                "net": nett.state_dict(),
                "optima": optimaa.state_dict(),
                "step": trn_inf['step'],
                "epo": trn_inf['epo']
            }
            log.info(f"saved new high {self.high_info['rec_val']} -> {tmp_res}  @{hh_mm_ss( time.time() - ttic)}")
            self.high_info['rec_val'] = tmp_res
        ## this can be used by ssetting an additional low_info
        ## that saves independent of hitting the rec_val when is eg 0.7500
        #else: ## still keep last vldt metrics results
        #    self.last_state = {
        #        "id": self.id,
        #        "net": None,
        #        "optima": None,
        #        "step": trn_inf['step'],
        #        "epo": trn_inf['epo'],
        #        "tmp_res": tmp_res
        #    }
    def save_state(self, net, optima, trn_inf):
        
        if self.high_state['net'] is not None:
            tmp_fn = f"{self.cfg.seed}--{self.high_state['epo']}_{self.high_state['step']}"
            save_path = osp.join(self.cfg.path.out_dir, tmp_fn)
            
            mstate = self.high_state
            log.info(f"saving from high state {self.high_info['rec_val']} {self.high_info['mtrc2wtch']} as : {tmp_fn}  ")
            
        else: raise NotImplementedError
        #    tmp_fn = f"{self.cfg.seed}--{trn_inf['epo']}_{trn_inf['step']}"
        #    save_path = osp.join(self.cfg.path.out_dir, tmp_fn)
        #    
        #    log.info(f"saving from last state {self.last_state['rec_val']} {self.high_info['mtrc2wtch']} as : {tmp_fn}  ")
        #    
        #    ## fine since endotrain
        #    net.to("cpu")
        #    self.optima_to(optima)
        #    
        #    mstate = {
        #        "id": self.id,
        #        "net": net.state_dict(), ## net state (needs same net def) 
        #        "optima": optima.state_dict(),
        #        "step": trn_inf['step'],
        #        "epo": trn_inf['epo'],
        #    }
            
        torch.save(mstate, f"{save_path}.state.pt")
        torch.save(net, f"{save_path}.pt")
        log.info(f"mstate save @ {save_path}.state.pt/.pt")
    
    def load_net_state(self, net_arch, net_state):
        log.warning(net_state)
        net_state_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in net_state.items():
            if k.startswith("module."):
                net_state_clean[k[7:]] = v
            else:
                net_state_clean[k] = v
        log.warning(net_state_clean)
        return net_arch.load_state_dict(net_state_clean, strict=self.cfg.load.strict_load)
    
    ########
    ## TRAIN    
    def load_train_state(self, trn_inf, net_arch, optima):
        if net_arch is None or optima is None: 
            raise NotImplementedError
        assert self.ckpt_path.slipt(".")[-2] == "state" ## _.state.pt
        
        mstate = torch.load(
            self.ckpt_path,
            map_location=torch.device(self.cfg.dvc),
        )
        
        net_load = self.load_net_state(net_arch, mstate["net"])
        optima.load_state_dict(mstate["optima"])
        trn_inf["step"] = mstate["step"]
        trn_inf["epo"] = mstate["epo"]
        
        log.info(f"starting from train_stat {osp.basename(self.ckpt_path)}")
        return net_load, optima 
    
    #######
    ## TEST
    def load_net(self, net_arch=None):
        mstate = torch.load(
            self.ckpt_path,
            map_location=torch.device(self.cfg.dvc),
        )
        
        if self.ckpt_path.split(".")[-2] != "state":
            log.warning(f"loading a full net struct from {ckpt_path}.pt")
            net = mstate
        
        elif net_arch is not None: 
            ## net_arch need to match struct of ones in mstate
            ## assert mstate['id'] == self.id
            log.info(f"loading state id @ epo {mstate['epo']}  step {mstate['step']} :: {osp.basename(self.ckpt_path)}")
            net = self.load_net_state(net_arch, mstate["net"])

        else: raise NotImplementedError
        
        return net
    
