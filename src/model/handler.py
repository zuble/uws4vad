import torch
import torch.nn as nn
import torch.nn.init as tc_init

import os, os.path as osp, time, copy, glob
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
    

'''
## https://github.com/facebookresearch/fvcore
## https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
def flops(self, shape=(3, 224, 224), verbose=True):
    # shape = self.__input_shape__[1:]
    supported_ops={
        "aten::silu": None, # as relu is in _IGNORED_OPS
        "aten::neg": None, # as relu is in _IGNORED_OPS
        "aten::exp": None, # as relu is in _IGNORED_OPS
        "aten::flip": None, # as permute is in _IGNORED_OPS
        # "prim::PythonOp.CrossScan": None,
        # "prim::PythonOp.CrossMerge": None,
        "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
    }

    model = copy.deepcopy(self)
    model.cuda().eval()

    input = torch.randn((1, *shape), device=next(model.parameters()).device)
    params = parameter_count(model)[""]
    Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    del model, input
    return sum(Gflops.values()) * 1e9
    return f"params {params} GFLOPs {sum(Gflops.values())}"
'''


class ModelHandler:
    def __init__(self, cfg, istrain=False):
        self.cfg = cfg
        
        self.id = f"{cfg.net.id}" ## if using dyn_name -> cfg.name can be used
        if cfg.net.vrs: self.id = self.id+f"_{cfg.net.vrs}"
        ## used later on test load to pre check if atual arch match state
        ## can fall into cases where eg cls used is diff !!??
        ## construct a + reliable id
        
        self.ckpt_path = cfg.load.ckpt_path
        
        if istrain:
            self.high_info = {
                'lbl2wtc': cfg.vldt.train.record_lbl,
                'mtrc2wtch': cfg.vldt.train.record_mtrc,
                'rec_val': 0.5000
            }
            self.high_state = {
                "net": None,
                "optima": None,
                "step": None,
                "epo": None
            }
            log.info(f"[train] load path set as {self.ckpt_path}") 
    
    
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
        ## and save high sate by pais
        ## sure there are some fx for this...
        tmp_res = mtrc_info[ self.high_info['lbl2wtc'] ][ self.high_info['mtrc2wtch'] ][0]
        if  tmp_res > self.high_info['rec_val']:
            
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
            log.info(f"saved new high {self.high_info['rec_val']} -> {tmp_res}")
            self.high_info['rec_val'] = tmp_res
        ## this can be used by setting an additional low_info
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
            torch.save(mstate, f"{save_path}.state.pt")
            torch.save(net, f"{save_path}.pt")
            
            log.info(f"savedfrom high state {self.high_info['rec_val']} {self.high_info['mtrc2wtch']} ")
            log.info(f"fn: {tmp_fn}")
            log.info(f"path: {save_path}.state.pt/.pt")
            
        else: 
            log.warning(f"run didnt reach initial {self.high_info['rec_val']} {self.high_info['mtrc2wtch']}")
            raise NotImplementedError
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
            
    ###########
    def load_net_state(self, net_arch, net_state):
        net_state_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in net_state.items():
            if k.startswith("module."):
                log.warning(k)
                net_state_clean[k[7:]] = v
            else:
                log.warning(k)
                net_state_clean[k] = v

        #log.warning(net_state_clean)
        return net_arch.load_state_dict(net_state_clean, strict=self.cfg.load.strict_load)
    ##########
    
    
    ########
    ## TRAIN    
    def get_train_state(self, trn_inf):
        #if net_arch is None or optima is None: 
        #    raise NotImplementedError
        assert self.ckpt_path.slipt(".")[-2] == "state" ## _.state.pt
        
        mstate = torch.load(
            self.ckpt_path,
            map_location=torch.device(self.cfg.dvc),
        )
        
        #net_load = self.load_net_state(net_arch, mstate["net"])
        #optima.load_state_dict(mstate["optima"])
        trn_inf["step"] = mstate["step"]
        trn_inf["epo"] = mstate["epo"]
        
        log.info(f"starting from train_stat {osp.basename(self.ckpt_path)}")
        return mstate["net"], mstate["optima"] #net_load, optima 
    
    
    #######
    ## TEST
    def get_test_state(self):
        
        if self.cfg.train: ## comming from traing, find state in now run_dir
            self.ckpt_path = glob.glob(f"{self.cfg.path.out_dir}/*.state.pt")[0]
            log.error(self.ckpt_path)
            if not self.ckpt_path: 
                raise Exception(f"no __.state.pt found in current dir {self.cfg.path.out_dir}")  
        
        elif not self.ckpt_path:
            raise Exception("provide load.ckpt_path")  
        
        elif self.ckpt_path.split(".")[-2] != "state":
            log.warning(f"loading a full net struct from {ckpt_path}.pt")
            raise NotImplementedError
        
        log.info(f"[test] loading from {osp.basename(self.ckpt_path)}")    
        mstate = torch.load(
            self.ckpt_path,
            map_location=torch.device(self.cfg.dvc),
        )
        log.info(f"state: id {mstate['id']} | epo {mstate['epo']} step {mstate['step']} :: ")
        assert mstate['id'] == self.id
        
        ## return state and load directly in test/tester
        ## https://discuss.pytorch.org/t/attributeerror-incompatiblekeys-object-has-no-attribute-eval-cnn/121060
        return mstate["net"]
        
