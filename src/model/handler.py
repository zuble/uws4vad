import torch
import torch.nn as nn
import torch.nn.init as tc_init

import os, os.path as osp, time, copy, glob, re
from hydra.utils import instantiate as instantiate
from collections import OrderedDict
#from tqdm import tqdm

from src.utils import get_log, hh_mm_ss
log = get_log(__name__)


class ModelHandler:
    def __init__(self, cfg, istrain=False):
        self.cfg = cfg
        
        self.id = f"{cfg.net.id}" ## if using dyn_name -> cfg.name can be used
        if cfg.net.vrs: self.id = self.id+f"_{cfg.net.vrs}"
        ## used later on test load to pre check if atual arch match state
        ## can fall into cases where eg cls used is diff !!??
        ## construct a + reliable id
        
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
                "epo": None,
                "mtrc_info":None
            }        
        else: self.get_ckpt_paths()
    
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
                            
    def record(self, mtrc_info, curv_info, table_res, net, optima, trn_inf):
        
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
                "epo": trn_inf['epo'],
                "mtrc_info": mtrc_info,
                "curv_info": curv_info
            }
            self.high_table = table_res
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
    
    def save_state(self, net=None, optima=None, trn_inf=None):
        
        if self.high_state['net'] is not None:
            tmp_fn = f"{self.cfg.seed}--{self.high_state['epo']}_{self.high_state['step']}"
            save_path = osp.join(self.cfg.path.out_dir, tmp_fn)
            
            mstate = self.high_state
            torch.save(mstate, f"{save_path}.state.pt")
            torch.save(mstate['net'], f"{save_path}.pt")
            
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
    def check_nd_find(self, path):
        pattern = f"**/*{path}*.state.pt"
        
        matches = glob.glob(osp.join(self.cfg.path.log_dir, pattern), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No checkpoint with seed {seed}")
        log.error(matches)
        return max(matches, key=osp.getmtime)  # Get most recent
    
        #seed = osp.basename(path).split(".")[0]
        #match = re.match(r'(\d+)--(.*)\.(state|pt)$', )
        #seed, epo, load_mode = match.groups()
        #log.error(match.groups())
        #match = re.search(r'seed=(\d+)', osp.basename(path))
        
        #if not match: raise ValueError(f"No seed in filename: {path}")        
    
    def get_ckpt_paths(self):
        
        if self.cfg.train: ## comming from traing, find state in now run_dir
            self.ckpt_path = glob.glob(f"{self.cfg.path.out_dir}/*.state.pt")
            log.error(self.ckpt_path)
            if not self.ckpt_path: 
                raise Exception(f"no __.state.pt found in current dir {self.cfg.path.out_dir}") 
            
        elif not self.cfg.load.ckpt_path:
            raise Exception("provide load.ckpt_path")
        
        else: ## in test
            tmp = self.cfg.load.get("ckpt_path")
            if isinstance(tmp, str): 
                if osp.exists(tmp): 
                    self.ckpt_path = [tmp]
                    return
            tmp = list(tmp)     
            self.ckpt_path = [self.check_nd_find(p) for p in tmp]    
    ########
    
    
    ########
    ## TRAIN    
    def get_train_state(self, trn_inf):
        self.get_ckpt_paths()
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
    def get_test_state(self, ckpt_path):
        
        if ckpt_path.split(".")[-2] != "state":
            log.warning(f"loading a full net struct from {ckpt_path}.pt")
            raise NotImplementedError
        
        log.info(f"[test] loading from {osp.basename(ckpt_path)}")    
        mstate = torch.load(
            ckpt_path,
            map_location=torch.device(self.cfg.dvc),
        )
        log.info(f"state: id {mstate['id']} | epo {mstate['epo']} step {mstate['step']} :: ")
        assert mstate['id'] == self.id, f"{self.id=}"
        
        ## return state and load directly in test/tester
        ## https://discuss.pytorch.org/t/attributeerror-incompatiblekeys-object-has-no-attribute-eval-cnn/121060
        return mstate["net"]
        
