import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import glob , os, os.path as osp
from hydra.utils import instantiate as instantiate

from src.utils.logger import get_log
log = get_log(__name__)




class Debug():
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        import re
        ckpt_path = "/mnt/t77/TH/zutc_vad_hydra/log/rtfm/000/runs/31-05_02-19-59/2000861580--1_1.state"
        
                
        log.error(seed)
        
        
    def debug_loss(self, cfg):
        ## set debug.model=1
        from src.model.loss.mgnt import Rtfm
        
        nc = 2
        bag = 4
        t = 16
        dfeat = 512
        
        loss_cfg = {
            '_target_': 'src.model.loss.mgnt.Rtfm',
            '_cfg': {
                'k': 3,
                'alpha': 0.0001,
                'margin': 100,
            }
        }
        
        L = instantiate( loss_cfg )
        _ = L(
            abnr_fmagn=torch.randn(bag, t),
            norm_fmagn=torch.randn(bag, t),
            abnr_feats=torch.randn(nc, bag, t, dfeat),
            norm_feats=torch.randn(nc, bag, t, dfeat),
            abnr_sls=torch.randn(bag, t),
            norm_sls=torch.randn(bag, t),
            ldata=
            {
                'label': torch.cat( (torch.zeros(bag), torch.ones(bag)) )
            }
        )

    def debug_net(self, cfg):
        from src.model.net.rtfm import Network
        net_cfg = {
            '_target_': 'src.model.net.rtfm.Network',
            '_cfg': {
                'do': 0.7,
            }
        }
        
        dfeat = cfg.data.ds.frgb.dfeat + (cfg.data.ds.faud.dfeat if cfg.data.ds.get("faud") else 0)
        
        if cfg.model.net.get("cls"): 
            net = instantiate(cfg.model.net.main, dfeat=dfeat, _cls=cfg.model.net.cls, _recursive_=False)#.to(cfg.dvc)
        else:
            net = instantiate(cfg.model.net.main, dfeat=dfeat)#.to(cfg.dvc)
            
    