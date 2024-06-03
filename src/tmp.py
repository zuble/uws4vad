import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import glob , os, os.path as osp
from hydra.utils import instantiate as instantiate

from src.data import get_trainloader, run_dl
from src.utils.logger import get_log
log = get_log(__name__)




class Debug():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug_data()
        
        
    def debug_data(self):
        traindl, trainfrmt = get_trainloader(self.cfg) 
        run_dl(traindl)
        
        
    
    def debug_loss(self):
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
        
        dfeat = self.cfg.data.ds.frgb.dfeat + (self.cfg.data.ds.faud.dfeat if self.cfg.data.ds.get("faud") else 0)
        
        if self.cfg.model.net.get("cls"): 
            net = instantiate(cfg.model.net.main, dfeat=dfeat, _cls=self.cfg.model.net.cls, _recursive_=False)#.to(cfg.dvc)
        else:
            net = instantiate(cfg.model.net.main, dfeat=dfeat)#.to(cfg.dvc)
            
    