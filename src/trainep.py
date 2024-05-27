import torch
import numpy as np

import time , os, os.path as osp, numpy, gc, json

from utils.misc import hh_mm_ss

log = None
def init(l):
    global log
    log = l


@torch.enable_grad() #need?
def train_epo(cfg, loader, frmter, net, net_pst_fwd, lossfx, ldata, optima, tmeter, trn_inf):
    
    net.train()
    for bat, tdata in enumerate(loader):
        trn_inf['bat']=bat
        btic = time.time()
        
        ##################
        ## FEAT/LDATA FRMT
        feat = frmter.fx(tdata, ldata, trn_inf)
        log.debug(f"------------E[{trn_inf['epo']+1}]B[{bat+1}]------------")
        #log.debug(f"feat: {feat.shape}")               

        ndata = net(feat)
        log.debug(f"{feat.shape} -> ")
        for key in list(ndata.keys())[1:]: log.debug(f"    {key} {ndata[key].shape}") if type(ndata[key]) == torch.Tensor else None
        ## if ndata['id'] == '...':
        
        loss_indv = net_pst_fwd.train(ndata, ldata, lossfx) 
        loss_glob = torch.sum(torch.stack(list(loss_indv.values())))
        
        ###########
        ## zero all of the gradients for the variables it will update 
        ## i.e. net learnable weights
        optima.zero_grad()
        ## compute gradient of the loss with respect to net parameters
        loss_glob.backward() ## gradients are accumulated
        optima.step() ## update to its parameters
        ###########
        
        ## monitor
        tmeter.update({'bat': loss_glob.item()})
        for key, value in loss_indv.items():
            ## Update individual loss parts
            tmeter.update({key: value.item()})  
        tmeter.log_bat(trn_inf)


