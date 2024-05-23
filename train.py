import torch
import numpy as np

import time , os, os.path as osp, gc
from collections import  defaultdict, deque

import trainep
from loss import get_loss
from nets import get_net, save
from data import get_trainloader, TrainFrmter, run_dl
from utils import LoggerManager, Visualizer, hh_mm_ss, get_optima
from vldt import Validate, Metrics


log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)
    trainep.init(log)



def trainer(cfg, dvc, dl=None):

    ## Data
    ## innit method in TrainDataset needs to run fristly to set nfeatures in cfg.DATA.RGB , cmala/mindspore only
    if dl: 
        log.warning(f"REUSING {dl} FROM PREVIOUS RUN")
        loader = dl
    else: loader, cfg_frgb, cfg_faud = get_trainloader(cfg) #;run_dl(loader); return
    frmter = TrainFrmter(cfg)

    ## NetWork
    net, net_pst_fwd = get_net(cfg, cfg_frgb, cfg_faud, dvc, cfg.NET.WGHT_INIT)
    net.to(dvc)
    #f = torch.randn(2, 32, cfg.DATA.RGB.NFEATURES).to(dvc)
    #y = net(f)  
    #log.info(f"{y = }")
    
    ## Solver
    optima, lrs = get_optima(cfg, net.parameters())
    lossfx, ldata = get_loss(cfg, dvc)
    
    ## Monitor
    vis = Visualizer(f'{cfg.EXPERIMENTPROJ}_{cfg.EXPERIMENTID}', restart=False, del_all=False)
    #vis.delete(f'{cfg.EXPERIMENTPROJ}/{cfg.EXPERIMENTID}')
    metrics = Metrics(cfg, 'train', vis)
    cfg_vldt = cfg.TRAIN.VLDT
    
    vldt = Validate(cfg, getattr(cfg.DS, cfg.TRAIN.DS[0].split('.')[0]), cfg_frgb, cfg_vldt, net_pst_fwd, dvc, metrics, watching=None)
    vldt.start(net);return
    if not cfg.TRAIN.LOG_PERIOD: cfg.merge_from_list(["TRAIN.LOG_PERIOD", cfg.TRAIN.EPOCHBATCHS // 2])
    
    ## Train
    train_epo = trainep.train_epo
    tmeter = TrainMeter(cfg, vis)
    trn_inf = {
        'epo': 0,
        'ttic': time.time(),
        'dvc': dvc
    }
    log.info(f'$$$$ TRAIN starting w/ cfg\n{cfg.TRAIN}')
    for epo in range(cfg.TRAIN.EPOCHS):
        trn_inf['epo'] = epo
        trn_inf['etic'] = time.time()
        
        ##########
        train_epo(cfg, loader, frmter, net, net_pst_fwd, lossfx, ldata, optima, tmeter, trn_inf)
        ##########
        
        ## monitor 
        tmeter.log_epo(trn_inf)
        tmeter.reset()
            
        ## validat
        if (epo + 1) % cfg.TRAIN.VLDT.PERIOD == 0:
            log.info(f'$$$ Validation at epo {epo+1}')
            ## !!!! anything similiar in torch ?? 
            #if not cfg.TRAIN.VLDT.CUDDNAUTOTUNE: os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
            _, _, mtrc_info = vldt.start(net)
            if cfg_vldt.VISPLOT: vis.plot_vldt_mtrc(mtrc_info,epo)
            vldt.reset()
            #if not cfg.TRAIN.VLDT.CUDDNAUTOTUNE: os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'    
        
        lrs.step() ## adjust lr
        
    log.info(f"$$$$ TRAIN completed in {hh_mm_ss(time.time() - ttic)}")
    
    if not cfg.DEBUG.TRAIN or not cfg.DEBUG.ROOT: save(cfg, net)
    #return net


class ScalarMeter(object):
    def __init__(self,window_size):
        self.dequa=deque(maxlen=window_size)
        self.total=0.0
        self.count=0
    def reset(self):
        self.dequa.clear()
        self.total=0.0
        self.count=0
    def add_value(self,value):
        self.dequa.append(value)
        self.total+=value
        self.count+=1
    def get_win_median(self):
        return np.median(self.dequa)
    def get_win_avg(self):
        return np.mean(self.dequa)
    def get_global_avg(self):
        return self.total/self.count

class TrainMeter:
    def __init__(self, cfg, vis):
        self._cfg = cfg
        self.vis = vis
        self.plot = cfg.TRAIN.PLOT_LOSS
        self.epochbatchs = cfg.TRAIN.EPOCHBATCHS
        self.log_period = cfg.TRAIN.LOG_PERIOD
        self.loss_meters = defaultdict(lambda: ScalarMeter(cfg.TRAIN.LOG_PERIOD))
        self.spacer = 0
        
    def update(self, lbat_indv):
        for loss_name, l in lbat_indv.items():
            self.loss_meters[loss_name].add_value(l)
            if self.plot: self.vis.plot_lines(f"bat-{loss_name}", l)
            
    def reset(self):
        for meter in self.loss_meters.values():
            meter.reset()
    
    def log_bat(self, trn_inf):
        ## logs / plots to visdom every loss_meter loss_bat_..
        if (trn_inf['bat'] + 1) % self.log_period == 0:
            sms = f" E[{trn_inf['epo']+1}] B[{trn_inf['bat']+1}] S[{(trn_inf['epo'])*self.epochbatchs+(trn_inf['bat']+1)}]"
            self.spacer = len(sms)
            for loss_name, meter in self.loss_meters.items():
                sms += f" {loss_name} {meter.get_win_avg():.4f}" #Med {meter.get_win_median():.4f},
            log.info(f"{sms}")
            
    def log_epo(self, trn_inf):
        ## logs / plots to visdom loss_epo_..
        sms = f"*E[{trn_inf['epo']+1}]"
        if self.spacer: sms += " "*(self.spacer-len(sms))
        for loss_name, meter in self.loss_meters.items():
            sms += f" {loss_name} {meter.get_global_avg():.4f}"
            if self.plot: self.vis.plot_lines(f"epo-{loss_name}", meter.get_global_avg())
        sms += f" @ {hh_mm_ss(time.time()-trn_inf['etic'])}"
        log.info(f"{sms}")