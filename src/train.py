import torch
import numpy as np

import time , os, os.path as osp, gc
from collections import  defaultdict, deque

from src.data import get_trainloader, get_testloader, run_dl
from src.model import ModelHandler

#from loss import get_loss
#from nets import get_net, save
#from data import get_trainloader, TrainFrmter, run_dl
#from utils import Visualizer, hh_mm_ss, get_optima, get_log
#from vldt import Validate, Metrics


def trainer(cfg):
    traindl, trainfrmt = data.get_trainloader(cfg) #;run_dl(loader); return
    
    ## ARCH
    #log.info(cfg.model.net.arch)
    self.net = instantiate(cfg.model.net.arch, feat_len=1024).to(self.dvc)
    log.debug(net)
    
    ## PSTFWD
    #log.info(cfg.model.net.pstfwd)
    netpstfwd = instantiate(cfg.model.net.pstfwd)
    log.debug(netpstfwd)
    
    ## OPTIMA
    #log.info(cfg.model.optima)
    optima = instantiate(cfg.model.optima, params=net.parameters(), _convert_="partial" )
    log.info(optima)

    ## LRS
    if cfg.model.get("lrs"):
        log.info(cfg.model.lrs)
        lrs = instantiate( cfg.model.lrs, optimizer=optima, _convert_="partial")
        log.info(lrs)
    
    ## LOSSFXS
    lossfxs = {lid: instantiate(cfg.model.loss[lid]) for lid in cfg.model.loss}
    for key, value in lossfxs.items():
        log.info({key: value})
    #aux_params = set()    
    #for lid, lcfg in cfg.model.loss.items():
    #    if lcfg.get("aux"):
    #        aux_params.update(lcfg.aux)
    #ldata = {p: None for p in aux_params}
    #for key, value in ldata.items():
    #    ## Update individual loss parts
    #    log.info({key: value}) 
    
    
    ## Monitor
    #vis = Visualizer(f'{cfg.EXPERIMENTPROJ}_{cfg.EXPERIMENTID}', restart=False, del_all=False)
    #vis.delete(f'{cfg.EXPERIMENTPROJ}/{cfg.EXPERIMENTID}')
    

    vldt = Validate(cfg, cfg.vldt.train) #vldt.start(net);return
    #if not cfg.model.log_loss: cfg.merge_from_list(["TRAIN.LOG_PERIOD", cfg.TRAIN.EPOCHBATCHS // 2])
    
    ## Train
    tmeter = TrainMeter(cfg)
    trn_inf = {
        'epo': 0,
        'ttic': time.time(),
    }
    try:
        for epo, tdata in enumerate(tqdm(cfg.model.epochs, leave = False, desc="Training/Batch:", unit='batch')):
        #for epo in range(cfg.model.epochs):
            trn_inf['epo'] = epo
            trn_inf['etic'] = time.time()
            
            ##########
            train_epo(cfg, loader, frmter, net, net_pst_fwd, lossfx, ldata, optima, tmeter, trn_inf)
            ##########
            
            ## monitor 
            tmeter.log_epo(trn_inf)
            tmeter.reset()
                
            ## validat
            if (epo + 1) % cfg_vldt.freq == 0:
                log.info(f'$$$ Validation at epo {epo+1}')
                ## !!!! anything similiar in torch ?? 
                #if not cfg.TRAIN.VLDT.CUDDNAUTOTUNE: os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
                _, _, mtrc_info = vldt.start(net)
                if cfg_vldt.visplot: vis.plot_vldt_mtrc(mtrc_info,epo)
                vldt.reset()
                #if not cfg.TRAIN.VLDT.CUDDNAUTOTUNE: os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'    
            
            lrs.step() ## adjust lr
            
        log.info(f"$$$$ TRAIN completed in {hh_mm_ss(time.time() - ttic)}")
        
        if not cfg.DEBUG.TRAIN or not cfg.DEBUG.ROOT: save(cfg, net)

    except Exception as e:
        log.error(traceback.format_exc())





    # load training state / network checkpoint
    if cfg.load.get("resume_state_path"): model.load_training_state()
    elif cfg.load.get("network_chkpt_path"): model.load_network()
    else:  log.info("Starting new training run.")

    try:
        epoch_step = 1
            
        for epoch in tqdm(range(model.epoch + 1, cfg.train.num_epoch, epoch_step), desc="Epoch", unit='epoch'):
            model.epoch = epoch
            model.train_epo(train_loader)
            
            if model.epoch % cfg.log.chkpt_interval == 0:
                model.save_network()
                model.save_training_state()
            model.test_model(test_loader)
        
        log.info("End of Train")
        
    except Exception as e:
            log.error(traceback.format_exc())




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
    def __init__(self, cfg, vis=None):
        self._cfg = cfg
        self.vis = vis
        self.plot = cfg.TRAIN.PLOT_LOSS
        self.epochbatchs = cfg.TRAIN.EPOCHBATCHS
        self.log_period = cfg.TRAIN.LOG_PERIOD
        self.loss_meters = defaultdict(lambda: ScalarMeter(cfg.TRAIN.LOG_PERIOD))
        self.spacer = 0
        self.log = logger.get_log(cfg, __name__)
        
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