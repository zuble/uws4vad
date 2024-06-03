import torch
import numpy as np
import math 

from hydra.utils import instantiate as instantiate, call as call
import time , os, os.path as osp, gc, traceback, sys
from collections import defaultdict, deque
from tqdm import tqdm

from src.data import get_trainloader, get_testloader, run_dl
from src.model import ModelHandler, build_net
from src.vldt import Validate, Metrics
from src.utils import hh_mm_ss, get_log, Visualizer
log = get_log(__name__)


def trainer(cfg):
    trn_inf = {
        'epo': 0,
        'bat': 0,
        'step': 0,
        'ttic': None,
        'dvc': cfg.dvc
    }
    
    ## DATA
    traindl, trainfrmt = get_trainloader(cfg) 
    if cfg.dataloader.train.dryrun: run_dl(traindl) #;return
    
    ## MODEL
    net, netpstfwd = build_net(cfg)
    optima = instantiate(cfg.model.optima, params=net.parameters() ) #, _convert_="partial"
    
    ## LOAD
    MH = ModelHandler(cfg)
    if cfg.load.get("chkpt_path"): 
        ## net_arch/optima need to match struct of ones in mstate
        net, optima = MH.load_train_state(trn_inf, net_arch=net, optima=optima)
    else: log.info("Starting fresh train")
        
    ## LRS
    lrs = None
    if cfg.model.get("lrs"):
        log.debug(cfg.model.lrs)
        lrs = instantiate( cfg.model.lrs, optimizer=optima, _convert_="partial")
        log.debug(lrs)
    
    ## LOSSFX
    #log.info(cfg.model.loss)
    #lossfx = {lid: instantiate(cfg.model.loss[lid], _convert_="partial") for lid in cfg.model.loss}
    lossfx = {}
    ## if lossfx is a function, set partial
    ## if class do normal
    for lid, lcfg in cfg.model.loss.items():
        if '_target_' in lcfg:
            target = lcfg._target_
            if target.split('.')[-1][0].isupper(): lossfx[lid] = instantiate(lcfg)
            else: lossfx[lid] =  instantiate(lcfg, _convert_="partial", _partial_=True)
    for key, value in lossfx.items():
        log.debug({key: value})
    ldata = {}
    
    ## MONITOR
    vis = Visualizer(f'{cfg.name} / {cfg.task_name}', restart=True, del_all=False)
    #vis.delete(f'{cfg.name}/{cfg.task_name}')
    
    cfg_vldt = cfg.vldt.train
    vldt = Validate(cfg, cfg_vldt, cfg.data.frgb, vis)
    if cfg.vldt.dryrun:
        log.info("DBG DRY VLDT RUN")
        vldt.start(net, netpstfwd); vldt.reset() #;return
        
    tmeter = TrainMeter( cfg.dataloader.train, cfg_vldt, vis)
    
    try:
        trn_inf['ttic'] = time.time()
        for epo in tqdm(range(0,cfg.model.epochs), desc="Epoch", unit='epo', file=sys.stdout):
        #for epo in range(cfg.model.epochs):
            trn_inf['epo'] = epo+1
            trn_inf['etic'] = time.time()
            
            ##########
            net.train()    
            #for tdata in tqdm(traindl, leave = False, desc="Batch:", unit='bat'):
            for tdata in traindl:
                trn_inf['bat'] =+ 1
                trn_inf['step'] =+ 1
                btic = time.time()

                feat = trainfrmt.fx(tdata, ldata, trn_inf) ## get feat + fill loss metadata
                ndata = net(feat)
                log.debug(f"{list(feat.shape)} -> ")
                for key in list(ndata.keys())[:]: log.debug(f"\t\t\t{key} {list(ndata[key].shape)}") if type(ndata[key]) == torch.Tensor else None
                ## if ndata['id'] == '...':

                loss_indv = netpstfwd.train(ndata, ldata, lossfx) 
                loss_glob = torch.sum(torch.stack(list(loss_indv.values())))
                
                #######
                optima.zero_grad()
                loss_glob.backward()
                optima.step() 
                #######
                
                if loss_glob.item() > 1e8 or math.isnan(loss_glob.item()):
                    log.error(f"E[{trn_inf['epo']}] B[{trn_inf['bat']}] S[{(trn_inf['step'])}] Loss exploded {loss_glob.item()}")
                    log.error(loss_indv)
                    raise Exception("Loss exploded")
                
                ## monitor
                tmeter.update({'bat': loss_glob.item()})
                for key, value in loss_indv.items():
                    ## Update individual loss parts
                    tmeter.update({key: value.item()})  
                tmeter.log_bat(trn_inf)
            
            if lrs is not None: lrs.step()
            ##########
                
            ## monitor 
            tmeter.log_epo(trn_inf)
            tmeter.reset()
            
            ## eval
            if (epo + 1) % cfg_vldt.freq == 0:
                log.info(f'$$$ Validation at epo {epo+1}')
                #torch.backends.cudnn.benchmark = False
                _, _, mtrc_info = vldt.start(net, netpstfwd)
                if cfg_vldt.mtrc_visplot: vis.plot_vldt_mtrc(mtrc_info,epo)
                vldt.reset()
                #torch.backends.cudnn.benchmark = True
                MH.record( mtrc_info, net, optima, trn_inf)
                
            
        log.info(f"$$$$ train done in {hh_mm_ss(time.time() - trn_inf['ttic'])}")
        
        if not cfg.get("debug"): 
            MH.save_state(net, optima, trn_inf)

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
    def __init__(self, cfg_loader, cfg_vldt, vis):
        
        if cfg_vldt.loss_log == 0:
            self.logfreq = cfg_loader.itersepo//2
        else:
            self.logfreq = cfg_vldt.loss_log

        self.vis = vis
        self.plot = cfg_vldt.loss_visplot
        self.loss_meters = defaultdict(lambda: ScalarMeter(self.logfreq))
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
        if (trn_inf['bat']) % self.logfreq == 0:
            sms = f" E[{trn_inf['epo']}] B[{trn_inf['bat']}] S[{(trn_inf['step'])}]"
            self.spacer = len(sms)
            for loss_name, meter in self.loss_meters.items():
                sms += f" {loss_name} {meter.get_win_avg():.4f}" #Med {meter.get_win_median():.4f},
            log.info(f"{sms}")
            
    def log_epo(self, trn_inf):
        ## logs / plots to visdom loss_epo_..
        sms = f"*E[{trn_inf['epo']}]"
        if self.spacer: sms += " "*(self.spacer-len(sms))
        
        for loss_name, meter in self.loss_meters.items():
            sms += f" {loss_name} {meter.get_global_avg():.4f}"
            
            if self.plot: 
                self.vis.plot_lines(f"epo-{loss_name}", meter.get_global_avg())
        
        sms += f" @ {hh_mm_ss(time.time()-trn_inf['etic'])}"
        log.info(f"{sms}")