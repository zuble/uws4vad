import torch
import numpy as np
import math 

from hydra.utils import instantiate as instantiate
import time , os, os.path as osp, gc, traceback
from collections import defaultdict, deque
from tqdm import tqdm

from src.data import get_trainloader, get_testloader, run_dl
from src.model import ModelHandler
from src.vldt import Validate, Metrics
from src.utils import hh_mm_ss, get_log, Visualizer
log = get_log(__name__)


def trainer(cfg):
    traindl, trainfrmt = get_trainloader(cfg) 
    if cfg.get("debug").get("data") > 1: run_dl(traindl) #;return
    
    model_handler = ModelHandler(cfg)
    
    ## ARCH
    #log.info(cfg.model.net.arch)
    feat_len = cfg.dl.ds.frgb.nfeats + (cfg.dl.ds.faud.nfeats if cfg.dl.ds.get("faud") else 0)
    net = instantiate(cfg.model.net.arch, feat_len=feat_len).to(cfg.dvc)
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
    lrs = None
    if cfg.model.get("lrs"):
        log.info(cfg.model.lrs)
        lrs = instantiate( cfg.model.lrs, optimizer=optima, _convert_="partial")
        log.info(lrs)
    
    ## LOSSFX
    lossfx = {lid: instantiate(cfg.model.loss[lid]) for lid in cfg.model.loss}
    for key, value in lossfx.items():
        log.info({key: value})
    ldata = {}
    #aux_params = set()    
    #for lid, lcfg in cfg.model.loss.items():
    #    if lcfg.get("aux"):
    #        aux_params.update(lcfg.aux)
    #ldata = {p: None for p in aux_params}
    #for key, value in ldata.items():
    #    ## Update individual loss parts
    #    log.info({key: value}) 
    
    ## Monitor
    vis = Visualizer(f'{cfg.name}/{cfg.task_name}', restart=False, del_all=False)
    #vis.delete(f'{cfg.name}/{cfg.task_name}')
    
    cfg_vldt = cfg.vldt.train
    vldt = Validate(cfg, cfg_vldt, cfg.dl.ds.frgb, vis)
    if cfg.get("debug").get("vldt") > 1: vldt.start(net) #;return
    
    
    ## Train
    tmeter = TrainMeter( cfg.dl.loader.train, cfg_vldt, vis)
    trn_inf = {
        'epo': 0,
        'bat': 0,
        'step': 0,
        'ttic': time.time(),
        'dvc': cfg.dvc
    }
    try:
        #for epo in tqdm(range(0,cfg.model.epochs), desc="Epoch", unit='epo'):
        for epo in range(cfg.model.epochs):
            trn_inf['epo'] = epo+1
            trn_inf['etic'] = time.time()
            
            ##########
            net.train()    
            #for tdata in tqdm(traindl, leave = False, desc="Batch:", unit='bat'):
            for tdata in traindl:
                trn_inf['bat'] =+ 1
                trn_inf['step'] =+ 1
                btic = time.time()

                feat = trainfrmt.fx(tdata, ldata, trn_inf)
                ndata = net(feat)
                log.debug(f"{feat.shape} -> ")
                for key in list(ndata.keys())[:]: log.debug(f"    {key} {ndata[key].shape}") if type(ndata[key]) == torch.Tensor else None
                ## if ndata['id'] == '...':

                loss_indv = netpstfwd.train(ndata, ldata, lossfx) 
                loss_glob = torch.sum(torch.stack(list(loss_indv.values())))
                
                #######
                optima.zero_grad()
                loss_glob.backward()
                optima.step() 
                #######
                
                if loss_glob.item() > 1e8 or math.isnan(loss_glob.item()):
                    logger.error(f"E[{trn_inf['epo']}] B[{trn_inf['bat']}] S[{(trn_inf['step'])}] Loss exploded {loss_glob.item()}")
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
                torch.backends.cudnn.benchmark = False
                _, _, mtrc_info = vldt.start(net, netpstfwd)
                if cfg_vldt.mtrc_visplot: vis.plot_vldt_mtrc(mtrc_info,epo)
                vldt.reset()
                torch.backends.cudnn.benchmark = True
        
        log.info(f"$$$$ train done in {hh_mm_ss(time.time() - trn_inf['ttic'])}")
        
        if not cfg.get("debug"): 
            model.save_net(net, trn_info, )

    except Exception as e:
        log.error(traceback.format_exc())



    ## load training state / network checkpoint
    #if cfg.load.get("resume_state_path"): 
    #    model.load_training_state(net, optima, trn_inf)
    #elif cfg.load.get("network_chkpt_path"): 
    #    model.load_network(net)
    #else: log.info("Starting new training run.")
    #
    #if model.epoch % cfg.log.chkpt_interval == 0:
    #    model.save_network(net, trn_inf)
    #    model.save_training_state()




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
        
        if cfg_vldt.get("loss_log") == 0:
            self.logfreq = cfg_loader.itersepo//2
        else:self.logfreq = cfg_vldt.loss_log

        self.vis = vis
        self.plot = cfg_vldt.loss_visplot
        self.loss_meters = defaultdict(lambda: ScalarMeter(cfg_vldt.loss_log))
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
            if self.plot: self.vis.plot_lines(f"epo-{loss_name}", meter.get_global_avg())
        sms += f" @ {hh_mm_ss(time.time()-trn_inf['etic'])}"
        log.info(f"{sms}")