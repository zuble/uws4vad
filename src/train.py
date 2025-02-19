import torch
import numpy as np
import math 

from hydra.utils import instantiate as instantiate, call as call
import time , os, os.path as osp, gc, traceback, sys
from collections import defaultdict, deque
#from tqdm import tqdm
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from src.data import get_trainloader, get_testloader, run_dl
from src.model import ModelHandler, build_net, build_loss
from src.vldt import Validate, Metrics, Plotter, Tabler
from src.utils import hh_mm_ss, get_log
log = get_log(__name__)


def trainer(cfg, vis):
    trn_inf = {
        'epo': 0,
        'bat': 0,
        'step': 0,
        'ttic': None,
        'dvc': cfg.dvc
    }
    
    ## DATA
    dataloader, collator = get_trainloader(cfg, vis) 
    if cfg.dataload.dryrun: run_dl(dataloader,iters=2,vis=True) ;return
    
    ## MODEL
    net, inferator = build_net(cfg)
    net.to(cfg.dvc)
    
    optima = instantiate(cfg.model.optima, params=net.parameters() , _convert_="partial") #;log.info(f"{optima=}")
    _, loss_computer = build_loss(cfg)
    #return
    
    ## LOAD
    MH = ModelHandler(cfg, True)
    if cfg.load.get("chkpt_path"): 
        ## net_arch/optima need to match struct of ones in mstate
        mstate = MH.get_train_state(trn_inf)
        net.load_state_dict(mstate["net"]) #, strict=True
        optima.load_state_dict(mstate["optima"])

    ## LRS
    lrs = None
    if cfg.model.get("lrs"):
        log.debug(cfg.model.lrs)
        lrs = instantiate( cfg.model.lrs, optimizer=optima, _convert_="partial")
        log.debug(lrs)
    
    ## VALIDATE
    cfg_vldt = cfg.vldt.train
    VLDT = Validate(cfg, cfg_vldt, cfg.data.frgb, vis=vis)
    if cfg.vldt.dryrun:
        log.info("DBG DRY VLDT RUN")
        VLDT.start(net, inferator); VLDT.reset() ;return    
    tmeter = TrainMeter( cfg.dataload, cfg_vldt, vis)
    MTRC = Metrics(cfg_vldt, vis)
    
    #progress = Progress(
    #    SpinnerColumn(),
    #    TextColumn("[progress.description]{task.description}"),
    #    BarColumn(),
    #    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    #    TimeRemainingColumn(),
    #    expand=True
    #)

    try:
        ## Live display context manager
        #with Live(progress, refresh_per_second=10):
        #    task1 = progress.add_task("[cyan]Training...", total=cfg.model.epochs)
            
        trn_inf['ttic'] = time.time()
        #for epo in tqdm(range(0,cfg.model.epochs), desc="Epoch", unit='epo', file=sys.stdout):
        for epo in range(cfg.model.epochs):
            trn_inf['epo'] = epo+1
            trn_inf['etic'] = time.time()
            
            ##########
            net.train()    
            #for batch in tqdm(dataloader, leave = False, desc="Batch:", unit='bat'):
            for bi, batch in enumerate(dataloader):
                trn_inf['bat'] =+ 1
                trn_inf['step'] =+ 1
                btic = time.time()
                
                ## data collate
                feat, ldata = collator(batch, trn_inf) ## get feat + fill loss metadata
                for key in list(ldata.keys())[:]: log.debug(f"\t\t\t{key} {list(ldata[key].shape)}") if type(ldata[key]) == torch.Tensor else None
                
                ## fwd
                ndata = net(feat)
                log.debug(f"{list(feat.shape)} -> ")
                for key in list(ndata.keys())[:]: log.debug(f"\t\t\t{key} {list(ndata[key].shape)}") if type(ndata[key]) == torch.Tensor else None
                ## if ndata['id'] == '...':
                
                loss_glob, loss_indv = loss_computer(ndata, ldata)
                
                #######
                optima.zero_grad()
                loss_glob.backward()
                if cfg.model.get("clipnorm"): ## bndfm
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optima.step() 
                #######
                
                ## monitor
                if loss_glob.item() > 1e8 or math.isnan(loss_glob.item()):
                    log.error(f"E[{trn_inf['epo']}] B[{trn_inf['bat']}] S[{(trn_inf['step'])}] Loss exploded {loss_glob.item()}")
                    log.error(loss_indv)
                    raise Exception("Loss exploded")
                
                tmeter.update({'global': loss_glob.item()})
                for key, value in loss_indv.items():
                    ## Update individual loss parts
                    tmeter.update({key: value.item()})  
                tmeter.log_bat(trn_inf)
                
                abn_bag_len = torch.sum(ldata["label"] == 1).item()
                vis.plot_lines('abn/nor bag ratio', 
                    abn_bag_len / ldata["label"].shape[0],
                    opts=dict(
                        title=f"Batch vs. abn/nor bag ratio",  #Abnormal Bag Ratio
                        ylabel='Abn Bag Tatio'
                    )
                )
                
                #if bi % 20 == 0:
                #    log.warning(f"{bi} ")
                #    *_ = VLDT.start(net, inferator)
                #    VLDT.reset()
                
            if lrs is not None: lrs.step()
                
            ## monitor 
            tmeter.log_epo(trn_inf)
            tmeter.reset()
            
            ## eval
            if (epo + 1) % cfg_vldt.freq == 0:
                log.info(f'$$$ Validation at epo {epo+1}')
                #torch.backends.cudnn.benchmark = False
                vldt_info, _ = VLDT.start(net, inferator)
                mtrc_info, curv_info, table_res = MTRC.get_fl(vldt_info)
                VLDT.reset()
                #torch.backends.cudnn.benchmark = True
                for table in table_res: log.info(f'\n{table}')
                if not cfg.get("debug"): 
                    MH.record( mtrc_info, curv_info, table_res, net, optima, trn_inf)
            
            #progress.update(task1, advance=1)
            
        log.info(f"$$$$ train done in {hh_mm_ss(time.time() - trn_inf['ttic'])}")
        if not cfg.get("debug"): ## move       <<<<<<<<<
            if cfg.get("save"): MH.save_state()
            if MH.high_info['rec_val'] > 0.5:
                pltr = Plotter(vis)
                pltr.metrics(MH.high_state['mtrc_info'])
                pltr.curves(MH.high_state['curv_info'])
                Tabler(send2visdom=True,vis=vis).table2img(MH.high_table,'high_res_table')
        
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
            self.logfreq = cfg_loader.itersepo//2 ## !!
        else:
            self.logfreq = cfg_vldt.loss_log

        self.vis = vis
        self.plot = cfg_vldt.loss_visplot
        self.loss_meters = defaultdict(lambda: ScalarMeter(self.logfreq))
        self.spacer = 0
        
    def update(self, lbat_indv):
        for loss_name, l in lbat_indv.items():
            self.loss_meters[loss_name].add_value(l)
            if self.plot:
                self.vis.plot_lines(
                    f"bat-{loss_name}",
                    l,
                    opts=dict(
                        title=f"Batch Loss - {loss_name}",
                        xlabel='Batch',
                        ylabel='Loss',
                        showlegend=True
                    )
                )    
                
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
                self.vis.plot_lines(
                    f"epo-{loss_name}",
                    meter.get_global_avg(),
                    opts=dict(
                        title=f"Average Loss per Epoch - {loss_name.capitalize()}",  
                        xlabel='Epoch',
                        ylabel='Loss',
                        showlegend=True
                    )
                )
        sms += f" @ {hh_mm_ss(time.time()-trn_inf['etic'])}"
        log.info(f"{sms}")