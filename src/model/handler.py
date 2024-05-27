import os
import os.path as osp
from collections import OrderedDict
import copy
import math

import torch
import torch.nn

from omegaconf import OmegaConf
from tqdm import tqdm


def weight_init(m):
    #classname = m.__class__.__name__
    #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        tc_init.xavier_uniform_(m.weight)
        #tc_init.constant_(m.bias, 0)
        if m.bias is not None:
            m.bias.data.fill_(0)


def count_parms(net):
    t = sum(p.numel() for p in net.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')
    
#log.info(f"Net structure: {net}\n\n") if 'blck' in cfg.NET.LOG_INFO else None 
#if 'prmt' in cfg.NET.LOG_INFO:
#    for name, param in net.named_parameters():
#        log.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def save(cfg, net):
    ## https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    ## https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
    PATH = osp.join(cfg.EXPERIMENTPATH,f'{cfg.NET.NAME}')

    try: 
        ## net arch&params
        p = f'{PATH}_{cfg.SEED}.net.pt'
        torch.save(net, p)
        log.info(f'µµµ SAVED full netowrk {cfg.NET.NAME}.pt in \n{p}')
    except Exception as e: log.error(f'{cfg.NET.NAME}.export: {e}')  
    
    try:    
        ## net params / state_dict (needs same net def) 
        p =  f'{PATH}_{cfg.SEED}.dict.pt'
        torch.save(net.state_dict(), p)
        log.info(f'µµµ SAVED state_dict of {cfg.NET.NAME}.pt in \n{cfg.EXPERIMENTPATH}')
    except Exception as e: log.error(f'{cfg.NET.NAME}-state_dict.save: {e}')  
    
def get_ldnet(cfg, dvc):
    
    if not cfg.TEST.LOADFROM: raise Exception (f'cfg.TEST.LOADFROM must be set')
    
    load_dict = parse_ptfn(cfg.TEST.LOADFROM)
    
    PATH = osp.join(cfg.EXPERIMENTPATH, f"{cfg.TEST.LOADFROM}.pt")
    log.info(f"loading {load_dict['mode']} from {PATH}")
    
    if load_dict['mode'] == 'net': 
        net = torch.load(PATH)
    elif load_dict['mode'] == 'dict':
        net = get_net(cfg, dvc)
        net.load_state_dict( torch.load(PATH) )
    else: raise Exception(f'{cfg.TEST.LOADFROM = } and should be (...).["net","dict"]')
    
    return net, load_dict['seed']


class ModelHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dvc = self.cfg.dvc

        self.step = 0
        self.epoch = -1
        self._logger = get_logger(cfg, os.path.basename(__file__))

    def train_epo(self, traindl):
        #logger = get_logger(self.cfg, os.path.basename(__file__), disable_console=True)
        
        self.net.train()
        for tdata in tqdm(traindl, leave = False, desc="Training/Batch:", unit='batch'):
            
            model_input = model_input.to(self.device)
            
            output = self.net(model_input)
            ndata = net(feat)
            log.debug(f"{feat.shape} -> ")
            for key in list(ndata.keys())[1:]: log.debug(f"    {key} {ndata[key].shape}") if type(ndata[key]) == torch.Tensor else None
            ## if ndata['id'] == '...':
            
            loss_indv = netpstfwd.train(ndata, ldata, lossfx) 
            loss_glob = torch.sum(torch.stack(list(loss_indv.values())))
            
            
            self.optimizer.zero_grad()
            loss_glob.backward()
            self.optimizer.step()
            
            # set log
            self.log.loss_v = loss_v.item()
            loss = self.log.loss_v
            
            self.step += 1
            
            if loss > 1e8 or math.isnan(loss):
                logger.error("Loss exploded to %.02f at step %d!" % (loss, self.step))
                raise Exception("Loss exploded")


            if self.step % self.cfg.log.summary_interval == 0:
                logger.info("Train Loss %.04f at step %d" % (loss, self.step))


    def validate(self, test_loader):
        ## enter validate here
        
        self.net.eval()
        with torch.no_grad():
            for tdata in tqdm(test_loader, leave = False, desc="Testing/Batch:"):
                output = self.run_network(model_input)
    
                
    def save_network(self, save_file=True):
        state_dict = self.net.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.to("cpu")
        if save_file:
            fname = "%s_%d.pt" % (self.epoch, self.step)
            path = osp.join(self.cfg.log.chkpt_dir, fname)
            torch.save(state_dict, save_path)
            self._logger.info("Saved network checkpoint to: %s" % save_path)
        return state_dict
    
    def load_network(self, loaded_net=None):
        
        if loaded_net is None:
            loaded_net = torch.load(
                self.cfg.load.network_chkpt_path,
                map_location=torch.device(self.device),
            )
        loaded_clean_net = OrderedDict()  # remove unnecessary 'module.'
        for k, v in loaded_net.items():
            if k.startswith("module."):
                loaded_clean_net[k[7:]] = v
            else:
                loaded_clean_net[k] = v

        self.net.load_state_dict(loaded_clean_net, strict=self.cfg.load.strict_load)
        self._logger.info(
            "Checkpoint %s is loaded" % self.cfg.load.network_chkpt_path
        )

    def save_training_state(self):
        tmp = "%s_%d.state" % (self.epoch, self.step)
        save_path = osp.join(self.cfg.path.work_dir, tmp)
        
        net_state_dict = self.save_network(False)
        state = {
            "model": net_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        torch.save(state, save_path)
        self._logger.info("Saved training state to: %s" % save_path)

    def load_training_state(self):
        resume_state = torch.load(
            self.cfg.load.resume_state_path,
            map_location=torch.device(self.device),
        )
        
        self.load_network(loaded_net=resume_state["model"])
        self.optimizer.load_state_dict(resume_state["optimizer"])
        self.step = resume_state["step"]
        self.epoch = resume_state["epoch"]
        
        self._logger.info(
            "Resuming from training state: %s" % self.cfg.load.resume_state_path
        )
