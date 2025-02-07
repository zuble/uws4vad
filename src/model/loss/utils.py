import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.pstfwd.utils import PstFwdUtils

from hydra.utils import instantiate as instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import logging
from src.utils.logger import get_log
log = get_log(__name__)


## ---- builder ----
def inject_pstfwd_utils(loss_class, pstfwd_utils, attrs_to_inject=None):
    """
    Injects attributes from pstfwd_utils into loss_class.

    Args:
        loss_class: The loss class instance.
        pstfwd_utils: The PostProcessUtils instance.
        attrs_to_inject (list, optional): A list of attribute names to inject. 
            If None, all attributes of pstfwd_utils will be injected.
    """
    import types
    
    if attrs_to_inject is None:
        # Get all public attributes (excluding methods and private attributes)
        attrs_to_inject = [
            attr for attr in dir(pstfwd_utils) \
            if not attr.startswith('_') \
            #and not callable(getattr(pstfwd_utils, attr))
            ]
        
    for attr_name in attrs_to_inject:
        if hasattr(loss_class, attr_name):
            log.warning(f"Attribute collision: '{attr_name}' already exists in '{loss_class.__class__.__name__}'. Skipping injection.")
            continue  # Skip injection if attribute already exists
        
        #setattr(loss_class, attr_name, getattr(pstfwd_utils, attr_name))
        attr_value = getattr(pstfwd_utils, attr_name)
        if callable(attr_value):  # If it's a function
            # Bind the function to the pstfwd_utils instance
            attr_value = types.MethodType(attr_value, pstfwd_utils) 
        setattr(loss_class, attr_name, attr_value)
            
def build_loss(cfg):
    #log.info(cfg.model.loss)
    #lossfx = {lid: instantiate(cfg.model.loss[lid], _convert_="partial") for lid in cfg.model.loss}
    lossfx = {}
    pstfwd_utils = instantiate(cfg.model.pstfwd)

    for lid, lcfg in cfg.model.loss.items():
        target = lcfg._target_
        log.debug(f"{target} {lcfg}")

        if target.split('.')[-1][0].isupper():
            #loss_instance = instantiate(lcfg)
            #log.warning(f"PRE INJECTION  {target} {dir(loss_instance)}") 
            #inject_pstfwd_utils(loss_instance, pstfwd_utils)
            #log.warning(f"POST INJECTION  {target} {dir(loss_instance)}") 
            ### or
            loss_cfg = DictConfig({
                '_target_': target,
                '_cfg': {
                    **{k: v for k, v in lcfg.items() if k != '_target_'}
                }
            })
            #log.error(loss_cfg)
            loss_instance = instantiate(loss_cfg, pfu=pstfwd_utils)
            
            lossfx[lid] = loss_instance

        else: ## a function, set partial 
            lossfx[lid] =  instantiate(lcfg, _convert_="partial", _partial_=True) 
            
    for key, value in lossfx.items():
        log.info({key: value})
    
    l_comp = LossComputer(lossfx, pstfwd_utils)
        
    return lossfx, l_comp



## ---- processor ----
def print_grad_fn_info(tensor, tensor_name="Tensor"):
    """Prints information about the grad_fn of a tensor."""
    if tensor.grad_fn is not None:
        log.debug(f"{tensor_name} grad_fn:")
        log.debug(f" - Type: {type(tensor.grad_fn)}")
        log.debug(f" - Name: {tensor.grad_fn.__class__.__name__}")
        log.debug(f" - Device: {tensor.device}")
        if hasattr(tensor.grad_fn, 'next_functions'):
            log.debug(" - Inputs:")
            for i, func in enumerate(tensor.grad_fn.next_functions):
                if func[0] is not None:
                    log.debug(f"   - Input {i}: {func[0].__class__.__name__}")
        log.debug("-" * 20)
    else:
        log.debug(f"{tensor_name} has no grad_fn (likely a leaf variable).")
        log.debug("-" * 20)

class LossComputer(nn.Module):
    def __init__(self, lfxs, pfu: PstFwdUtils):
        super().__init__()
        self.lfxs = lfxs
        self.pfu = pfu
        #self._ = pfu.

    def forward(self, ndata, ldata):
        if log.isEnabledFor(logging.DEBUG):
            self.pfu.logdat(ndata,ldata)
        
        loss_glob = torch.tensor(0., requires_grad=True, device=self.pfu.dvc)
        loss_dict = {}

        for loss_name, loss_fn in self.lfxs.items():
            
            ## ndata = self.pfu.some_predifend_pipeline(ndata)
    
            loss_output = loss_fn(ndata, ldata)
            if log.isEnabledFor(logging.DEBUG):
                self.pfu.logdat(loss_output)
            
            ## this ugly !!!!!!
            for component_name, component_value in loss_output.items():
                #log.debug(f"{component_name}  {component_value}")
                print_grad_fn_info(component_value,component_name)
                loss_glob = loss_glob + component_value 
                
                loss_dict[f"{loss_name}/{component_name}"] = component_value
                
        print_grad_fn_info(loss_glob, "Final Loss") 
        return loss_glob, loss_dict