import torch
import torch.nn as nn

from hydra.utils import instantiate as instantiate
from omegaconf.dictconfig import DictConfig

from uws4vad.common.registry import registry
from uws4vad.utils.logger import get_log
log = get_log(__name__)

class BaseNetwork(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg.net
        log.debug(f"Initial {self.cfg=}")
        
        ## [rgb, aud]
        self.dfeat = [ cfg.data.frgb.dfeat, (cfg.data.faud.dfeat if cfg.data.get("faud") else 0)]
        log.debug(f"Initial {self.dfeat=}")
        ## each Network __init__ is responsible to assign correct 
        ## length of fm/cls input layer when calling _build 
        ## self.__ = self._build("__", din=...)


    def _build(self, component_type, dfeat, **kwargs):
        """Build a component (fm: feature modulator or cls: classifier) using Hydra's instantiate or registry."""
        if component_type not in self.cfg:
            return None

        component_cfg = self.cfg[component_type]
        
        ## ---------
        # Case 1: Use Hydra's instantiate if _target_ is present
        if "_target_" in component_cfg:
            log.debug(f"Instantiating {component_type} via Hydra: {component_cfg['_target_']}")
            component_cfg = DictConfig({
                '_target_': component_cfg._target_,
                'din': dfeat,
                **{k: v for k, v in component_cfg.items() if k != '_target_'}
            })
            log.debug(f"{component_cfg=}")
            component_cfg = {**component_cfg, **kwargs}
            log.debug(f"{component_cfg=}")
            return instantiate(component_cfg)

        ## ---------
        # Case 2: Fallback to registry-based construction
        #if override_name is not None: 
        #    #override config and use the config in cfg.net.fm.{override_name}.yaml to build
        #    # TODO
        #    raise NotImplementedError 
        #    raise ValueError(f"No default name provided for {component_type} and no _target_ in config.")
        
        if "name" not in component_cfg:
            raise ValueError(f"'name' key not provided for {component_type} in config.")
        
        log.debug(f"Instantiating {component_type} via Registry: {component_cfg['name']}")
        name = component_cfg.name
        component_cfg = DictConfig({ 
            **{k: v for k, v in component_cfg.items() if k != 'name' }
        })  
        log.debug(f"{component_cfg=}")
        component_cfg = {**component_cfg, **kwargs}
        log.debug(f"{component_cfg=}")
        
        try:
            component_class = registry.get(component_type, name)
        except KeyError as e:
            raise ValueError(f"Component name {name} not registered for {component_type}.") from e
        
        # Pass dfeat and respective cfg
        return component_class(din=dfeat, **component_cfg)
    
    def forward(self, x):
        # Your forward logic here
        pass