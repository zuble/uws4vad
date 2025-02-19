import os, os.path as osp
import pyrootutils

import torch
#print(f"{torch.__version__=}")
import hydra
from hydra.utils import instantiate as instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from src import utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "cfg"),
    "config_name": "xdv.yaml", ## -> override w/ -cn=
}

#@utils.cleanup_on_exit()
@utils.reg_custom_resolvers(**_HYDRA_PARAMS)  ## @src/utils/cfgres.py
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    log = utils.get_log(__name__, cfg) ## -> runs once 2 set right lvl's per module
    
    #cleaner = utils.cleanup_on_exit.get_cleaner()
    #if not cleaner: raise RuntimeError("Cleanup manager not initialized!")
        
    if cfg.get("tmp"):
        from src import tmp
        utils.xtra(cfg)
        
        #func_name = cfg.get("tmp")  
        #target_func = getattr(tmp, func_name) 
        #target_func(cfg)
        ## or
        #instance = target_func()  
        
        tmp.Debug(cfg).testset()
        #tmp.Debug(cfg)
        #tmp.aud_emb(cfg)
        #tmp.aud_len_mat()

    ## feature extraction
    elif cfg.get("fext"):
        utils.xtra(cfg)
        
        modal_handler = {
            'rgb': ('src.fext', 'VisFeatExtract'),
            'aud': ('src.fext', 'AudFeatExtract')
        }.get(cfg.modal)
        if not modal_handler:
            raise ValueError(f"Unsupported modality: {cfg.modal}")

        module = __import__(modal_handler[0], fromlist=[modal_handler[1]])
        extractor = getattr(module, modal_handler[1])
        extractor(cfg)

    ## train/test.py    
    else:
        vis = utils.Visualizer(cfg)
        vis.textit(['out_dir',cfg.path.out_dir,
                    #'choices', HydraConfig.get().runtime.choices, 
                    'overrides', HydraConfig.get().overrides.task,
                    'sweeper', HydraConfig.get().sweeper.params
                    ])
        # Register Visdom cleanup
        #cleaner.register(
        #    vis.delete, 
        #    envn=vis.env_name,
        #    #reason="Session cleanup"
        #)
        
        if cfg.get("train"):
            from src.train import trainer
            utils.init_seed(cfg)
            utils.xtra(cfg)
            
            trainer(cfg, vis)
            
            ## 4 detailed inspection
            if cfg.get("test"):
                from src.test import tester
                tester(cfg, vis)
        
        elif cfg.get("test"):
                from src.test import tester
                ## with multi nets to iter, 
                #utils.init_seed(cfg, False)
                utils.xtra(cfg)
                tester(cfg, vis)
        
        else: log.error("任选其一 fext / tmp / train / test")

    
if __name__ == "__main__":
    ## tf32
    ## https://pytorch.org/docs/1.8.1/notes/cuda.html#tf32-on-ampere
    #print(torch.tensor([1.2, 3]).dtype )
    #print(f"{torch.backends.cuda.matmul.allow_tf32=} {torch.backends.cudnn.allow_tf32=}")
    #print(f"{torch.get_default_dtype()=}")
    #torch.set_default_dtype('float32')
    #os.environ.setdefault('HYDRA_FULL_ERROR', '1')
    #torch.cuda.set_per_process_memory_fraction(0.7, device=f"cuda:0")
    #print('\n\n',' '*11,'_'*33,'\n\n')
    
    main()
