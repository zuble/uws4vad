import os, os.path as osp
import pyrootutils

import torch
#print(f"{torch.__version__=}")
import hydra
from hydra.utils import instantiate as instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

import uws4vad
from uws4vad import model, utils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "src/config"),
    "config_name": "xdv.yaml", ## -> override w/ -cn=
}

#@utils.cleanup_on_exit()
@utils.reg_custom_resolvers(**_HYDRA_PARAMS)  ## @uws4vad/utils/cfgres.py
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    log = utils.get_log(__name__)
    
    #cleaner = utils.cleanup_on_exit.get_cleaner()
    #if not cleaner: raise RuntimeError("Cleanup manager not initialized!")
        
    if cfg.get("tmp"):
        from uws4vad import tmp
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
            'rgb': ('uws4vad.fext', 'VisFeatExtract'),
            'aud': ('uws4vad.fext', 'AudFeatExtract')
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
            from uws4vad.train import trainer
            utils.init_seed(cfg)
            utils.xtra(cfg)
            
            trainer(cfg, vis)
            
            ## 4 detailed inspection
            if cfg.get("test"):
                from uws4vad.test import tester
                tester(cfg, vis)
        
        elif cfg.get("test"):
                from uws4vad.test import tester
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
