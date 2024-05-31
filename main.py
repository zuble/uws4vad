import os, os.path as osp
import pyrootutils

import torch
print(f"{torch.__version__=}")
import hydra
from hydra.utils import instantiate as instantiate
from omegaconf import DictConfig, OmegaConf

from src import utils
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "cfg"),
    "config_name": "001.yaml",
}

@utils.reg_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    log = utils.get_log(__name__, cfg)
    #log.debug(f"Working dir : {os.getcwd()}")
    #log.debug(f"Output dir  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    #log.debug(f"Original dir : {hydra.utils.get_original_cwd()}")

    #log.debug(utils.collect_random_states())
    
    if cfg.get("tmp"):
        from src import tmp
        utils.xtra(cfg)
        
        tmp.Debug(cfg)
        
        
    elif cfg.get("test"):
        from src import test
        utils.init_seed(cfg, False)
        utils.xtra(cfg)
        test.test(cfg)
            
    elif cfg.get("train"):
        from src import train
        utils.init_seed(cfg)
        utils.xtra(cfg)
        loader = train.trainer(cfg)


if __name__ == "__main__":
    ## tf32
    ## https://pytorch.org/docs/1.8.1/notes/cuda.html#tf32-on-ampere
    print(torch.tensor([1.2, 3]).dtype )
    print(f"{torch.backends.cuda.matmul.allow_tf32=} {torch.backends.cudnn.allow_tf32=}")
    print(f"{torch.get_default_dtype()=}")
    #torch.set_default_dtype('float32')
    os.environ.setdefault('HYDRA_FULL_ERROR', '1')
    #torch.cuda.set_per_process_memory_fraction(0.7, device=f"cuda:0")
    
    print('\n\n',' '*11,'_'*33,'\n\n')
    main()