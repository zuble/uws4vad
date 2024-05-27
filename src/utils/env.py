import os
import random
from typing import Any, Dict, Callable

import torch
import numpy as np



from src.utils.logger import get_log
log = get_log(__name__)


def init_seed(cfg, istrain=True):
    ## https://pytorch.org/docs/1.8.1/notes/randomness.html
    ## https://github.com/henrryzh1/UR-DMU/blob/master/utils.py#L34
    ## https://github.com/Roc-Ng/DeepMIL/blob/master/main.py#L9
    ## https://github.com/Lightning-AI/pytorch-lightning/blob/baeef935fb172a5aca2c84bff47b9b59d8e35b8a/src/lightning/fabric/utilities/seed.py#L37 
    
    def get_seed():
        ## https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
        RAND_SIZE = 4
        random_data = os.urandom( RAND_SIZE )  ## Return a string of size random bytes suitable for cryptographic use.
        return int.from_bytes(random_data, byteorder="big")


    if istrain:
        seed = cfg.get("seed")
        if seed == 0: ## failed to get from cfg
            seed = get_seed()
    else: 
        seed = parse_ptfn(cfg.TEST.LOADFROM)['seed']
        if seed == 0: ## failed to get from filename
            seed = cfg.get("seed")
            if seed == 0:
                seed = get_seed()
                log.warning(f'seed not present in net filename, using a newly generated')
            else: log.warning(f'seed not present in net filename, using cfg.seed')

    log.info(f'SEEDIT~WEEDIT {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) ## cpu
    torch.cuda.manual_seed(seed) ## g sing
    torch.cuda.manual_seed_all(seed) ## g mult
    
    ## https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        log.info(f"{torch.are_deterministic_algorithms_enabled()}")
        assert torch.are_deterministic_algorithms_enabled() == True

    assert torch.initial_seed() == int(os.environ['PYTHONHASHSEED'])

def seed_sade(worker_id):
    ## https://pytorch.org/docs/stable/notes/randomness.html
    wi = torch.utils.data.get_worker_info()
    log.warning(f'seed_sade {worker_id=} {wi}')
    
    tmp = torch.initial_seed() % 2**32
    np.random.seed(tmp)
    random.seed(tmp)       

def parse_ptfn(fn):
    match = re.match(r'(.*)_(\d+)\.(net|dict)$', fn)
    if not match: 
        log.warming(f'no match for pattern name_seed.[net|dict]')
        d = {'bn':None,'seed':None,'mode':None}
    else:
        base_name, seed, load_mode = match.groups()
        d = {'bn':base_name,'seed':int(seed),'mode':mode}
    log.info(f'{fn} parsed as {d}')
    return d

def collect_random_states() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch.cuda": torch.cuda.random.get_rng_state_all(),
    }



def set_max_threads(max_threads: int = 32) -> None:
    """Manually set max threads
    Threads set up for:
    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS
    Args:
        max_threads (int): Max threads value. Default to 32.
    """
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)


