import torch
import random
import numpy as np

import os, os.path as osp
import functools, signal, sys, importlib
from types import FrameType
from typing import Any, Dict, Callable, Optional

from uws4vad.utils.logger import get_log
log = get_log(__name__)


def init_seed(cfg, ckpt_path='', istrain=True):
    ## https://pytorch.org/docs/1.8.1/notes/randomness.html
    ## https://github.com/henrryzh1/UR-DMU/blob/master/utils.py#L34
    ## https://github.com/Lightning-AI/pytorch-lightning/blob/baeef935fb172a5aca2c84bff47b9b59d8e35b8a/src/lightning/fabric/utilities/seed.py#L37 
    
    def get_seed():
        ## https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
        RAND_SIZE = 4
        random_data = os.urandom( RAND_SIZE )
        return int.from_bytes(random_data, byteorder="big")


    if istrain:
        seed = cfg.get("seed")
        if seed == 0: ## failed to get from cfg
            seed = get_seed()
    else:
        if not ckpt_path: 
            if cfg.load.get("ckpt_path"): 
                ckpt_path = cfg.load.ckpt_path    
            else: raise Exception
        seed = int(osp.basename(ckpt_path).split("--")[0])
        if seed is None: ## failed to get from filename
            log.warning(f'seed not in ckpt_path filename !!')
            seed = cfg.get("seed")
            if seed == 0:
                seed = get_seed()
                log.warning(f'using a newly generated')
            else: log.warning(f'using cfg.seed')
        else: log.info(f'using ckpt_path filename seed :)')

    log.info(f'SEEDIT~WEEDIT {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    cfg.seed = seed
    
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
    log.debug(f'seed_sade {worker_id=} {wi}')
    
    tmp = torch.initial_seed() % 2**32
    np.random.seed(tmp)
    random.seed(tmp)       

def parse_ptfn(fn):
    match = re.match(r'(.*)--(\d+)\.(state|pt)$', fn)
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


## https://github.com/facebookresearch/mmf/blob/main/mmf/utils/env.py#L100
def import_files(file_path: str, module_name: str = None):
    """The function imports all of the files present in file_path's directory.
    This is useful for end user in case they want to easily import files without
    mentioning each of them in their __init__.py. module_name if specified
    is the full path to module under which all modules will be imported.

    my_project/
        my_models/
            my_model.py
            __init__.py

    Contents of __init__.py

    ```
    from mmf.utils.env import import_files

    import_files(__file__, "my_project.my_models")
    ```

    This will then allow you to import `my_project.my_models.my_model` anywhere.

    Args:
        file_path (str): Path to file in whose directory everything will be imported
        module_name (str): Module name if this file under some specified structure
    """
    for file in os.listdir(os.path.dirname(file_path)):
        if file.endswith(".py") and not file.startswith("_"):
            import_name = file[: file.find(".py")]
            if module_name:
                full_module = f"{module_name}.{import_name}"
            else:
                full_module = import_name
            try:
                importlib.import_module(full_module)
                #print(f"Imported {full_module}")  # Debug: Verify imports
            except Exception as e:
                print(f"Failed to import {full_module}: {e}")


class cleanup_on_exit:
    """Decorator to manage cleanup operations with Hydra compatibility"""
    _current = None  # Track active cleaner instance

    def __init__(self):
        self.cleanup_actions = []
        self.original_signal = None

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def hydra_compatible_wrapper(*args, **kwargs) -> Optional[Any]:
            # Set as current cleaner before Hydra executes
            cleanup_on_exit._current = self
            self.original_signal = signal.getsignal(signal.SIGINT)
            
            try:
                signal.signal(signal.SIGINT, self._handle_signal)
                result = func(*args, **kwargs)
            finally:
                self._execute_cleanup()
                signal.signal(signal.SIGINT, self.original_signal)
                cleanup_on_exit._current = None
                
            return result
            
        return hydra_compatible_wrapper

    @classmethod
    def get_cleaner(cls):
        """Get the active cleaner instance from any context"""
        return cls._current

    def _handle_signal(self, signum: int, frame: Optional[FrameType]) -> None:
        """Handle interrupt signals"""
        print(f"\n🛑 Received signal {signum}. Cleaning up...")
        self._execute_cleanup()
        sys.exit(1)

    def _execute_cleanup(self) -> None:
        """Execute all registered cleanup actions"""
        for action, args, kwargs in reversed(self.cleanup_actions):
            try:
                action(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Cleanup error: {str(e)}")

    def register(self, func: Callable, *args, **kwargs) -> None:
        """Register a cleanup action"""
        self.cleanup_actions.append((func, args, kwargs))