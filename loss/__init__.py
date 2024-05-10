import importlib
import pkgutil

from ._loss import get_loss

__all__ = []

MODULE_CONFIG = {
    'clas': True,
    'clawl': False,
    'cma_mil': False,
    'mbs': False,
    'rnkg': True,
    'tadl': False,
}

def import_modules():
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if MODULE_CONFIG.get(module_name, True):
            module = importlib.import_module(f".{module_name}", package=__name__)
            #print("LOSS",module)
            globals()[module_name] = module
            __all__.append(module_name)
            
import_modules()