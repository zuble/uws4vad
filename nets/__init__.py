import importlib
import pkgutil

from ._nets import init, get_net, get_ldnet, save

__all__ = []

MODULE_CONFIG = {
    'attnomil': True,
    'cmala': False,
    'dtr': False,
    'mindspore': False,
    'rtfm': False,
    'zzz': False
}

def import_modules():
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name != "_nets" and MODULE_CONFIG.get(module_name, True):
            module = importlib.import_module(f".{module_name}", package=__name__)
            #print("NET",module)
            globals()[module_name] = module
            __all__.append(module_name)
            
import_modules()



