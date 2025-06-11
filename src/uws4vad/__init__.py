# Import core modules to trigger registration
from uws4vad.model import *  # Triggers model/net/__init__.py

#from uws4vad.common.registry import registry  # Ensure registry is accessible
#from uws4vad.model.net import *  # Ensure registration happens
#print("Registered networks uws4vad:", registry._registry["network"])


# Example: Expose key functions/classes at top level
#from .model.net._builder import build_net
#from .model.handler import ModelHandler
#from .utils.logger import get_log  # Example utility

## https://github.com/facebookresearch/mmf/blob/main/mmf/utils/env.py#L134
#from .utils.env import setup_imports 
#if __name__ == "__main__":
#    setup_imports()



