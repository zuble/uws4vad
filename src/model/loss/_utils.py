import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)


class LossUtils:
    """Provides utility functions for loss calculations."""

    def __init__(self, _cfg):
        self.k = _cfg.k  # Assuming k is a common parameter for feature selection
        self.do = nn.Dropout(0.7)  # Dropout could be a common component
    
#class Loss(nn.Module):
#    def __init__(self, _cfg):
#        super(Loss, self).__init__()
#    def plot_loss(self, LC , LN_1, LG_0, LG_1, LSmt, LSpr, L):
#        self.vis.plot_lines('LCls (LC)', LC.asnumpy())
#        self.vis.plot_lines('LNorm (LN_1)', LN_1.asnumpy())
#        self.vis.plot_lines('LGuia_0 (LG_0)', LG_0.asnumpy())
#        self.vis.plot_lines('LGuia_1 (LG_1)', LG_1.asnumpy())
#        self.vis.plot_lines('LSmth (LSmt)', LSmt.asnumpy())
#        self.vis.plot_lines('LSpars (LSpr)', LSpr.asnumpy())
#        self.vis.plot_lines('L (tmp)', L.asnumpy())
#    def forward(self):
#        l = 0.0
#        return {
#            'loss_': l
#            }
        
