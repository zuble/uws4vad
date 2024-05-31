import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_log
log = get_log(__name__)

class Loss(nn.Module):
    def __init__(self, _cfg):
        super(Loss, self).__init__()
        
    def forward(self):
        l = 0.0
        return {
            'loss_': l
            }