import os, os.path as osp, glob, sys, time, argparse, gc
import torch
print(f"{torch.__version__=}")
import numpy as np

import data, nets, tmp
from utils import setup_init, misc, viz
from vldt import vldt, metric

parser = argparse.ArgumentParser(description='fext')
parser.add_argument('--exp','-e',
                    type=str, 
                    default='cfg/weddit.yml',
                    help='either full or relative path to experiment .yml')
parser.add_argument('--loop_ptrn', '-lp',
                    type=str,
                    default='', ## cfg/attnomil_1*.yml
                    help='loop over all cfg present in loop_ptrn ')
args = parser.parse_args()


## init those w/ init fx
def initialize_modules():
    ## creates a log instance per module so based on cfg.DEBUG the log level is set
    ## init test only if enabled because of interactive mode and ssh con 
    mods = [data, metric, misc, nets, tmp, vldt , viz] 
    for mod in mods: mod.init()


def main():
    from utils.log import LoggerManager
    log = LoggerManager.get_logger(__name__)
    log.info(f"{dvc=} , torch acess to {torch.cuda.device_count()}")

        
    if cfg.TMP: 
        ## quick test stuff
        ## end of the setup here (logger,experiment cfg values, etc)
        tmp.main(cfg, dvc)
    
    else:
    
        if cfg.TRAIN.ENABLE:
            import train
            ## use cfg.SEED if set, else get one trough os
            misc.init_seed(cfg)
            train.init()
            loader = train.trainer(cfg, dvc)
            
            
            #opts = ["TRAIN.XEL.ENABLE", True]
            #cfg.merge_from_list(opts)
            #_ = train.trainer(cfg, dvc, loader)
            

        if cfg.TEST.ENABLE: 
            import test
            ## use seed present in .pt filename, else get one trough os
            misc.init_seed(cfg, False)
            test.init()
            test.test(cfg, dvc)


if __name__ == "__main__":

    ## tf32
    ## https://pytorch.org/docs/1.8.1/notes/cuda.html#tf32-on-ampere
    print(torch.tensor([1.2, 3]).dtype )
    print(f"{torch.backends.cuda.matmul.allow_tf32=} {torch.backends.cudnn.allow_tf32=}")
    print(f"{torch.get_default_dtype()=}")
    #torch.set_default_dtype('float32')
    
    ## expeeeriment config selection
    exps=[]
    if args.loop_ptrn: 
        exps = glob.glob( osp.join(os.getcwd(),args.loop_ptrn) )
    else: exps.append(args.exp)
    
    print('\n\n','_'*33,'\n\n')
    if not exps or not osp.exists(exps[0]) or not osp.exists(osp.join(os.getcwd(),exps[0])): 
        ## user input
        cfg_files = glob.glob(osp.join(os.getcwd(), 'cfg') + '/*.yml')
        cfg_dict = {str(i): t for i, t in enumerate(cfg_files)}
        for key, value in cfg_dict.items():print(f"{key}: {value}")
        sel_idx = input("Select 1 or + cfg to run, comma separated: ").split(",")
        exps = [cfg_dict[index.strip()] for index in sel_idx if index.strip() in cfg_dict]
    
    exps = sorted(exps)
    print(f'experiments: {exps}')
    
    for expi, exp in enumerate(exps):
        print(f'\n\nEXPERIMENT {expi+1}/{len(exps)}: {exp}\n\n')

        args.exp = exp
        cfg = setup_init(args) ## call log.LoggerManager.setup
        
        initialize_modules() ## create a log instance per module
        
        ## device setup
        if cfg.GPUID[0] != -1 and torch.cuda.is_available(): 
            dvc = torch.device(f'cuda:{cfg.GPUID[0]}')
            
        elif len(cfg.GPUID) == 1:
            dvc = torch.device('cpu')
            
        else: raise Exception(f'invalid gpuid {cfg.GPUID}')
        
        main()