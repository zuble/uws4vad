import os, os.path as osp, subprocess

from cfg.default import get_cfg


def init(args):
    """
    Given the arguemnts, load cfg and initialize experiment dir struture
    Args:
        args (argument):
    """
    
    ## define the full path of experiment .yml file
    if osp.isfile(args.exp): expryml_path = args.exp
    else:
        if args.exp.startswith("/"): args.exp=args.exp[1:]
        expryml_path = osp.join(os.getcwd() , args.exp)
        #print(f'1 {expryml_path}')
        if not (osp.isfile(expryml_path)):
            expryml_path = osp.join(os.getcwd() , "cfg/"+args.exp)
            #print(f'2 {expryml_path}')
    assert osp.isfile(expryml_path), f'{expryml_path} does not exist'
    
    #print(f'3 {expryml_path}')

    ## merge default with experiment
    cfg = get_cfg(expryml_path) 

    ## Creates experiment main dir 
    cfg.EXPERIMENTPROJ = osp.basename(os.getcwd())
    cfg.EXPERIMENTID = osp.splitext(osp.basename(expryml_path))[0]
    exp_path = osp.join(cfg.EXPERIMENTDIR,cfg.EXPERIMENTID)
    if not osp.exists(exp_path):os.mkdir(exp_path)    
    cfg.EXPERIMENTPATH = exp_path
    
    ## sets all gpus as avaiable
    gpus = subprocess.check_output(['nvidia-smi'], text=True).count('%')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, range(gpus))) ## '0,...,gpus-1'
    
    from utils.log import LoggerManager
    log_path = LoggerManager.setup(exp_path, cfg.DEBUG)
    
    #cfg.freeze()
    logger = LoggerManager.get_logger(__name__)
    logger.info('\n*******\n ********\n33344333\n ********\n******')
    if cfg.LOG_CFG_INFO: 
        
        logger.info('cfg:\n{}'.format(cfg.dump()))
        #logger.info(f"{cfg=}")
        logger.info(f'log file @ {log_path}')
    
    return cfg