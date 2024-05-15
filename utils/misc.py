import torch
import numpy as np #, cv2

import os, os.path as osp, glob, time, visdom, json, pickle, random, re, subprocess

import logging, matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from .log import LoggerManager

log = None
def init():
    ## set cfg.DEBUG.MISC
    global log
    log = LoggerManager.get_logger(__name__)


#########################
## SEED
def init_seed(cfg, istrain=True):
    ## https://pytorch.org/docs/1.8.1/notes/randomness.html
    ## https://github.com/henrryzh1/UR-DMU/blob/master/utils.py#L34
    ## https://github.com/Roc-Ng/DeepMIL/blob/master/main.py#L9
    ## https://github.com/Lightning-AI/pytorch-lightning/blob/baeef935fb172a5aca2c84bff47b9b59d8e35b8a/src/lightning/fabric/utilities/seed.py#L37 
    
    if istrain:
        seed = cfg.SEED
        if seed is None: ## failed to get from cfg
            seed = os.environ.get('PYTHONHASHSEED', None)
            if seed is None: ## check os.var ? resets everyrune, useless ?
                seed = get_truly_random_seed_through_os()
                log.info(f'new os.seed: {seed}')
            else: log.info(f'PYTHONHASHSEED: {seed}')
        else: log.info(f'cfg.SEED: {seed}')
                
    else: 
        seed = parse_ptfn(cfg.TEST.LOADFROM)['seed']
        if seed is None: ## failed to get from filename
            seed = cfg.SEED
            if seed is None:
                seed = get_truly_random_seed_through_os()
                log.warning(f'seed not present in net filename, using a newly generated')
            else: log.warning(f'seed not present in net filename, using cfg.SEED')

    log.info(f'SEEDIT~WEEDIT {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) ## cpu
    ## gpus
    torch.cuda.manual_seed(seed) ## single
    torch.cuda.manual_seed_all(seed) ## multiple
    
    ## https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    if cfg.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        log.info(f"{torch.are_deterministic_algorithms_enabled()}")
        assert torch.are_deterministic_algorithms_enabled() == True

    assert torch.initial_seed() == int(os.environ['PYTHONHASHSEED'])  
    cfg.merge_from_list(['SEED',seed])
    
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom( RAND_SIZE )  ## Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

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
        d = {'bn':None,'seed':None,'load_mode':None}
    else:
        base_name, seed, load_mode = match.groups()
        d = {'bn':base_name,'seed':int(seed),'load_mode':load_mode}
    log.info(f'{fn} parsed as {d}')
    return d







########################
## MP4
def mp4_rgb_info(vpath):
    if osp.exists(vpath):
        vv = cv2.VideoCapture(vpath)
        fps = vv.get(cv2.CAP_PROP_FPS)
        tframes = vv.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = vv.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        vv.release()
        cv2.destroyAllWindows()
        return dur,int(tframes),fps
    else: raise Exception(f"{vpath} no exist")
    

######################
## FEATURES
def view_feat(f):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, sharex='row', figsize=(5,4))
    ax.imshow(f, aspect='auto', interpolation='nearest')
    ax.set_title('cfg')
    plt.tight_layout()
    plt.show()
    
def comp_feat(ds,ds0):
    import matplotlib.pyplot as plt
    
    log.info(f'{len(ds.flist)} {len(ds0.flist)}')
    
    for i, ((f, l), (f0, l0)) in enumerate(zip(ds,ds0)):
        
        f = numpy.concatenate(f.asnumpy())
        f0 = numpy.concatenate(f0.asnumpy())
        log.info(f'[{i}]: {f.shape},{l}  {f0.shape},{l0}')
        
        fig, ax = plt.subplots(2, 1, sharex='row', figsize=(10,8))
        ax[0].imshow(f, aspect='auto', interpolation='nearest')
        ax[0].set_title('cfg')
        ax[1].imshow(f0, aspect='auto', interpolation='nearest')
        ax[1].set_title('original')
        plt.tight_layout()
        plt.show()
        
        if i == 5: break


######################
## vldt/watch_info .pkl i/o
def load_pkl(path,wut):
    if wut == 'watch':
        p = osp.join(path,'watch_info.pkl')
        if not osp.exists(p):
            log.error(f"there's none {p}")
            raise Exception(f"run once with cfg.TEST.WATCH.SAVEPKL: true / .FROMPKL: false")
        
        log.info(f"loading {p}")
        with open(p, 'rb') as f: data = pickle.load(f)
        return data
    
    elif wut == 'vldt':
        p = osp.join(path,'vldt_info.pkl')
        if not osp.exists(p):
            log.error(f"there's none {p}")
            raise Exception(f"run once with cfg.TEST.VLDT.SAVEPKL: true / .FROMPKL: false")
        
        log.info(f"loading {p}")
        with open(p, 'rb') as f: data = pickle.load(f)
        return data 
    
    else: raise Exception(f"wut must be watch or vldt")

def save_pkl(path,data,wut):
    if wut == 'watch':
        p = osp.join(path,'watch_info.pkl')
        log.info(f"saving {p}")
        with open(p, 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    elif wut == 'vldt':
        p = osp.join(path,'vldt_info.pkl')
        log.info(f"saving {p}")
        with open(p, 'wb') as f: pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    else: raise Exception(f"wut must be watch or vldt")

######################
def hh_mm_ss(seconds):
    #hours = seconds // 3600
    #minutes = (seconds % 3600) // 60
    #seconds = seconds % 60
    #return f"{hours:02}:{minutes:02}:{seconds:02}"
    return time.strftime('%H:%M:%S', time.gmtime(seconds))