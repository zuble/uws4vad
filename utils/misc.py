import torch
import numpy as np #, cv2

import os, os.path as osp, glob, time, visdom, json, pickle, random, re, subprocess
from collections import OrderedDict
import logging, matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from .log import LoggerManager

log = None
def init():
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

#########################
## visdom helper methods
class Visualizer(object):
    '''
        https://github.com/fossasia/visdom
        http://127.0.0.1:8097
        if in ssh add LocalForward 127.0.0.1:8097 127.0.0.1:8097 to .ssh/config
        
        prior to start a session of experiments run:
            nohup python -m visdom.server > vis.out
        if something wrong:
            ps aux | grep "visdom.server"
            kill -2 pid    (SIGINT)
            kill -15 pid    (SIGTERM)
            kill -9 pid     (SIGKILL)
    '''  
    def __init__(self, env, restart, del_all, **kwargs):
        self.check_server()
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.env = env
        self.index = {}
        self.mtrc_win = {}
        log.info(f"vis envs: {self.vis.get_env_list()}")
        log.info(f"vis created for {env = }")
        log.info(f'vis connection check {self.vis.check_connection()}')
        if del_all:
            log.info(f"vis delleting all envs")
            self.delete(all=True)
        if restart:
            log.info(f"vis restarted {env = }")
            self.close()
            
    def check_server(self):
        #up = subprocess.check_output("ps aux | grep 'visdom.server'| grep -v grep ", shell=True)
        up = subprocess.Popen("ps aux | grep 'visdom.server'| grep -v grep", shell=True, stdout=subprocess.PIPE).stdout.read().decode()
        #log.info(up)
        if not up: 
            log.error("RUN :   nohup python -m visdom.server > vis.out &&"); 
            raise Exception("")

        
    def plot_lines(self, name, y, **kwargs):
        #self.plot('loss', 1.00)
        
        x = self.index.get(name, 0)
        self.vis.line(Y=numpy.array([y]), X=numpy.array([x]),
                        win=str(name),
                        opts=dict(title=name),
                        update=None if x == 0 else 'append',
                        **kwargs)
        self.index[name] = x + 1

    def lines(self, name, line, X=None):
        if X is None: self.vis.line(Y=line, win=str(name), opts=dict(title=name))
        else: self.vis.line(X=X, Y=line, win=str(name), opts=dict(title=name))
            
    def scatter(self, xdata, ydata, opts, win=None, **kwargs):
        if not ydata: 
            #if win: self.vis.scatter(xdata,opts=opts,update='append',win=win)
            #else: return 
            self.vis.scatter(xdata,opts=opts,**kwargs)
        else: 
            #if win: self.vis.scatter(X=xdata,Y=ydata,opts=opts,update='append',win=win)
            #else: return 
            self.vis.scatter(X=xdata,Y=ydata,opts=opts,**kwargs)


    def plot_vldt_mtrc(self, data, epo):
        ## used only in train
        for metric in data[next(iter(data))].keys():
            if metric not in self.mtrc_win:
                self.vis.close(metric)##starts fresh
                opts = dict( xlabel='Epoch', 
                            ylabel=metric, 
                            title=metric, legend=[lbl for lbl in data.keys()] )
                self.mtrc_win[metric] = self.vis.line(
                    X=numpy.column_stack([numpy.array([epo]) for _ in data.keys()]),
                    Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys()]),
                    opts=opts )
            else:
                self.vis.line(
                    X=numpy.column_stack([numpy.array([epo]) for _ in data.keys()]),
                    Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys()]),
                    win=self.mtrc_win[metric],
                    update='append' )
    
    def bar(self, data, wtitle, opts):
        self.vis.bar( X=numpy.array([data]), win=wtitle, opts=opts )
    
    def potly(self, fig):
        self.vis.plotlyplot(fig)
    
    def disp_image(self, img, wtitle, opts):
        self.vis.image(img, win=wtitle, opts=opts)
    
    def vid(self, vid, wtitle):
        self.vis.video(vid, win=wtitle)
    
    def close(self, wtitle=None):
        ''' either a window or all windows off self.env '''
        if not wtitle:
            self.vis.close(win=wtitle, env=self.env)
            return
        
        tmp = json.loads(self.vis.get_window_data(env=self.env))
        for win_id, win_props in tmp.items():
            if win_props['title'] == wtitle: 
                
                if self.vis.win_exists(win=win_id,env=self.env):
                    self.vis.close(win=win_id, env=self.env) #, env=env
                    log.warning(f"VIS closing {wtitle} {win_id} ")
        
    def exists(self, wtitle): return self.vis.win_exists(win=wtitle,env=self.env)
    
    def delete(self, envn=None, all=False):
        if all: ## except itself
            envs = self.vis.get_env_list()
            for env in envs:
                if self.env != env: self.vis.delete_env(env)
        elif envn: self.vis.delete_env(envn)


############################
## PATHS AND SUCH
class FeaturePathListFinder:
    """
        From cfg_ds.FROOT/fname/ finds a folder with mode in it (train / test)
        then based on cfg procedes to filter the features paths
        so it retrieves accurate features list to use
        used in data/get_trainloader && data/get_testloader
    """
    def __init__(self, cfg, mode:str, modality:str, cfg_ds):
        self.listBG , self.listA = [], []
        
        #if not cfg_ds:
        #    if mode in 'train': cfg_ds = getattr(cfg.DS, cfg.TRAIN.DS)
        #    ## test mode is used by vldt/Validate on 2 scenarios:
        #    ## train for validation and test
        #    ## it must be specified in order to get the right cfg_ds
        #    ## as train and test might have different ds to operate on
        #    else: raise Exception("cfg_ds is None with mode in test")
        
        if modality == 'rgb': fname = cfg_ds.RGBFNAME
        elif modality == 'aud': 
            fname = cfg_ds.AUDFNAME
            if not fname :
                ## UCF
                log.warning(f'fname is empty for {mode}/aud while being enabled: feature lists empty')
                return
            
        fpath = ''
        for root, dirs, _ in os.walk(cfg_ds.FROOT):
            if fname == osp.basename(root):
                for d in dirs:
                    if mode in d:
                        #log.info(f'{d}')
                        fpath = osp.join(root, d)
                        log.info(f'{mode} features path {fpath}')
                        break        
        if not fpath: 
            log.error(f'{fname} or {mode} not found in {cfg_ds.FROOT}')

        flist = glob.glob(fpath + '/**.npy')
        flist.sort()
        log.debug(f"feat flist pre-filt in {mode} {modality} : {len(flist)}")
        
        #################        
        ## if cropasvideo
        ##      if train, 
        ##          if cfg.DATA.RGB.NCROPS == to the amount of crops in the flist 
        ##              all features will treatead as a video so leave all fn in list
        ##          if != load only correspondant crop, eg if cfg.DATA.RGB.NCROPS == 1 and the flist contains 5 crops for each vid, form flist with flist excluding the __1.npy __2.npy __3.npy __4.npy crops for each video 
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## elif cfg,DATA.NCROPS is set means that features files have crops even if only 1 is used
        ##      if train, take unique basename from crop fns -> feed each crop feat -> mean scores over crop dimension
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## else means that all files are a feature of video full view

        if modality == 'rgb':
            if cfg.DATA.CROPASVIDEO and mode in 'train':
                ## Get the unique video identifiers from the first cfg.DATA.RGB.NCROPS * 2 files
                unique_video_ids = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist[:cfg.DATA.RGB.NCROPS * 2]]))
                ## if features dataset ncrops corresponds to cfg.DATA.RGB.NCROPS select all
                if len(unique_video_ids) == 2: flist = [f[:-4] for f in flist]
                ## selects only the frist cfg.DATA.RGB.NCROPS crop files
                else:
                    flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                    flist = [f"{f}__{i}" for f in flist for i in range(cfg.DATA.RGB.NCROPS)]
            
            ## cfg.DATA.CROPASVIDEO is False or the mode is "test"
            elif cfg.DATA.RGB.NCROPS:
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
            
            ## ds w/ 1 rgbf per video
            else: flist = [f[:-4] for f in flist]
        
        ## 1 audf npy per video
        else: flist = [f[:-4] for f in flist]

        log.debug(f"feat flist pos-filt in {mode} {modality} : {len(flist)}")
        
        
        ######
        ## filters into anom and norm w/ listA and listBG
        ## filters for watching purposes w/ fn_label_dict
        self.fn_label_dict = {lbl: [] for lbl in cfg_ds.LBLS_INFO[:-1]}
        fist = list(self.fn_label_dict.keys())[0]  ## 000.NORM
        last = list(self.fn_label_dict.keys())[-1] ## 111.ANOM

        for fp in flist:
            fn = osp.basename(fp)
            
            ## v=Z12t5h2mBJc__#1_label_B1-0-0
            if 'label_' in fn: xearc = fn[fn.find('label_'):]
            else: xearc = fn
            
            #log.debug(f'{fp} {xearc} ')
            
            if cfg_ds.LBLS[0] in xearc:
                self.listA.append(fp)
                self.fn_label_dict[fist].append(fn)
                #log.debug(f'{xearc} {fn}')
            else:
                self.listBG.append(fp)
                self.fn_label_dict[last].append(fn)

                for key in self.fn_label_dict.keys():
                    if key in xearc: 
                        self.fn_label_dict[key].append(fn)
                        #log.debug(f'{key} {fn}')
                        
        #for label, lst in self.fn_label_dict.items(): 
        #    log.debug(f'[{label}]: {len(self.fn_label_dict[label])}  ') ##{self.fn_label_dict[label]}
                
                
    def get(self, mode, watch_list=[]):
        if mode == 'BG': l = self.listBG
        elif mode == 'A': l = self.listA
        elif mode == 'watch': 
            l = []
            for lbl2wtch in watch_list:
                ## find the labels that match the ones provided which need to be the nummberss prior to dot/.
                lbl2wtch = [lbl for lbl in list(self.fn_label_dict.keys()) if lbl.split('.')[0] == lbl2wtch][0]
                log.debug(f'{lbl2wtch} {len(self.fn_label_dict[lbl2wtch])}')
                l.extend(self.fn_label_dict[lbl2wtch])
        else: log.error(f'{mode} not found')
        return l


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