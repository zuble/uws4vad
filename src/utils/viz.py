import visdom, json, numpy, subprocess, time, sys
import os, os.path as osp
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.utils.logger import get_log
log = get_log(__name__)


helper = """
    https://github.com/fossasia/visdom 
    prior to start a session of experiments run under main.py dir:
        nohup python -m visdom.server > vis.out &
    
    visdom server will be avaiable at
        http://127.0.0.1:8097
    
    if in ssh add to .ssh/config
        LocalForward 127.0.0.1:8097 127.0.0.1:8097
    
    if something wrong:
        ps aux | grep 'visdom.server'
        kill -? pid    ?:(2SIGINT) (15 SIGTERM) (9 SIGKILL)
            or 
        ps -ef | grep 'python -m visdom.server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 
"""
            
#########################
## visdom helper methods
class Visualizer:

    def __init__(self, cfg, **kwargs):
        self.check_server()
        self.nameit(cfg)
        self.vis = visdom.Visdom(env=self.env_name, **kwargs)
        self.index = {}
        self.mtrc_win = {}
        
        log.debug(f"vis envs: \n\n{self.vis.get_env_list()}\n")
        log.info(f"vis created for {self.env_name = }")
        log.debug(f'vis connection check {self.vis.check_connection()}')
        
        if cfg.xtra.vis.delete: 
            self.delete(needle=cfg.xtra.vis.delete)
            self.delete(envn=self.env_name)
        if cfg.xtra.vis.restart:
            log.info(f"vis restarted {self.env_name = }");self.close()
        
    def nameit(self, cfg):
        ## ----------------------------------
        ## configure vis based on hydra params
        #vis=None
        #if not cfg.get("debug", False):
        self.env_name = f"{cfg.name}_{cfg.task_name}"
        if cfg.exp_name: self.env_name += f"__{cfg.exp_name}"
        if 'MULTIRUN' in str(HydraConfig.get().mode):
            path_parts = cfg.path.out_dir.split("/")
            self.env_name += f"__{path_parts[-2]}__{path_parts[-1]}" ## date_jobnum
        else: 
            self.env_name += f"___{osp.basename(cfg.path.out_dir)}"
        #log.info(f"{self.env_name=}")
        ## ------------------------------

    def check_server(self):
        #up = subprocess.check_output("ps aux | grep 'visdom.server'| grep -v grep ", shell=True)
        up = subprocess.Popen("ps aux | grep 'visdom.server'| grep -v grep", shell=True, stdout=subprocess.PIPE).stdout.read().decode()
        #log.info(up)
        if not up: log.error(helper); sys.exit() 
        
        
    def plot_lines(self, wname, y, **kwargs):
        #self.plot('loss', 1.00)
        x = self.index.get(wname, 0)
        self.vis.line(Y=numpy.array([y]), X=numpy.array([x]),
                        win=str(wname),
                        #opts=dict(title=wname),
                        update=None if x == 0 else 'append',
                        **kwargs)
        self.index[wname] = x + 1

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

    def textit(self, data):
        if isinstance(data, (str, int, float)): ## JSON-serializable
            self.vis.text(data)
        elif isinstance(data, DictConfig): # DictConfig
            self.vis.text(json.dumps(OmegaConf.to_yaml(data, resolve=True), indent=4))
        elif isinstance(data, list) and len(data) % 2 == 0:  # List of name-value pairs
            text_string = ""
            for i in range(0, len(data), 2):
                name = data[i]
                value = data[i + 1]
                # Convert value to JSON-serializable format
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, indent=4)
                elif isinstance(value, DictConfig):
                    value = OmegaConf.to_yaml(value, resolve=True)
                    if name == 'choices':
                        parts = value.split(" ")
                        value = ":<br> ".join(parts)
                else:
                    value = str(value)
                text_string += f"{name}:<br>    {value}<br><br>"
            #log.debug(text_string)
            self.vis.text(text_string)
        elif isinstance(data, (dict, list, tuple)): ## convert to JSON string
            self.vis.text(json.dumps(data, indent=4))
        else:
            try: self.vis.text(str(data))
            except TypeError: raise TypeError("Data is not JSON serializable and cannot be converted to a string.")
    
    def bar(self, data, wtitle, opts):
        self.vis.bar( X=numpy.array([data]), win=wtitle, opts=opts )
    
    def potly(self, fig):
        self.vis.plotlyplot(fig)
    
    def disp_image(self, img, wtitle, opts):
        self.vis.image(img, win=wtitle, opts=opts)
    
    def vid(self, vid, wtitle, opts):
        self.vis.video(vid, win=wtitle, opts=opts)
    
    
    def close(self, wtitle=None):
        ''' either a window or all windows off self.env_name '''
        #if not wtitle:
        if wtitle is None:
            self.vis.close(win=wtitle, env=self.env_name)
            return
        
        tmp = json.loads(self.vis.get_window_data(env=self.env_name))
        for win_id, win_props in tmp.items():
            if win_props['title'] == wtitle: 
                
                if self.vis.win_exists(win=win_id,env=self.env_name):
                    self.vis.close(win=win_id, env=self.env_name) #, env=env
                    log.warning(f"VIS closing {wtitle} {win_id} ")
        
    def exists(self, wtitle): return self.vis.win_exists(win=wtitle,env=self.env_name)
    
    def delete(self, envn=None, needle=''):
        ''' Delete envn if provided, else finds envs with needle in its name
            'all' and others strings expect 'debug' ask for confirmation
            a delay of 7 seconds in the end for cancel time '''
        if envn:
            if envn == self.env_name: confirmation = input(f"delete running {envn=}?  (y to continue) ")
            else: confirmation = input(f"delete {envn=}?  (y to continue) ")
            if confirmation.lower() == 'y':
                log.warning(f"Deleting {envn}")
                visdom.Visdom(env=envn).delete_env(envn)
                ## if envn to del match current env, exit program
                if envn == self.env_name: sys.exit(1)
                
            else:log.info("all safe.")
            
        elif needle:
            envs_name = self.vis.get_env_list()
            if envs_name:
                if needle == 'all':
                    tobedel = []
                    for env_name in envs_name:
                        if self.env_name != env_name: tobedel.append(env_name)
                    if not tobedel: log.info("nothing to be deleted"); confirm=False
                    else:
                        confirmation = input(f"\ndelete AAALLL envs ( \n {tobedel} \n ) except {self.env_name=}?  (y to continue)")
                        if confirmation.lower() == 'y': confirm=True
                        else: confirm=False
                else:
                    tobedel = []
                    for env_name in envs_name: 
                        if self.env_name != env_name and needle in env_name: tobedel.append(env_name)
                    
                    if not tobedel: log.info("nothing to be deleted"); confirm=False
                    elif needle == 'debug': confirm=True
                    else:
                        confirmation = input(f"delete envs with {needle} : (  {tobedel}  ) except {self.env_name=}?  (y to continue)")
                        if confirmation.lower() == 'y': confirm=True
                        else: confirm=False
                
                if confirm and tobedel:
                    log.warning(f"Deleting \n{tobedel}\n in 7 seconds still time to ctrl+c")
                    time.sleep(7)
                    for env_name in tobedel:
                        log.warning(f"Deleting {env_name}")
                        visdom.Visdom(env=env_name).delete_env(env_name)
            else: 
                log.info("nothing to be deleted")            
        
        log.info(f"vis envs: \n\n{self.vis.get_env_list()}\n")
        
    '''
    def plot_vldt_mtrc(self, data, epo):
        ## used only in train
        for lbl in data.keys():  # Iterate over labels
            for metric in data[lbl].keys():  # Iterate over metrics for each label
                if metric not in self.mtrc_win:
                    self.vis.close(metric)
                    opts = dict(
                        xlabel='Epoch',
                        ylabel=metric,
                        title=metric,
                        legend=[lbl for lbl in data.keys() if metric in data[lbl]]  # Include only relevant labels
                    )
                    # Handle single-value metrics (like FAR)
                    if len(data[lbl][metric]) == 1:  
                        self.mtrc_win[metric] = self.vis.line(
                            X=numpy.array([epo]),
                            Y=numpy.array([data[lbl][metric][0]]),
                            opts=opts
                        )
                    else:  # Handle multi-value metrics
                        self.mtrc_win[metric] = self.vis.line(
                            X=numpy.column_stack([numpy.array([epo]) for _ in data.keys() if metric in data[lbl]]),
                            Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys() if metric in data[lbl]]),
                            opts=opts
                        )
                else:
                    # Handle single-value metrics (like FAR)
                    if len(data[lbl][metric]) == 1:
                        self.vis.line(
                            X=numpy.array([epo]),
                            Y=numpy.array([data[lbl][metric][0]]),
                            win=self.mtrc_win[metric],
                            update='append'
                        )
                    else:  # Handle multi-value metrics
                        self.vis.line(
                            X=numpy.column_stack([numpy.array([epo]) for _ in data.keys() if metric in data[lbl]]),
                            Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys() if metric in data[lbl]]),
                            win=self.mtrc_win[metric],
                            update='append'
                        )
            
        #for metric in data[next(iter(data))].keys():
        #    if metric not in self.mtrc_win:
        #        self.vis.close(metric)##starts fresh
        #        opts = dict( xlabel='Epoch', 
        #                    ylabel=metric, 
        #                    title=metric, legend=[lbl for lbl in data.keys()] )
        #        self.mtrc_win[metric] = self.vis.line(
        #            X=numpy.column_stack([numpy.array([epo]) for _ in data.keys()]),
        #            Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys()]),
        #            opts=opts )
        #    else:
        #        self.vis.line(
        #            X=numpy.column_stack([numpy.array([epo]) for _ in data.keys()]),
        #            Y=numpy.column_stack([numpy.array([data[lbl][metric][-1]]) for lbl in data.keys()]),
        #            win=self.mtrc_win[metric],
        #            update='append' )'''
