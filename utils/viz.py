import visdom, json, numpy, subprocess

from .log import LoggerManager

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
        self.log = log = LoggerManager.get_logger(__name__)
        
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
        #self.log.info(up)
        if not up: 
            self.log.error("RUN :   nohup python -m visdom.server > vis.out &&"); 
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
                    self.log.warning(f"VIS closing {wtitle} {win_id} ")
        
    def exists(self, wtitle): return self.vis.win_exists(win=wtitle,env=self.env)
    
    def delete(self, envn=None, all=False):
        if all: ## except itself
            envs = self.vis.get_env_list()
            for env in envs:
                if self.env != env: self.vis.delete_env(env)
        elif envn: self.vis.delete_env(envn)