import torch
import numpy as np
import math 
import cv2
#import av, decord ##order matters
#from gluoncv.utils.filesystem import try_import_decord
#decord = try_import_decord()
#import av ## otehrwise vis.video gives segment fault

from hydra.utils import instantiate as instantiate
import os, os.path as osp, numpy, time, tkinter as tk, sys, select

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt, matplotlib
matplotlib.use('TkAgg')

from src.model import ModelHandler, build_net
from src.vldt import Validate, Metrics
from src.utils import hh_mm_ss, get_log, Visualizer
log = get_log(__name__)


def test(cfg):
    
    cfg_vldt = cfg.vldt.test
    log.error(cfg_vldt)
    
    ## Monitor
    vis = Visualizer(f'{cfg.name}/{cfg.task_name}', restart=False, del_all=False)
    #vis.delete(f'{cfg.name}/{cfg.task_name}')
    
    ## shortcut to validate directly from a watch_info.pkl w/o Validate
    if cfg_vldt.frompkl:
        tic = time.time()
        log.info(f'$$$$ Validating from pkl')
        raise NotImplementedError
        #vldt_info = load_pkl(cfg.EXPERIMENTPATH, 'vldt')
        #log.info(f'$$ {vldt_info.per_what=}')
        #Metrics(cfg, 'test', vis).get_fl(vldt_info)
        #log.info(f'$$$$ vldt done in {hh_mm_ss(time.time() - tic)}')
        return
    
    ## shortcut to watch directly from a watch_info.pkl w/o Validate
    if cfg_vldt.watch.frompkl:
        tic = time.time()
        log.info(f'$$$$ Watching from pkl')
        raise NotImplementedError
        #watch_info = load_pkl(cfg.EXPERIMENTPATH,'watch')
        #Watch(cfg, watch_info, vis)
        #log.info(f'$$$$ watched 4 {hh_mm_ss(time.time() - tic)}')
        return
    
    tic = time.time()
    log.info(f'$$$$ TEST starting')

    ### !!
    ## make standart key troughout out of net with att
    if 'attws' in cfg.test.watch.frmt:
        raise NotImplementedError
        ## asserts that cfg is ready to watch attention_weights
        if 'attnomil' not in cfg.model.net.id:
            cfg_vldt.watch.frmt.remove('attws')
            log.warning(f'cfg_vldt.watch.ATTWS clear as the net in use ({cfg.NET.NAME}) isnt attnomil ')
        else: cfg.model.net.main._cfg.ret_att = True

    watching = [] #cfg_vldt.watch.frmt
    V = Validate(cfg, cfg_vldt, cfg.data.ds.frgb, vis, watching)
    if cfg.get("debug") and cfg.get("debug").get("vldt") > 1: vldt.start(net) #;return

    ## create network after Validate is constructed
    ## as testloader is created there
    ## so cfg features dim information is filled
    ## as it may be need to net class in use
    #net = get_ldnet(cfg, dvc)
    #net(np.random.uniform(size=(32,2048),ctx=dvc[0]))
    
    #vldt_info, watch_info, _ = V.start(net) ## class, dict, _

    #if cfg_vldt.savepkl: save_pkl(cfg.EXPERIMENTPATH, vldt_info, 'vldt')
    #if cfg_vldt.watch.savepkl: save_pkl(cfg.EXPERIMENTPATH, watch_info, 'watch')
    #if watching: Watch(cfg, watch_info, vis)

    log.info(f'$$$$ Test Completed in {hh_mm_ss(time.time() - tic)}')


class Watch:
    def __init__(self, cfg, watch_info, vis):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data.ds.info
        self.data = watch_info
        self.cfg_wtc = cfg_wtc = cfg_vldt.watch
        
        if not cfg_wtc.frmt: cfg_wtc.frmt = input(f"asp,gtfl,attws ? and/or comma separated").split(",")
        
        ## anomaly score player
        if 'asp' in cfg_wtc.frmt: self.asp = ASPlayer(cfg_wtc, vis).play
        else: self.asp = lambda *args, **kwargs: None
        log.info(f"Watch.asp {self.asp}")
        
        ## gtfl(grount-truth frame-level) ( /attws)
        if any(x in cfg_wtc.frmt for x in ['gtfl', 'attws']):
            self.plot = Plotter(cfg_wtc.frtend, vis, 'asp' in cfg_wtc.frmt).fx
        else: self.plot = lambda *args, **kwargs: None
        log.info(f"Watch.plot {self.plot}")
        
        self.init_gui() if cfg_wtc.GUIVIDSEL else self.init_lst()    

    
    def process(self, fn):
        ## common worker for both gui or lst
        idx = self.data['ALL']['FN'].index(fn)
        gt = self.data['ALL']['GT'][idx]
        fl = self.data['ALL']['FL'][idx]
        attw = self.data['ALL']['ATTWS'][idx]
        log.info(f'watch {fn} , gt {len(gt)} {type(gt)} | fl {len(fl)} {type(fl)} | attw {type(attw)} {len(attw)} ')
        
        self.plot(fn, gt, fl, attw) ## plot with full lenght of vframes
        vpath = osp.join(self.cfg_dsinf.vroot , fn)+'.mp4'
        stop = self.asp(vpath, gt, fl) ## play with frame_skip
        return stop
    
    def init_lst(self):
        if not self.cfg_wtc.label:
            self.cfg_wtc.label = input(f"1 or + labels from: {self.cfg_dsinf.lbls} 'label1,labeln' ").split(",")
        
        fnlist = FeaturePathListFinder(self.cfg, 'test', 'rgb', self.cfg_dsinf).get('watch', self.cfg_wtc.label)
        log.info(f'Watching lst init in {self.cfg_wtc.frmt} formats for {self.cfg_wtc.label} w/ {len(fnlist)} vids')
        for fn in fnlist: 
            stop = self.process(fn)
            if stop: log.warning(f"watch.init_lst just broke"); break
    
    
    def on_vid_sel(self, event):
        widget = event.widget
        index = int(widget.curselection()[0])
        fn = widget.get(index)
        self.process(fn)
        
    def init_gui(self):
        log.info(f"Watch is starting in GUI vid sel ")
        self.root = tk.Tk()
        self.root.title("VidSel")
        self.video_listbox = tk.Listbox(self.root)
        self.video_listbox.pack(side="left", fill="both", expand=True)
        fnlist = self.data['ALL']['FN']
        for fn in fnlist: self.video_listbox.insert(tk.END, fn)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical")
        self.scrollbar.config(command=self.video_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.video_listbox.config(yscrollcommand=self.scrollbar.set)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_vid_sel)
        self.root.mainloop()
    
    
#######################
## gtfl / attws plotter
class Plotter:
    def __init__(self, mode, vis, overwrite):
        #plt.ion()
        self.fx = {
            'wnshow': self.matplt,
            'visdom': self.plotly
        }.get(mode)        
        self.vis = vis
        ## if asp enabled, clear previous video related windows in visdom, otherwise accumulates
        self.overwrite = overwrite 
        self.cmap = 'viridis'

    def plotly(self, fn, gt, fl, attw):
        ## global
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('GT/FL', 'ATTWS'),
            vertical_spacing=0.1,  # Space between the subplots
            row_heights=[0.7, 0.3] ) # Relative heights of rows (heatmap is larger)
        
        nframes = gt.shape[0] 
        ## GT scores line plot to the second row
        gt_trace = go.Scatter(
            x=list(range(nframes)),
            y=gt,
            mode='lines',
            name='GT Scores',
            line=dict(color='red', dash='dash') )
        fig.add_trace(gt_trace, row=1, col=1)
        
        ## FL scores line plot to the second row
        fl_trace = go.Scatter(
            x=list(range(nframes)),
            y=fl,
            mode='lines',
            name='FL Scores',
            line=dict(color='blue', dash='dash') )
        fig.add_trace(fl_trace, row=1, col=1)

        fig.update_xaxes(title_text="Frames", row=1, col=1)
        fig.update_yaxes(title_text="Scores", row=1, col=1)
        
        if attw:
            ## heatmap for attention weights to the first row
            nattmaps = attw.shape[1]    
            heatmap = go.Heatmap(
                z=attw.transpose(),
                x=list(range(1, nframes + 1)),
                y=[f'#{i+1}' for i in range(nattmaps)],
                colorscale='Viridis')
            fig.add_trace(heatmap, row=2, col=1)
            fig.update_yaxes(title_text="Attention Map Number", row=2, col=1)
        
        if self.overwrite:
            title = f"Plotter"
            self.vis.close(title)
        else: title = f"Plotter - {fn}"
        
        ## layout for the whole figure
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=title)
        #fig.show()
        
        self.vis.potly(fig)
        
    def matplt(self, fn, gt, fl, attw):
        '''
        Plots the attention weights for each attention map across all frames.

        Parameters:
        - attw: A 2D array of shape (nframes, nattmaps), containing the attention weights for each frame.
        - overlay: If True, create a color map plot where each y value corresponds to an attention map.
        '''
        nframes = attw.shape[0]
        nattmaps = attw.shape[1]

        ## MATPLOT
        ## 1 fig w/ 2 sbplt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, nattmaps), sharex=True, gridspec_kw={'height_ratios': [nattmaps, 1]})
        # attention weights colormap in subplot 1
        attw = attw.transpose() ## (nattmaps, nframes)
        cax = ax1.imshow(attw, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_yticks(list(range(nattmaps)))
        ax1.set_yticklabels([f'#{i+1}' for i in range(nattmaps)])
        ax1.set_ylabel('Attention Map Number')
        ## Add colorbar for the attention weights
        cbar = fig.colorbar(cax, ax=ax1, orientation='vertical')
        cbar.set_label('Attention weights')
        ## GT/FL in subplot 2
        ax2.plot(gt, label='GT Scores', color='red', linestyle='--', linewidth=1)
        ax2.plot(fl, label='FL Scores', color='blue', linestyle='--', linewidth=1)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Scores')
        ## shared x-axis
        ax2.set_xticks([0, nframes - 1])
        ax2.set_xticklabels(['1', str(nframes)])
        
        '''
            fig, ax = plt.subplots(figsize=(12, nattmaps))
            
            cax = ax.imshow(attw, aspect='auto', cmap=self.cmap, interpolation='nearest')
            ax.set_xticks([0, nframes - 1])
            ax.set_xticklabels(['1', str(nframes)])
            ax.set_yticks(list(range(nattmaps)))
            ax.set_yticklabels([f'{self.title} #{i+1}' for i in range(nattmaps)])

            # Now scale the normalized scores to match the y-axis of the attention maps image
            # This is assuming you want to plot the scores across the entire height of the image.
            gt = gt * (nattmaps - 1)
            fl = fl * (nattmaps - 1)

            ax.plot(numpy.arange(nframes) , gt[:nframes], label='GT Scores', color='red', linestyle='--', linewidth=1)
            ax.plot(numpy.arange(nframes) , fl[:nframes], label='FL Scores', color='blue', linestyle='--', linewidth=1)
            
            cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
            cbar.set_label(self.ylabel)
            cbar.ax.set_ylabel('Attention weights')

            ax.set_xlabel('Frames')
            ax.set_ylabel('Attention Map Number')
            ax.legend(loc='upper right')
        '''

        plt.tight_layout()
        #plt.show(block=block)
        plt.close('all')
        plt.draw()
        plt.pause(3)
        

###############################
## cv windows viewer of results 
class ASPlayer:
    def __init__(self, cfg_wtc, vis):
        self.frtend = cfg_wtc.frtend
        self.frame_skip = cfg_wtc.asp.fs
        self.thrashold = cfg_wtc.asp.th
        self.cvputfx = {
            'float': self.cvputfloat,
            'color': self.cvputcolor
        }.get(cfg_wtc.asp.cvputfx)
        self.vis = vis

    def cvputfloat(self, frame, gt_fidx, fl_fidx):    
        cv2.putText(frame,'AS '+str('%.4f' % (fl_fidx)),(10,15),self.font,self.fontScale+0.2,[0,0,255],self.thickness,self.lineType)
        cv2.putText(frame,'GT '+str(gt_fidx), (10, 40), self.font,self.fontScale+0.2,[100,250,10],self.thickness,self.lineType)
        #new_time = time.time()
        #cv2.putText(frame, '%.2f' % (1/(new_time-tic))+' fps',(140,int(self.height)-10),self.font,self.fontScale,[0,50,200],self.thickness,self.lineType)
        #tic = new_time
    
    def cvputcolor(self, frame, gt_fidx, fl_fidx):
        
        def layer(frame, color, alpha):
            overlay = np.full_like(frame, color, dtype=np.uint8)
            return cv2.addWeighted(frame, 1, overlay, alpha, 0)
        
        center = frame.shape[1]//2
        g = (0, 255, 0) ## GREEN = NORMAL
        r = (0, 0, 255) ## RED = ABNORMAL
        
        ## LEFT SIDE = MACHINE
        cv2.putText(frame,str('%.3f' % (fl_fidx)),(100,15),self.font,self.fontScale+0.2,[0,0,255],self.thickness,self.lineType)
        if fl_fidx < self.thrashold: frame[:, :center] = layer(frame[:, :center], g, 0.5)
        else: frame[:, :center] = layer(frame[:, :center], r, 0.5)
    
        ## RIGTH SIDE = GT
        cv2.putText(frame,'GT '+str(gt_fidx), (int(self.width) - 100, 15), self.font,self.fontScale+0.2,[100,250,10],self.thickness,self.lineType)
        if gt_fidx == 0: frame[:, center:] = layer(frame[:, center:], g, 0.5)
        else: frame[:, center:] = layer(frame[:, center:], r, 0.5)    
    
    def init_cv(self, vpath):
        ## cv video info
        video = cv2.VideoCapture(vpath)
        self.total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = video.get(cv2.CAP_PROP_FPS)
        self.frame_time_ms = int(1000/self.fps)
        self.width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video.release()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5; self.thickness = 1; self.lineType = cv2.LINE_AA

    
    def overlay(self, vpath, gt, fl):
        ## generates gt/fl overlayed vid tensor
        dvr = decord.VideoReader(vpath)
        vframes = len(dvr)
        assert vframes == len(gt) == len(fl), f'{vframes = } , {len(gt) = } , {len(fl) = } '

        flist = list(range(0, vframes, self.frame_skip)) ##vframes+1
        log.debug(f"{flist.shape=} {self.frame_skip=}")
        vdata = dvr.get_batch(flist).asnumpy()
        #self.vis.vid(vdata, "vid"); return
        
        self.init_cv(vpath)    
        self.data = { "vpath": vpath, "frames": [], "gt": [], "fl": []}
        for i, fidx in enumerate(flist):
            #frame = dvr[fidx].asnumpy()
            frame = vdata[i]
            gt_fidx = gt[fidx]
            fl_fidx = fl[fidx]
            #log.debug(f"{i = } {fidx = } {gt_fidx = } {fl_fidx = } ")
            self.cvputfx(frame, gt_fidx, fl_fidx)  
            self.data["frames"].append(frame)
            self.data["gt"].append(gt_fidx)
            self.data["fl"].append(fl_fidx)
    
    def play(self, vpath, gt, fl):
        log.info(f"asp playing {vpath}")
        self.overlay(vpath, gt, fl)
        
        if self.frtend == 'wnshow':
            wn = 'as' + os.path.splitext(os.path.basename(vpath))[0]
            cv2.namedWindow(wn) 
            for frame in self.data["frames"]:
                cv2.imshow(wn, frame)
                key = cv2.waitKey(self.frame_time_ms)  
                if key == ord('q'): return False ## quit video
                if key == ord(' '):  ## pause
                    while True:
                        key = cv2.waitKey(1)
                        if key == ord(' '):break
                        if key == ord('q'):break ## quit pause
                        if key == ord('p'):return True ## quit asplayer
            time.sleep(1)
            cv2.destroyAllWindows()
            
        elif self.frtend == 'visdom':
            self.vis.close("vid")
            self.vis.vid(self.data["frames"], "vid")

            timeout = 3
            end_time = time.time() + timeout
            print(f"\n2 stop press Enter within {timeout} secs...")
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = input()
                    return True
                if time.time() > end_time: break

        return False



## simple vis gtfl plot
#if vis.exists(f'GT'): vis.close(f'GT')
#if vis.exists(f'FL'): vis.close(f'FL')
#vis.lines(f'GT', self.vgtfl, X=[f for f in range(0,len(self.vgtfl))])
#vis.lines(f'FL', self.vasfl, X=[f for f in range(0,len(self.vasfl))])

class RealTimePlot:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_data = []
        self.asfl_data = []
        self.gtfl_data = []

        self.asfl_line, = self.ax.plot(self.x_data, self.asfl_data, color='red', label='asfl')
        self.gtfl_line, = self.ax.plot(self.x_data, self.gtfl_data, color='blue', label='gtfl')

        self.ax.set_xlabel('Frame number')
        self.ax.set_ylabel('Value')
        self.ax.legend()

    def updt(self, x, asfl, gtfl):
        self.x_data.append(x)
        self.asfl_data.append(asfl)
        self.gtfl_data.append(gtfl)

        self.asfl_line.set_data(self.x_data, self.asfl_data)
        self.gtfl_line.set_data(self.x_data, self.gtfl_data)

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def go(self):
        plt.show(block=False)
        plt.pause(0.01)