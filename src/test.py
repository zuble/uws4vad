import torch
import numpy as np
import math 

import cv2
import av ## prior 2 decord, or vis.video gives seg fault
import decord
#decord.bridge.set_bridge('torch')

from dataclasses import dataclass
from typing import Optional, Any
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as instantiate
import os, os.path as osp, time, glob, copy, pickle
import tkinter as tk, sys, select

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib 
#matplotlib.use('TkAgg') # Qt5Agg
import matplotlib.pyplot as plt

from src.model import ModelHandler, build_net
from src.vldt import Validate, Metrics, Plotter, Tabler
from src.utils import hh_mm_ss, get_log, Visualizer, init_seed
log = get_log(__name__)



def find_hydra_cfg(ckpt_path):
    """Search upwards for .hydra/config.yaml"""
    path = osp.dirname(ckpt_path)
    while path != '/':
        hydra_dir = osp.join(path, '.hydra')
        cfg_path = osp.join(hydra_dir, 'config.yaml')
        if osp.exists(cfg_path): return cfg_path
        path = osp.dirname(path)
    raise FileNotFoundError(f"No hydra config found for {ckpt_path}")
    
def merge_cfgs(cur_cfg, og_cfg):
    """Preserve validation parameters but allow override of original network config"""
    merged = cur_cfg.copy()
    merged.net = og_cfg.net
    return merged

def vldt_cfg(cur_cfg, og_cfg):
    log.info(f"{cur_cfg.data.frgb}  {og_cfg.data.frgb}")
    assert cur_cfg.data.frgb.id == og_cfg.data.frgb.id
    assert cur_cfg.data.frgb.dfeat == og_cfg.data.frgb.dfeat



@dataclass
class VldtRslt:
    ckpt_path: str
    vldt_info: dict
    wtch_info: dict
    mtrc_info: Optional[dict] = None
    curv_info: Optional[dict] = None
    tables: Optional[list] = None

@dataclass
class VldtPckg:
    rslts: list[VldtRslt]

######################
## .val.pkl i/o
def load_pkl(ckpt_path):
    p = f"{ckpt_path}.val.pkl"
    if not osp.exists(p):
        log.error(f"none {p} -> run once with cfg.vldt.test.savepkl:true AND .frompkl:'' ")
        return None

    with open(p, 'rb') as f: data = pickle.load(f)
            
    return VldtRslt(
        ckpt_path=ckpt_path,
        vldt_info=data['vldt_info'],
        wtch_info=data['wtch_info']
    )

def save_pkl(vldtrslt):
    p = f"{vldtrslt.ckpt_path}.val.pkl"
    if osp.exists(p): 
        inp = input(f".val.pkl already present {p}\noverwrite? (y/n)")
        if inp != 'y': return
        
    with open(p, 'wb') as f:
        pickle.dump({
            'vldt_info': vldtrslt.vldt_info,
            'wtch_info': vldtrslt.wtch_info
        }, f, pickle.HIGHEST_PROTOCOL)
    log.info(f"saved vldt_info at {p}")


def tester(cfg, vis):
    tic = time.time()
    log.info(f'$$$$ TEST starting')
    
    pckg = VldtPckg([])
    cfg_vldt = cfg.vldt.test
    watching = cfg_vldt.watch.frmt
    log.info(f"{watching=}")
    
    MH = ModelHandler(cfg)
    VLDT = Validate(cfg, cfg_vldt, cfg.data.frgb, vis=vis, watching=watching)
    MTRC = Metrics(cfg_vldt, vis)
    log.error(MH.ckpt_path)
    for ckpt_path in MH.ckpt_path:
        
        ## only load when not comming from train and frompkl is set
        rslt = load_pkl(ckpt_path)
        if rslt and cfg_vldt.frompkl and not cfg.train:
            if cfg_vldt.per_what == rslt.vldt_info.per_what: 
                rslt.mtrc_info, rslt.curv_info, rslt.tables = MTRC.get_fl(rslt.vldt_info)
                pckg.rslts.append(rslt)
                log.info(f"loaded from val.pkl {ckpt_path}")
                continue
            log.warning(f"Not using loaded .val.pkl\n{cfg_vldt.per_what=} {rslt.vldt_info.per_what=}")
        rslt = VldtRslt(ckpt_path,{},{})
        
        if not cfg.train:  ## set seed / cfg.net / vldt cfg.data
            init_seed(cfg, ckpt_path, False)
            
            hydra_cfg_path = find_hydra_cfg(ckpt_path)  
            og_cfg = OmegaConf.load(hydra_cfg_path)  
            ## check for mismatch in data/frgb as diff features need diff dataloader
            vldt_cfg(cfg, og_cfg)
            ## preserve running vldt cfg / override net cfg
            cfg = merge_cfgs(cfg, og_cfg)

        net, inferator = build_net(cfg)
        net.to(cfg.dvc)
        
        ##########
        if 'attws' in watching[:]:
            raise NotImplementedError
            ## from dflt all net.main._cfg has ret_att set to false
            ## as long as the dyn_retatt is in use, if attws in frmt -> ret_att set to true
            ## even if the net doesnt has that option is handled in Validate
            assert cfg.model.net.main._cfg.ret_att == True, log.error(f"{cfg.model.net.id} w ret_att False ???")
            #raise NotImplementedError
            log.warning(f'watch attws from {cfg.model.net.id}')
        ##########
        
        if cfg.vldt.dryrun:
            log.info("DBG DRY VLDT RUN")
            vi,_ = VLDT.start(net, inferator); VLDT.reset() #;return
            _,_,ts = MTRC.get_fl(vi)
            for t in ts: log.info(f'\n{t}')
            continue
        
        net.load_state_dict( MH.get_test_state(ckpt_path) )
        rslt.vldt_info, rslt.wtch_info = VLDT.start(net, inferator)
        if cfg_vldt.savepkl: save_pkl(rslt) ## save before mtrc, as its based on cfg
        
        rslt.mtrc_info, rslt.curv_info, rslt.tables = MTRC.get_fl(rslt.vldt_info)
        pckg.rslts.append(rslt)
        
        ## clean
        VLDT.reset(); MTRC.reset()
        del net
        torch.cuda.empty_cache()

    del VLDT ## free DL

    ## Reporting
    if cfg_vldt.per_what == 'lbl' and pckg.rslts:
        pltr = Plotter(vis)
        for rslt in pckg.rslts:
            log.info(f"Results for {rslt.ckpt_path}")
            pltr.metrics(rslt.mtrc_info)
            pltr.curves(rslt.curv_info)
            for table in rslt.tables:
                log.info(f'\n{table}')
                
    elif cfg_vldt.per_what == 'vid':
        T = Tabler(cfg_vldt)
        
        if len(pckg.rslts) == 1:
            for table in pckg.rslts[0].tables: log.info(f'\n{table}')
            
        #for rslt in pckg.rslts:
            #log.info(f"Results for {rslt.ckpt_path}")
            #MTRC.datler.proc_by_label(rslt.vldt_info.DATA)
            #table = MTRC.tabler.log_per_lbl( MTRC.datler.mtrc_info )
            #log.error(table)
            #T.log_per_lbl(rslt.mtrc_info)
            #for table in rslt.tables: log.info(f'\n{table}')
        
        if len(pckg.rslts) > 1:
            T.log_per_vid_per_ckpt(pckg.rslts)
            T.log_per_lbl_per_ckpt(pckg.rslts)
            
    if watching:
        Watcher(
            cfg, cfg_vldt.watch,
            pckg.rslts,
            vis
        )
        
    log.info(f'$$$$ Test Completed in {hh_mm_ss(time.time() - tic)}')
    sys.exit()
    
    
    
class Watcher:
    def __init__(self, cfg, cfg_wtc, rslts, vis):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data
        self.cfg_wtc = cfg_wtc 
        self.rslts = rslts
        
        if not cfg_wtc.frmt: cfg_wtc.frmt = input(f"asp,gtfl,attws ? and/or comma separated").split(",")
        
        ## anomaly score player
        if 'asp' in cfg_wtc.frmt: self.asp = ASPlayer(cfg_wtc, vis).play
        else: self.asp = lambda *args, **kwargs: None
        log.info(f"Watcher.asp {self.asp}")
        
        ## gtfl(grount-truth frame-level) ( /attws) viewer
        if any(x in cfg_wtc.frmt for x in ['gtfl', 'attws']):
            self.plot = ASPlotter(cfg_wtc.frtend, vis, 'asp' in cfg_wtc.frmt).fx
        else: self.plot = lambda *args, **kwargs: None
        log.info(f"Watcher.plot {self.plot}")
        
        log.info(f"Watcher initialized with {len(self.rslts)} models")
        self.init_gui() if cfg_wtc.guividsel else self.init_lst()    
        
    def process(self, fn):
        ## common worker for both gui or lst
        idx = self.rslts[0].wtch_info['ALL']['FN'].index(fn)
        gt = self.rslts[0].wtch_info['ALL']['GT'][idx]
        sms=f'Watching {fn}\n gt {len(gt)} {type(gt).__name__}'
        
        tmp = [{
            'ckpt': rslt.ckpt_path,
            'fl': rslt.wtch_info['ALL']['FL'][idx],
            'attw': rslt.wtch_info['ALL']['ATTWS'][idx]
        } for rslt in self.rslts]

        sms = [
            f"Watching {fn}",
            f"GT: {len(gt)} frames ({type(gt).__name__})"
        ]
        for i, d in enumerate(tmp):
            sms.append(
                f"  Model {i+1} {d['ckpt']}:\n"
                f"  - FL: {len(d['fl'])} frames ({type(d['fl']).__name__})\n"
                f"  - ATTW: {type(d['attw']).__name__ if d['attw'] is not None else 'None'}"
            )
        log.info('\n'.join(sms))
        self.plot(fn, gt, tmp)
        
        vpath = osp.join(self.cfg_dsinf.vroot, f"TEST/{fn}.mp4")
        if len(tmp) > 1:
            log.warning("Multiple ckpts detected")
            log.info(f"{[m['ckpt'] for m in tmp]}")
        stop = self.asp(vpath, gt, tmp) ##[m['fl'] for m in tmp]
        return stop
    
    def init_lst(self):
        if not self.cfg_wtc.label:
            
            self.cfg_wtc.label = input(f"1 or + labels from: {self.cfg_dsinf.lbls} 'label1,labeln' ").split(",")
            
            fnlist = FeaturePathListFinder(self.cfg, 'test', 'rgb').get('watch', self.cfg_wtc.label)
            log.info(f'Watching lst init in {self.cfg_wtc.frmt} formats for {self.cfg_wtc.label} w/ {len(fnlist)} vids')
            for fn in fnlist: 
                stop = self.process(fn.replace(" ",""))
                if stop: log.warning(f"watch.init_lst just broke"); return
            
        else:            
            ## Continuously process filenames input by the user
            while True:
                fns = input("1 or + fns 'fn1,fn2' (double-click+(ctrl+v)) | empty enter -> stop: ")
                if fns.strip() == '': return
                
                fnlist = fns.split(',')
                for fn in fnlist:
                    stop = self.process(fn)
                    if stop: log.warning(f"watch.init_lst just broke"); return
    
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
        fnlist = self.rslts[0].wtch_info['ALL']['FN']
        for fn in fnlist: self.video_listbox.insert(tk.END, fn)
        self.scrollbar = tk.Scrollbar(self.root, orient="vertical")
        self.scrollbar.config(command=self.video_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.video_listbox.config(yscrollcommand=self.scrollbar.set)
        self.video_listbox.bind('<<ListboxSelect>>', self.on_vid_sel)
        self.root.mainloop()
    
    
#######################
## gtfl / attws plotter
class ASPlotter:
    def __init__(self, mode, vis, overwrite):
        #plt.ion()
        self.fx = {
            'wnshow': self.matplt,
            'visdom': self.plotly
        }.get(mode)
        self.vis = vis
        self.overwrite = overwrite
        self.cmap = 'viridis'
        self.colors = [
            'blue', 
            'green', 
            'orange', 
            'purple'
        ]

    def _short_name(self, path):
        """Extract meaningful short name from checkpoint path"""
        return path.split('/')[-1].split('.')[0]
        #return int(osp.basename(path).split("--")[0])

    def plotly(self, fn, gt, data):
        """Plotly version supporting multiple FL curves"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('GT/FL', 'ATTWS'),
            vertical_spacing=0.1, # Space between the subplots
            row_heights=[0.7, 0.3]  # Relative heights of rows (heatmap is larger)
        )
        # fig = make_subplots(
        #     rows=2 if attw_list else 1, cols=1,
        #     subplot_titles=('Score Comparison', 'Attention Weights') if attw_list else None,
        #     vertical_spacing=0.1
        # )
        
        ## GT
        nframes = len(gt)
        fig.update_layout(yaxis_range = [0,1])
        #fig.update_layout(yaxis2_range=[0,1]) 
        fig.add_trace(
            go.Scatter(
                x=list(range(nframes)),
                y=gt,
                mode='lines',
                name='GT',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        ## each FL curve with checkpoint info
        for idx, d in enumerate(data):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(d['fl']))),
                    y=d['fl'],
                    mode='lines',
                    name=f'FL {self._short_name(d["ckpt"])}',
                    line=dict(color=self.colors[idx % len(self.colors)], width=1)
                ), 
                row=1, col=1
            )
        
        ## attention weights (if available)
        if any(d['attw'] is not None for d in data):
            for i, d in enumerate(data):
                if d['attw'] is not None:
                    nattmaps = d['attw'].shape[1]  
                    fig.add_trace(
                        go.Heatmap(
                            z=d['attw'].T,
                            x=list(range(nframes)),
                            y=[f'{self._short_name(d["ckpt"])} Map {j+1}' for j in range(nattmaps)],
                            colorscale='Viridis',
                            showscale=False,
                            name=f'ATTW {self._short_name(d["ckpt"])}'
                        ),
                        row=2, col=1
                    )
        
        if self.overwrite:
            title = f"Plotter"
            self.vis.close(title)
        else: title = f"Plotter - {fn}"
        
        ## layout for the whole figure
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text=title
        )
        #fig.show()
        
        self.vis.potly(fig)
        
        
    def matplt(self, fn, gt, data):
        '''
        Plots the attention weights for each attention map across all frames, with multiple models.

        Parameters:
        - attw: A 2D array of shape (nframes, nattmaps), containing the attention weights for each frame.
        - overlay: If True, create a color map plot where each y value corresponds to an attention map.
        '''     
        
        nframes = len(gt)
        
        ## Create figure
        if any(d['attw'] is not None for d in data):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        else: ## Only plot GT/FL scores if no attention weights provided
            fig, ax2 = plt.subplots(figsize=(12, 4))
        
        ## Plot GT and FL
        ax2.plot(gt, label='GT', color='red', linestyle='--', linewidth=1)
        for i, d in enumerate(data):
            ax2.plot(d['fl'], 
                label=f'Model {i+1} FL', 
                color=self.colors[i % len(self.colors)], 
                linestyle='--'
            )
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Scores')
        ax2.set_xticks([0, nframes - 1])
        ax2.set_xticklabels(['1', str(nframes)])
        
        # Plot attention weights (if available)
        if any(d['attw'] is not None for d in data):
            for i, d in enumerate(data):
                if d['attw'] is not None:
                    cax = ax1.imshow(d['attw'].T, aspect='auto', cmap=self.cmap, alpha=0.7)
                    ax1.set_yticks(range(d['attw'].shape[1]))
                    ax1.set_yticklabels([f'Model {i+1} Map {j+1}' for j in range(d["ckpt"].shape[1])])
                    ax1.set_ylabel('Attention Maps')
        
        plt.ion()
        plt.tight_layout()
        plt.show()  #block=block

        '''
        #if len(attw): 
        if attw is not None and len(attw) > 0 and len(attw[0]) > 0:
            
            assert nframes == attw.shape[0]
            nattmaps = attw.shape[1]
            
            ## MATPLOT
            ## 1 fig w/ 2 sbplt
            #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, nattmaps), sharex=True, gridspec_kw={'height_ratios': [nattmaps, 1]})
            
            ## Plot attention weights in subplot 1
            #attw = attw.transpose() ## (nattmaps, nframes)
            cax = ax1.imshow(attw.T, aspect='auto', cmap=self.cmap, interpolation='nearest')
            ax1.set_yticks(list(range(nattmaps)))
            ax1.set_yticklabels([f'#{i+1}' for i in range(nattmaps)])
            ax1.set_ylabel('Attention Map Number')

            ## Add colorbar for the attention weights
            cbar = fig.colorbar(cax, ax=ax1, orientation='vertical')
            cbar.set_label('Attention weights')
        else:
            
            fig, ax2 = plt.subplots(figsize=(12, 4))

        # Plot GT and FL scores in subplot 2 or the only subplot
        ax2.plot(gt, label='GT Scores', color='red', linestyle='--', linewidth=1)
        ax2.plot(fl, label='FL Scores', color='blue', linestyle='--', linewidth=1)
        ax2.legend(loc='upper right')
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Scores')
        ax2.set_xticks([0, nframes - 1])
        ax2.set_xticklabels(['1', str(nframes)])

        plt.ion()
        plt.tight_layout()
        plt.show()  #block=block
        
        
        #plt.tight_layout()
        #plt.close('all')
        #plt.draw()
        #plt.pause(3)

        '''
            # fig, ax = plt.subplots(figsize=(12, nattmaps))
            
            # cax = ax.imshow(attw, aspect='auto', cmap=self.cmap, interpolation='nearest')
            # ax.set_xticks([0, nframes - 1])
            # ax.set_xticklabels(['1', str(nframes)])
            # ax.set_yticks(list(range(nattmaps)))
            # ax.set_yticklabels([f'{self.title} #{i+1}' for i in range(nattmaps)])

            # # Now scale the normalized scores to match the y-axis of the attention maps image
            # # This is assuming you want to plot the scores across the entire height of the image.
            # gt = gt * (nattmaps - 1)
            # fl = fl * (nattmaps - 1)

            # ax.plot(numpy.arange(nframes) , gt[:nframes], label='GT Scores', color='red', linestyle='--', linewidth=1)
            # ax.plot(numpy.arange(nframes) , fl[:nframes], label='FL Scores', color='blue', linestyle='--', linewidth=1)
            
            # cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
            # cbar.set_label(self.ylabel)
            # cbar.ax.set_ylabel('Attention weights')

            # ax.set_xlabel('Frames')
            # ax.set_ylabel('Attention Map Number')
            # ax.legend(loc='upper right')
        
        

###############################
## cv windows viewer of rslts 
class ASPlayer:
    def __init__(self, cfg_wtc, vis):
        self.frtend = cfg_wtc.frtend
        self.frame_skip = cfg_wtc.asp.fs
        self.thrashold = cfg_wtc.asp.th
        self.cvputfx = {
            'float': self._cvputfloat,
            'color': self._cvputcolor
        }.get(cfg_wtc.asp.cvputfx)
        self.vis = vis
        self.colors = [
            (0, 200, 100),  # Teal
            (0, 150, 150),  # Cyan
            (0, 100, 200),  # Blue-green
            (0, 50, 255)    # Blue
        ]

    def _cvputfloat(self, frame, gt_fidx, fls):
        ## GT at bottom
        cv2.putText(frame, f'GT {gt_fidx}', (10, int(self.height)-10), 
                    self.font, self.fontScale+0.2, (100,250,10), 
                    self.thickness, self.lineType)
        
        ## each model's FL with descending offset
        y_start = 15
        for i, fl in enumerate(reversed(fls)):
            y = y_start + (25 * i)
            color = self.colors[i % len(self.colors)]
            cv2.putText(frame, f'M{len(fls)-i}: {fl:.4f}', (10, y),
                    self.font, self.fontScale+0.2, color,
                    self.thickness, self.lineType)
        #cv2.putText(frame,'AS '+str('%.4f' % (fl_fidx)),(10,15),self.font,self.fontScale+0.2,[0,0,255],self.thickness,self.lineType)
        
        #new_time = time.time()
        #cv2.putText(frame, '%.2f' % (1/(new_time-tic))+' fps',(140,int(self.height)-10),self.font,self.fontScale,[0,50,200],self.thickness,self.lineType)
        #tic = new_time
    
    def _cvputcolor(self, frame, gt_fidx, fls):
        
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
    
    def _init_cv(self, vpath):
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

    
    def overlay(self, vpath, gt, fls):
        ## generates gt/fl overlayed vid tensor
        dvr = decord.VideoReader(vpath)
        vframes = len(dvr)
        
        for fl in fls:
            assert vframes == len(fl), f'{vframes} != {len(fl)}'
            assert vframes == len(gt) == len(fl), f'{vframes = } , {len(gt) = } , {len(fl) = } ' ## when match_gtfl == truncate, this fails

        flist = list(range(0, vframes, self.frame_skip)) ##vframes+1
        log.debug(f"{len(flist)=} {self.frame_skip=}")
        vdata = dvr.get_batch(flist).asnumpy()
        #self.vis.vid(vdata, "vid"); return
        
        self._init_cv(vpath)    
        self.data = { "vpath": vpath, "frames": [], "gt": [], "fl": []}
        for i, fidx in enumerate(flist):
            #frame = dvr[fidx].asnumpy()
            frame = vdata[i]
            gt_fidx = gt[fidx]
            fl_fidx = [fl[fidx] for fl in fls] #fl[fidx]
            #log.debug(f"{i = } {fidx = } {gt_fidx = } {fl_fidx = } ")
            self.cvputfx(frame, gt_fidx, fl_fidx)  
            self.data["frames"].append(frame)
            self.data["gt"].append(gt_fidx)
            self.data["fl"].append(fl_fidx)
    
    def play(self, vpath, gt, wi):
        log.info(f"asp playing {vpath}")
        fls = [w['fl'] for w in wi]
        self.overlay(vpath, gt, fls)
        
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
            #log.error(self.data['frames'])
            ## TODO: assert C order  LxHxWxC.
            ## https://github.com/fossasia/visdom/issues/210
            #log.error(f"{type(data['frames'])}")
            #log.error(f"{self.data['frames'].shape}")
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
        

## v=eqtJjxsTgtg__#1_label_G-0-0.mp4
## v=UK--hvgP2uY__#1_label_G-0-0.mp4 --smoking sets anom high