import torch
import os, os.path as osp, numpy as np, time, cv2, gc

from src.vldt.metric import Metrics
from src.data import get_testloader, run_dl
from src.utils import get_log, hh_mm_ss

log = get_log(__name__)


################
## dict handlers
class WatchInfo:
    def __init__(self, cfg_ds, watching):
        self.lbl = cfg_ds.lbls_info[-1] ## 'ALL' normal and anom
        self.DATA = {self.lbl: {'FN': [], 'GT': [], 'FL': [], 'ATTWS': []}} if watching else {}
        self.watching = watching

    def updt(self, fn, gt, fl, attws):
        if not self.watching: return
        self.DATA[self.lbl]['FN'].append(fn)
        self.DATA[self.lbl]['GT'].append(gt)
        self.DATA[self.lbl]['FL'].append(fl)
        self.DATA[self.lbl]['ATTWS'].append(attws)
    
    def upgrade(self):
        if not self.watching: return
        #for lbl, metrics in self.DATA.items():
        #    metrics['GT'] = [ numpy.array(gt) for gt in metrics['GT'] ]
        #    metrics['FL'] = [ fl.asnumpy() for fl in metrics['FL'] ]
        #    if 'attws' in self.watching:
        #        metrics['ATTWS'] = [ attw.asnumpy() for attw in metrics['ATTWS'] ]

    def log(self):
        if not self.watching: return
        
        log.info("WatchInfo.DATA")
        for lbl, metrics in self.DATA.items():
            for key, _ in metrics.items():
                if self.DATA[lbl][key]:
                    log.info(f'[{lbl}][{key}]: {len(self.DATA[lbl][key])} {type(self.DATA[lbl][key])} {type(self.DATA[lbl][key][0])}')
            log.info("")
    
    def reset(self):
        if not self.watching: return
        self.DATA[self.lbl]['FN'] = []
        self.DATA[self.lbl]['GT'] = []
        self.DATA[self.lbl]['FL'] = []
        self.DATA[self.lbl]['ATTWS'] = []


class VldtInfo:
    def __init__(self, cfg_ds, per_what):
        self.per_what = per_what
        self.cfg_ds = cfg_ds
        self.norm = cfg_ds.lbls_info[0]
        self.anom = cfg_ds.lbls_info[-2]
        self.all = cfg_ds.lbls_info[-1]
        
        ## "metrics per_what prespective?"
        if per_what == 'glob':
            ## global: store normal and all abnormal
            #self.DATA = { self.norm: {'GT': [], 'FL': []}, self.anom: {'GT': [], 'FL': []} , self.all: {'GT': [], 'FL': []} }  
            self.DATA = { self.anom: {'GT': [], 'FL': []} , self.all: {'GT': [], 'FL': []} }  
            self.updt = self._updt_glob
        elif per_what == 'lbl':
            ## store previous plus specific labels of ds
            #self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.lbls_info}
            self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.lbls_info[1:]}
            self.updt = self._updt_lbl
        elif per_what == 'vid':
            ## store previous but keep a record for each video
            #self.DATA = {lbl: {'FN': [], 'GT': [], 'FL': []} for lbl in cfg_ds.lbls_info}
            self.DATA = {lbl: {'FN': [], 'GT': [], 'FL': []} for lbl in cfg_ds.lbls_info[1:]}
            self.updt = self._updt_vid 
        else: raise NotImplementedError
    
    def _updt_glob(self, label, fn, gt, fl):
        ## updates the global view of validation
        ## keeps track of anom only + total dataset
        if label[0] != self.norm:
            self.DATA[self.anom]['GT'].append(gt)
            self.DATA[self.anom]['FL'].append(fl)
        #else:
        #    self.DATA[self.norm]['GT'].append(gt)
        #    self.DATA[self.norm]['FL'].append(fl)
            
        self.DATA[self.all]['GT'].append(gt)
        self.DATA[self.all]['FL'].append(fl)
        
    def _updt_lbl(self, label, fn, gt, fl):
        ## updates anomalies subclasses with GT/FL
        ## keeps track of each anom sublabel
        if label[0] != self.norm:
            for lbl in label:
                self.DATA[lbl]['GT'].append(gt)
                self.DATA[lbl]['FL'].append(fl)
        self._updt_glob(label, fn, gt, fl)
        
    def _updt_vid(self, label, fn, gt, fl):
        ## updates fn in global classes and anom subclasses
        self._updt_lbl(label, fn, gt, fl)
        if label[0] != self.norm:
            self.DATA[self.anom]['FN'].append(fn)
            for lbl in label:
                self.DATA[lbl]['FN'].append(fn)
        #else: self.DATA[self.norm]['FN'].append(fn)       
        self.DATA[self.all]['FN'].append(fn)
        
    def upgrade(self):
        if self.per_what != 'vid': ## flattens the list into numpy array 
            
            for lbl, metrics in self.DATA.items():
                metrics['GT'] = np.concatenate((metrics['GT']), axis=0)
                metrics['FL'] = np.concatenate((metrics['FL']), axis=0)
                #metrics['FL'] = numpy.concatenate((metrics['FL']), axis=0)
                
        #else: ## assures every element in GT/FL is numpy
        ## done inside metric/Metrics/get_fl
        ## as prior to calculate metrics per vid 
        #    for lbl, metrics in self.DATA.items():
        #        metrics['GT'] = [ numpy.array(gt) for gt in metrics['GT'] ]
        #        metrics['FL'] = [ fl.asnumpy() for fl in metrics['FL'] ]

    def log(self):
        log.debug("VldtInfo.DATA")
        if self.per_what == 'vid':
            for lbl, metrics in self.DATA.items(): log.debug(f'[{lbl}][FN/GT/FL]: {len(self.DATA[lbl]["FN"])} {len(self.DATA[lbl]["GT"])} {len(self.DATA[lbl]["FL"])} ') #{type(self.DATA[lbl]["GT"])} {type(self.DATA[lbl]["FL"])}
        else: 
            for lbl, metrics in self.DATA.items(): log.debug(f'[{lbl}][GT/FL]: {len(self.DATA[lbl]["GT"])} {len(self.DATA[lbl]["FL"])} ')
    
    def reset(self):
        for lbl, metrics in self.DATA.items():
            for key, _ in metrics.items(): metrics[key] = []


###############
class Validate:
    def __init__(self, cfg, cfg_vldt, cfg_frgb, vis=None, watching=None):
        
        self.dvc = cfg.dvc
    
        self.cfg_ds = cfg_ds = cfg.data.ds.info
        self.cfg_frgb = cfg_frgb ## to get fstep used during fext
        self.cfg_vldt = cfg_vldt

        self.DL = get_testloader(cfg) 
        if cfg.get("debug") and cfg.get("debug").get("data") > 1: run_dl(self.DL) #;return
        log.info(f'Validate w/ DL{self.DL}\n')
        
        self.metrics = Metrics(cfg_vldt, vis)
        self.gtfl = GTFL(cfg_ds)
        
        ## selects the net forward, as attnomil needs splitting to get SegmLev scores
        ## or maybe test if results dffer when its not given the full context of video
        ## and make a global decision in vldt.train or test
        self.cfg_net = cfg.model.net
        self.fwd = {
            'attnomil': self._fwd_attnomil,
            'dflt': self._fwd_glob,
        }.get(cfg.model.net.id, self._fwd_glob)
        
        self.vldt_info = VldtInfo(cfg_ds, cfg_vldt.per_what)
        self.watch_info = WatchInfo(cfg_ds, watching)
        
        ## sets the ret_att only if we watching 'attws' 
        ## so net_out['attws'] are saved/returned in watch_info.DATA
        if watching: self.ret_att = 'attws' in watching
        else: self.ret_att = False
        
        ## after self.fwd() run, self.scores will have the SL scores acquired from chosen net forward
        self.scores, self.attws = [], [] #None, None
    
        
    def reset(self):
        ## reset the wtch/vldt_info.DATA
        ## so reruns of start method in train have empty lists up on the start
        ## as the metrics need the dict keys as arrays
        self.vldt_info.reset()
        self.watch_info.reset()
        
    @staticmethod
    def reshp_nd_fill(arr1, new_len):
        new_arr = np.full(new_len, arr1[-1], dtype=arr1.dtype)
        new_arr[:arr1.shape[0]] = arr1
        return new_arr
    
    
    #@torch.no_grad()
    def _fwd_attnomil(self, net, netpstfwd, feat):
        '''
        as one score per forward, segment&repeat to get len(scores)=len(feats)
        #if self.cfg_frgb.fstep == 64: split_size = 8
        #elif self.cfg_frgb.fstep == 32: split_size = 16 
        #elif self.cfg_frgb.fstep == 16: split_size = 9 ## orignal
        '''
        split_size = self.cfg_net.vldt_split
        
        self.scores, self.attws = [], []
        
        l = list(range(0,feat.shape[1]))
        splits = list(range(split_size, feat.shape[1], split_size))
        chucks = torch.split( l, splits, dim=0)
        log.debug(f'l: {len(l)} , splits: {splits} , chucks: {chucks}')
        
        ## compare it might be faster
        #chucks = torch.split(feat, splits, axis=1)
        
        for ci, chuck in enumerate(chucks):
            out = net(feat[:,chuck])
            
            self.scores.append( out["vlscores"].repeat( len(chuck), dim=0 ) )
            #log.debug(f'[{ci}] {chuck} {feat[:,chuck].shape} --> {out["scores"]}')
            
            if self.ret_att: ## out['attw'].shape (nsegments, 3)
                #tmp_attw = np.repeat( out['attw'], self.cfg_frgb.fstep, axis=0 )
                #log.debug(f'tmp_attw[{ci}] {tmp_attw.shape} {type(tmp_attw)}')
                self.attws.append( out['attw'].repeat( self.cfg_frgb.fstep, dim=0 ) )

        self.scores = torch.cat((self.scores), dim=0)
        log.debug(f'scores: {self.scores.shape} {self.scores.ctx}')
        
        if self.ret_att: 
            self.attws = torch.cat((self.attws), dim=0)
            log.debug(f"attws: {self.attws.shape} {self.attws.ctx}") 
    
    #@torch.no_grad()    
    def _fwd_glob(self, net, netpstfwd, feat):
        ndata = net(feat)
        
        #if self.cfg_net.id == 'zzz':
        #    if 'attw' in ndata:
        #        log.debug(f'slcores {ndata["slscores"]=}')
        #        log.debug(f'attw {ndata["attw"]=}')
        #        #self.scores = ndata['attw'] * ndata['slscores']
        #        self.scores = ndata['slscores']
        #    else:
        #        self.scores = ndata['slscores']
                
        self.scores = netpstfwd.infer(ndata)

        ## edge cases    
        log.debug(f'scores post0 {self.scores.shape=}')
        self.scores = self.scores.squeeze() #axis=1
        if self.scores.shape == (): ## == (1,1)
            self.scores = np.expand_dims(self.scores, axis=0)
        log.debug(f'scores post1 {self.scores.shape=}') #
    
    @torch.no_grad()    
    def start(self, net, netpstfwd):
        net.eval()
        
        gt_vl, scor_vl = [], []
        #gt_sl, scor_sl = [], []
        #gt_fl, scor_fl = [], []
        tic = time.time()
        
        log.info(f'$$$$ Validate starting')
        for i, data in enumerate(self.DL):
            log.debug(f'[{i}] ********************')

            feat=data[0][0].to(self.dvc); label=data[1][0]; fn=data[2][0]
            #feat = feat.unsqueeze(0)
            log.debug(f'[{i}] {feat.shape} , {fn} , {label}')
            self.fwd(net, netpstfwd, feat)
            ## self.scores is at segment level 
            
            ##############
            ## frame-level
            ## base the length of GT/FL by number of frames in video 
            ##      -> reshp_nd_fill FL to match gt length
            ## base the length of GT/FL by length of scores (which are at segment level) * length of feat_ext wind
            ##      -> truncate the generated GT (which has video nframes length) to match the FL 
            
            tmp_gt = self.gtfl.get(fn)
            log.debug(f' tmp_gt: {len(tmp_gt)}')
                
            ## 1 segmnt = self.cfg_frgb.fstep = (64 frames, slowfast mxnet) (32 frames, i3d mxnet) (16, i3dtorchdmil)
            tmp_fl = self.scores.cpu().numpy()
            tmp_fl = np.repeat(tmp_fl, self.cfg_frgb.fstep)
            log.debug(f' tmp_fl: {len(tmp_fl)} {tmp_fl.shape} {type(tmp_fl)} {type(tmp_fl[0])}')
            
            ########
            ## as len of seqlen * self.cfg_frgb.fstep != number frames original video
            ## fill tmp_fl w/ last value until len is the same as gt
            if len(tmp_fl) < len(tmp_gt):  tmp_fl = self.reshp_nd_fill(tmp_fl, len(tmp_gt))
            ## or truncate tmp_gt with len(tmp_fl), thhis is how RocNG/XDVioDet does
            else:
                min_len = min(len(tmp_fl), len(tmp_gt))
                tmp_fl = tmp_fl[:min_len]
                tmp_gt = tmp_gt[:min_len]
            ## clip feats can accomodate more "snippets" ??
            log.debug(f' tmp_fl_rshp: {len(tmp_fl)}')
            
            assert len(tmp_fl) == len(tmp_gt), f'{self.cfg_frgb.fstep} cfg_frgb.fstep * {self.scores.shape[0]} len  != {len(tmp_gt)} orign video frames'
            
            self.vldt_info.updt(label, fn, tmp_gt, tmp_fl)
            self.watch_info.updt(fn, tmp_gt, tmp_fl, self.attws)
        
        self.vldt_info.log()        
        self.vldt_info.upgrade()
        self.vldt_info.log()
        
        self.watch_info.log()        
        self.watch_info.upgrade()
        self.watch_info.log()

        mtrc_info = self.metrics.get_fl(self.vldt_info)
        log.error(mtrc_info)
        log.info(f'$$$$ VALIDATE completed in {hh_mm_ss(time.time() - tic)}')
        return self.vldt_info, self.watch_info.DATA, mtrc_info


###########
class GTFL:
    '''
        Retrieves the Ground Truth Frame Level for the specified video
        either for the XDV or UCF dataset
        based on total frames / annotations
    '''
    def __init__(self, cfg_ds):
        #log.debug(f'GTFL:\n{cfg_ds}')
        self.get_data(cfg_ds.gt, cfg_ds.tframes)
        
    def get_data(self, gt, tf):
        with open(gt, 'r') as txt: data = txt.read()
        self.gtlines = [line.split() for line in data.split('\n') if line]
        
        with open(tf, 'r') as txt: data = txt.read()
        self.tflines = [line.split('@') for line in data.split('\n') if line]
        
    def get(self, vn):
        nframes = next(( int(item[1]) for item in self.tflines if str(item[0]) == vn), None)    
        tmp_gt = [0 for _ in range(nframes)]
        
        vline = next((item for item in self.gtlines if str(item[0]) in vn), None)
        ## if in annotations -> abnormal
        if vline is not None:
            log.debug(f'GTFL.get({vn}) found {vline}')
            
            ## ucf: Burglary005_x264 Burglary 4710 5040 -1 -1
            if len(vline) > 1 and vline[1].isalpha(): 
                pairs = [(vline[i], vline[i+1]) for i in range(2, len(vline), 2) if vline[i+1] != '-1']
            
            ## xdv: v=wVey5JDRf_g__#00-04-09_00-05-06_label_B6-0-0 85 200 352 438 547 690 777 833 918 952 1055 1170 1314 1365
            else: pairs = zip(vline[1::2], vline[2::2])
            
            ## in the correspondant intervals fill with 1
            for anomi,(start, end) in enumerate(pairs):
                start_anom, end_anom = int(start), int(end)
                #log.debug(f'anomi[{anomi}]: [{start_anom}:{end_anom}]')
                for frame in range(start_anom, min(end_anom, nframes)):
                    tmp_gt[frame] = 1
        
        #else: full 0's     
            
        log.debug(f'GTFL.get({vn}) gt {len(tmp_gt)}')
        return np.array(tmp_gt)

