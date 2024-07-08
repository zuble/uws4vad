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
        self.lbl = cfg_ds.lbls.info[-1] ## 'ALL' normal and anom
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
                    log.debug(f'[{lbl}][{key}]: {len(self.DATA[lbl][key])} {type(self.DATA[lbl][key])} {type(self.DATA[lbl][key][0])}')
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
        self.norm = cfg_ds.lbls.info[0]
        self.anom = cfg_ds.lbls.info[-2]
        self.all = cfg_ds.lbls.info[-1]
        
        ## "metrics per_what prespective?"
        ## 000.NORM | B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM | ALL
        if per_what == 'glob':
            ## global: store normal and all abnormal
            ## 000.NORM | 111.ANOM | ALL
            self.DATA = {   self.norm:  {'GT': [], 'FL': []}, ## far
                            self.anom: {'GT': [], 'FL': []}, ## auc
                            self.all: {'GT': [], 'FL': []} }  ## auc
            self.updt = self._updt_glob
        elif per_what == 'lbl':
            ## store previous plus specific labels of ds
            ## B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM | ALL
            #self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.lbls.info[1:]}
            ## 000.NORM | B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM | ALL
            self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.lbls.info}
            self.updt = self._updt_lbl
        elif per_what == 'vid':
            ## store previous but keep a record for each video
            ## 000.NORM | B1.FIGHT | B2.SHOOT | B4.RIOT | B5.ABUSE | B6.CARACC | G.EXPLOS | 111.ANOM
            self.DATA = {lbl: {'FN': [], 'GT': [], 'FL': []} for lbl in cfg_ds.lbls.info[:-1]}
            self.updt = self._updt_vid 
        else: raise NotImplementedError
    
    def _updt_glob(self, label, fn, gt, fl):
        ## standart view
        ## keeps track of total video from anom only + total dataset
        if label[0] != self.norm:
            self.DATA[self.anom]['GT'].append(gt)
            self.DATA[self.anom]['FL'].append(fl)
        else:
            self.DATA[self.norm]['GT'].append(gt)
            self.DATA[self.norm]['FL'].append(fl)
        ## all ds view
        #if self.per_what != 'vid':
        self.DATA[self.all]['GT'].append(gt)
        self.DATA[self.all]['FL'].append(fl)
        
    def _updt_lbl(self, label, fn, gt, fl):
        ## keeps track of total videos from each anom sublabel
        if label[0] != self.norm:
            self.DATA[self.anom]['GT'].append(gt)
            self.DATA[self.anom]['FL'].append(fl)
            for lbl in label:
                self.DATA[lbl]['GT'].append(gt)
                self.DATA[lbl]['FL'].append(fl)
        else:
            self.DATA[self.norm]['GT'].append(gt)
            self.DATA[self.norm]['FL'].append(fl)
        
        if self.per_what != "vid":
            ## discard this when view is per vid
            self.DATA[self.all]['GT'].append(gt)
            self.DATA[self.all]['FL'].append(fl)
        
    def _updt_vid(self, label, fn, gt, fl):
        ## keeps each video clustered per lbl aswell total anom set
        ## by addyng fn key in each lbl dict
        ## call updt_lbl to updt gt,fl
        self._updt_lbl(label, fn, gt, fl)
        if label[0] != self.norm:
            self.DATA[self.anom]['FN'].append(fn)
            for lbl in label:
                self.DATA[lbl]['FN'].append(fn)
        else: 
            self.DATA[self.norm]['FN'].append(fn)
        
    def upgrade(self):
        if self.per_what != 'vid': 
            ## flattens the list into numpy array 
            for lbl, metrics in self.DATA.items():
                metrics['GT'] = np.concatenate((metrics['GT']), axis=0)
                metrics['FL'] = np.concatenate((metrics['FL']), axis=0)

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
    
        self.cfg_ds = cfg.data
        self.cfg_frgb = cfg_frgb ## to get fstep used during fext
        self.cfg_vldt = cfg_vldt

        self.DL = get_testloader(cfg) 
        #if cfg.dataloader.test.dryrun:
        #    log.warning(f"DBG DRY TEST DL")            
        #    run_dl(self.DL)
        #    return
        #log.info(f'Validate w/ DL{self.DL}\n')
        
        self.metrics = Metrics(cfg_vldt, vis)
        self.gtfl = GTFL(cfg.data)
        
        ## selects the net forward, as attnomil needs splitting to get SegmLev scores
        ## exprmnt impact of such in different archs
        if cfg.vldt.fwd_siz is not None:
            self.fwd = self._fwd_attnomil
            self.chuk_size = cfg.vldt.fwd_siz
        else: self.fwd = self._fwd_glob
        
        ## 
        self.vldt_info = VldtInfo(cfg.data, cfg_vldt.per_what)
        self.watch_info = WatchInfo(cfg.data, watching)
        
        ## sets the ret_att only if we watching 'attws' 
        ## so ndata['attws'] are saved/returned in watch_info.DATA
        ## can aswell modulate according presence and content of ndata["attws"]
        ## presence means network has that option
        ## None means ret_att is False
        if watching: self.ret_att = 'attws' in watching
        else: self.ret_att = False
        assert cfg.model.net.main._cfg.ret_att == self.ret_att
        
        ## after self.fwd() run, self.sls will have the SL scores acquired from chosen net forward
        ## attws will be populated is set, otherwise empty list are flag
        self.sls, self.attws = [], [] #None, None
    
    def reset(self):
        ## reset the wtch/vldt_info.DATA
        ## so reruns of start method in train have empty lists up on the start
        ## so metrics reflect epo states
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
        #if self.cfg_frgb.fstep == 64: chuk_size = 8
        #elif self.cfg_frgb.fstep == 32: chuk_size = 16 
        #elif self.cfg_frgb.fstep == 16: chuk_size = 9 ## orignal
        '''
        self.sls, self.attws = [], []
        
        l = list(range(0,feat.shape[1]))
        splits = list(range(self.chuk_size, feat.shape[1], self.chuk_size))
        chucks = np.split( l, splits, axis=0)
        #log.debug(f'l: {len(l)} , splits: {splits} , chucks: {chucks}')
        
        ## compare it might be faster
        #chucks = torch.split(feat, splits, axis=1)
        
        for ci, chuck in enumerate(chucks):
            ndata = net(feat[:,chuck])
            #log.debug(f'[{ci}] {chuck} {feat[:,chuck].shape} --> {type(ndata["vls"])}')
            
            self.sls.append( ndata["vls"].repeat( len(chuck)) )
            
            if self.ret_att: ## ndata['attw'].shape (nsegments, 3)
                assert ndata.get('attw') is not None
                #tmp_attw = np.repeat( ndata['attw'], self.cfg_frgb.fstep, axis=0 )
                #log.debug(f'tmp_attw[{ci}] {tmp_attw.shape} {type(tmp_attw)}')
                self.attws.append( ndata['attw'].repeat( self.cfg_frgb.fstep, dim=0 ) )

        self.sls = torch.cat((self.sls), dim=0)
        
        if self.ret_att: 
            self.attws = torch.cat((self.attws), dim=0)
            log.debug(f"attws: {self.attws.shape} {self.attws.ctx}") 
    
    #@torch.no_grad()    
    def _fwd_glob(self, net, netpstfwd, feat):
        ndata = net(feat)
        self.sls = netpstfwd.infer(ndata)
        #self.sls = self.sls.view(-1) ## !!!!! take care in infer with a super() on return
        
    
    @torch.no_grad()    
    def start(self, net, netpstfwd):
        net.eval()
        tic = time.time()
        
        log.info(f'$$$$ Validate starting')
        for i, data in enumerate(self.DL):
            #log.debug(f'[{i}] ********************')

            feat=data[0][0].to(self.dvc); label=data[1][0]; fn=data[2][0]
            #log.debug(f'[{i}] {feat.shape} , {fn} , {label}')
            if feat.ndim == 2: 
                ## ??? dl shouldnt put extra bat dim 
                feat = feat.unsqueeze(0) 
            elif feat.ndim == 3: pass ## no crop in ds
            elif feat.ndim == 4 and feat.shape[0] == 1: 
                ## 1 crop atleast, atm netpstfwd.infer isnot account this
                feat = feat.view(-1, feat.shape[2], feat.shape[3]) ## 1*nc, t, f
            else: raise ValueError(f'[{i}] feat.ndim {feat.ndim}')
            #log.debug(f'[{i}] {feat.shape} , {fn} , {label}')
            
            self.fwd(net, netpstfwd, feat)
            #log.debug(f'-> sls: {self.sls.shape}')
            ## self.sls is at segment level 
            
            
            #if '000' not in label[0]:
            #log.warning(f'[{i}] {feat.shape} , {fn} ') #, {label}
            #log.error(f'-> sls: {self.sls.shape} {self.sls.mean()} {self.sls.max()}')
            
            
            ##############
            ## frame-level
            ## base the length of GT/FL by number of frames in video 
            ##      -> reshp_nd_fill FL to match gt length
            ## base the length of GT/FL by length of scores (which are at segment level) * length of feat_ext wind
            ##      -> truncate the generated GT (which has video nframes length) to match the FL 
            
            tmp_gt = self.gtfl.get(fn)
            #log.debug(f' tmp_gt: {len(tmp_gt)}')
                
            ## 1 segmnt = self.cfg_frgb.fstep = (64 frames, slowfast mxnet) (32 frames, i3d mxnet) (16, i3dtorchdmil)
            tmp_fl = self.sls.cpu().detach().numpy()
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
            ## clip feats can accomodate more snippets, or just dif sample strat ??
            #log.debug(f' tmp_fl_rshp: {len(tmp_fl)}')
            
            assert len(tmp_fl) == len(tmp_gt), f'{self.cfg_frgb.fstep} cfg_frgb.fstep * {self.sls.shape[0]} len  != {len(tmp_gt)} orign video frames'
            
            self.vldt_info.updt(label, fn, tmp_gt, tmp_fl)
            self.watch_info.updt(fn, tmp_gt, tmp_fl, self.attws)
        
        self.vldt_info.log()        
        self.vldt_info.upgrade()
        self.vldt_info.log()
        
        #self.watch_info.log()        
        self.watch_info.upgrade()
        #self.watch_info.log()

        mtrc_info = self.metrics.get_fl(self.vldt_info)
        log.info(f'$$$$ VALIDATE @ {hh_mm_ss(time.time() - tic)}')
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
            #log.debug(f'GTFL.get({vn}) found {vline}')
            
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
            
        #log.debug(f'GTFL.get({vn}) gt {len(tmp_gt)}')
        return np.array(tmp_gt)

