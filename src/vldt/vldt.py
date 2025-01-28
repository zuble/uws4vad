import torch
import os, os.path as osp, numpy as np, time, cv2, gc
from sklearn.metrics import average_precision_score

from src.vldt.metric import Metrics, Plotter
from src.data import get_testloader, run_dltest
from src.utils import get_log, hh_mm_ss

log = get_log(__name__)


################
## dict handlers
class WatchInfo:
    def __init__(self, cfg_ds, watching):
        self.lbl = cfg_ds.lbls.info[-1] ## 'ALL' normal and anom
        self.DATA = {self.lbl: {'FN': [], 'GT': [], 'FL': [], 'ATTWS': []}} if watching else {}
        self.watching = watching

    def updt(self, fn, gt, fl, attws=None):
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
                            self.anom: {'GT': [], 'FL': []}, ## auc / ap
                            self.all: {'GT': [], 'FL': []} }  ## auc / ap on fstep16, after upgrade 2330384
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
            ## flattens the list into numpy array cause of append in _updt_..
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
        #log.info(f'Validate w/ DL{self.DL}\n')
        
        self.cropasvideo = cfg.dataproc.cropasvideo.test
        self.ncrops = cfg.dataproc.crops2use.test
        
        ## selects the net forward, as attnomil needs splitting to get SegmLev scores
        ## exprmnt impact of such in different archs
        if cfg.vldt.fwd_siz is not None:
            self.fwd = self._fwd_attnomil
            self.chuk_size = cfg.vldt.fwd_siz
        else: self.fwd = self._fwd_glob
        
        
        self.metrics = Metrics(cfg_vldt, vis)
        self.gtfl = GTFL(cfg.data)
        
        if cfg.vldt.match_gtfl not in ['rshpndfill', 'truncate']:
            raise ValueError(f"match_gtfl {cfg.vldt.match_gtfl} not implemented")
        self.match_gtfl = cfg.vldt.match_gtfl
        self.multi_or_single = cfg.vldt.multi_or_single
        
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
        log.warning(f"TODO ret_att or ret_emb 4 visdom")
        #assert cfg.net.main._cfg.ret_att == self.ret_att
        
        ## after self.fwd() run, sls will have the SL scores acquired from chosen net forward
        ## attws will be populated if set, otherwise empty list is flag
        #sls, self.attws = [], [] #None, None
    
    def reset(self):
        ## reset the wtch/vldt_info.DATA
        ## so reruns of start method in train have empty lists up on the start
        ## so metrics reflect epo states
        self.vldt_info.reset()
        self.watch_info.reset()
        
    @torch.no_grad()
    def _fwd_attnomil(self, net, inferator, feat):
        '''
        as one score per forward, segment&repeat to get len(scores)=len(feats)
        #if self.cfg_frgb.fstep == 64: chuk_size = 8
        #elif self.cfg_frgb.fstep == 32: chuk_size = 16 
        #elif self.cfg_frgb.fstep == 16: chuk_size = 9 ## orignal
        '''
        ## segment-level scores
        sls, self.attws = [], []
        
        l = list(range(0,feat.shape[1]))
        splits = list(range(self.chuk_size, feat.shape[1], self.chuk_size))
        chucks = np.split( l, splits, axis=0)
        log.debug(f'l: {len(l)} , splits: {splits} , chucks: {chucks}')
        
        ## compare it might be faster
        #chucks = torch.split(feat, splits, axis=1)
        
        for ci, chuck in enumerate(chucks):
            ndata = net(feat[:,chuck])
            tmp_sls = inferator(ndata)
            log.debug(f'[{ci}] {chuck} {feat[:,chuck].shape} --> {tmp_sls}')
            
            sls.append( tmp_sls.repeat( len(chuck)) )
            
            if self.ret_att: ## ndata['attw'].shape (nsegments, 3)
                assert ndata.get('attw') is not None
                #tmp_attw = np.repeat( ndata['attw'], self.cfg_frgb.fstep, axis=0 )
                #log.debug(f'tmp_attw[{ci}] {tmp_attw.shape} {type(tmp_attw)}')
                self.attws.append( ndata['attw'].repeat( self.cfg_frgb.fstep, dim=0 ) )

        sls = torch.cat((sls), dim=0)
        
        if self.ret_att: 
            self.attws = torch.cat((self.attws), dim=0)
            log.debug(f"attws: {self.attws.shape} {self.attws.ctx}") 
        
        return sls
    
    @torch.no_grad()    
    def _fwd_glob(self, net, inferator, feat):
        ndata = net(feat)
        sls = inferator(ndata)
        if self.cropasvideo: ## nc, t  (og)
            assert sls.ndim == 2
            if self.ncrops:
                assert sls.shape[0] == self.ncrops
            sls = sls.mean(0)
        else: ## 1, t
            sls = sls.view(-1)
        return sls
            
    @torch.no_grad()    
    def start(self, net, inferator):
        net.eval()
        tic = time.time()
        _gt,_fl = [],[]
        
        log.info(f'$$$$ Validate starting')
        for i, data in enumerate(self.DL):
            
            feat=data[0][0].to(self.dvc); label=data[1]; fn=data[2][0]
            log.debug(f'[{i}] {"*"*4} {fn} , {label} ')
            
            if feat.ndim == 2: feat = feat.unsqueeze(0)  #log.warning("ndim2"); 
            elif feat.ndim == 3: pass ## 1,t,f no crop in ds
            elif feat.ndim == 4 and feat.shape[0] == 1: 
                ## 1 crop atleast, atm inferator.infer expects one dim=3
                feat = feat.view(-1, feat.shape[2], feat.shape[3]) ## 1*nc, t, f
            else: raise ValueError(f'[{i}] feat.ndim {feat.ndim}')
            log.debug(f'[{i}] {feat.shape}')
            
            sls = self.fwd(net, inferator, feat)
            log.debug(f'\t\t-> sls: {sls.shape}')
            
            #if '000' not in label[0]:
            #log.warning(f'[{i}] {feat.shape} , {fn} ') #, {label}
            #log.error(f'-> sls: {sls.shape} {sls.mean()} {sls.max()}')
            
            ##############
            ## frame-level
            ## base the length of GT/FL by number of frames in video 
            ##      -> reshp_nd_fill FL to match gt length
            ## base the length of GT/FL by length of scores (which are at segment level) * length of feat_ext wind
            ##      -> truncate the generated GT (which has video nframes length) to match the FL 
            tmp_gt = self.gtfl.get(fn)
            log.debug(f'[{i}] gt: {len(tmp_gt)}')
            ## 1 segmnt = self.cfg_frgb.fstep = (64 frames, slowfast mxnet) (32 frames, i3d mxnet) (16, i3dtorchdmil)
            tmp_fl = sls.cpu().detach().numpy()
            tmp_fl = np.repeat(tmp_fl, self.cfg_frgb.fstep)
            log.debug(f'[{i}] fl: {len(tmp_fl)} {tmp_fl.shape}') #{type(tmp_fl)} {type(tmp_fl[0])}
            
            #######
            ## as len of seqlen * self.cfg_frgb.fstep != number frames original video
            if len(tmp_fl) != len(tmp_gt):
                if len(tmp_gt) > len(tmp_fl):
                    ## fill tmp_fl w/ last value until len is the same as gt
                    if self.match_gtfl == 'rshpndfill':
                        new_fl = np.full(len(tmp_gt), tmp_fl[-1], dtype=tmp_fl.dtype)
                        new_fl[:tmp_fl.shape[0]] = tmp_fl
                        tmp_fl = new_fl
                        log.debug(f'[{i}] new_fl: {len(tmp_fl)}')
                    ## or truncate tmp_gt with len(tmp_fl), this is how RocNG does
                    elif self.match_gtfl == 'truncate':
                        tmp_gt = tmp_gt[:len(tmp_fl)]
                        log.debug(f'[{i}]   new_gt: {len(tmp_fl)}')
                else:
                    tmp_fl = tmp_fl[:len(tmp_gt)]
                    log.debug(f'[{i}]   new_fl: {len(tmp_gt)}')

            assert len(tmp_fl) == len(tmp_gt), f'{self.cfg_frgb.fstep} cfg_frgb.fstep * {sls.shape[0]} len  != {len(tmp_gt)} orign video frames'
            
            ## dirt but dont have time 
            ## dataloader puts list elements as tuples [('label1'),('label2')]
            label = [l[0] for l in label] ## ['label1','label2']
            if self.multi_or_single == 'single' and len(label) > 1: ## xdv and dflt
                label = [label[0]]  ## ['label1']
            elif self.multi_or_single == 'multi': pass
            
            self.vldt_info.updt(label, fn, tmp_gt, tmp_fl)
            self.watch_info.updt(fn, tmp_gt, tmp_fl, ) #self.attws
            
            _gt.append(tmp_gt); _fl.append(tmp_fl)
        
        _fl = np.concatenate(_fl, axis=0)
        _gt = np.concatenate(_gt, axis=0)
        app = average_precision_score(_gt, _fl)
        log.error(app)
        
        self.vldt_info.log()        
        self.vldt_info.upgrade()
        self.vldt_info.log()
        
        #self.watch_info.log()        
        self.watch_info.upgrade()
        #self.watch_info.log()

        mtrc_info, curv_info, table_res = self.metrics.get_fl(self.vldt_info)
        
        log.info(f'$$$$ VALIDATE @ {hh_mm_ss(time.time() - tic)}')
        ## outs are 4 -> test / test / train*2 (to save alongside the model state dict) / train to send 
        return self.vldt_info, self.watch_info.DATA, mtrc_info, curv_info, table_res


###########
class GTFL:
    '''
        Retrieves the Ground Truth Frame Level for the specified video
        either for the XDV or UCF dataset
        based on total frames to generate initial 0's sequence
        and from annotations sets 1's
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

