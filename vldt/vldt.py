import torch
import os, os.path as osp, numpy, time, cv2, gc

from data import get_testloader
from utils import LoggerManager, hh_mm_ss


log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)


################
## dict handlers
class WatchInfo:
    def __init__(self, cfg_ds, watching):
        self.lbl = cfg_ds.LBLS_INFO[-1] ## 'ALL' normal and anom
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
        for lbl, metrics in self.DATA.items():
            metrics['GT'] = [ numpy.array(gt) for gt in metrics['GT'] ]
            metrics['FL'] = [ fl.asnumpy() for fl in metrics['FL'] ]
            if 'attws' in self.watching:
                metrics['ATTWS'] = [ attw.asnumpy() for attw in metrics['ATTWS'] ]

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
        self.norm = cfg_ds.LBLS_INFO[0]
        self.anom = cfg_ds.LBLS_INFO[-2]
        self.all = cfg_ds.LBLS_INFO[-1]
        
        ## "metrics per_what prespective?"
        if per_what == 'glob':
            ## global: store normal and all abnormal
            #self.DATA = { self.norm: {'GT': [], 'FL': []}, self.anom: {'GT': [], 'FL': []} , self.all: {'GT': [], 'FL': []} }  
            self.DATA = { self.anom: {'GT': [], 'FL': []} , self.all: {'GT': [], 'FL': []} }  
            self.updt = self._updt_glob
        elif per_what == 'lbl':
            ## store previous plus specific labels of ds
            #self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.LBLS_INFO}
            self.DATA = {lbl: {'GT': [], 'FL': []} for lbl in cfg_ds.LBLS_INFO[1:]}
            self.updt = self._updt_lbl
        elif per_what == 'vid':
            ## store previous but keep a record for each video
            #self.DATA = {lbl: {'FN': [], 'GT': [], 'FL': []} for lbl in cfg_ds.LBLS_INFO}
            self.DATA = {lbl: {'FN': [], 'GT': [], 'FL': []} for lbl in cfg_ds.LBLS_INFO[1:]}
            self.updt = self._updt_vid 
        else: raise NotImplementedError
    
    def _updt_glob(self, label, fn, gt, fl):
        ## updates the global view of validation
        ## the NORMAL/frist key and ABNORMAL/last key lists in dict
        ## GT/FL is viewed for ALL anomalies
        if label[0] != self.norm:
            self.DATA[self.anom]['GT'].append(gt)
            self.DATA[self.anom]['FL'].append(fl)
        #else:#if label[0] != self.norm:
        #    self.DATA[self.norm]['GT'].append(gt)
        #    self.DATA[self.norm]['FL'].append(fl)
            
        self.DATA[self.all]['GT'].append(gt)
        self.DATA[self.all]['FL'].append(fl)
        
    def _updt_lbl(self, label, fn, gt, fl):
        ## updates anomalies subclasses with GT/FL
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
                metrics['GT'] = numpy.concatenate((metrics['GT']), axis=0)
                metrics['FL'] = torch.cat((metrics['FL']), axis=0).numpy()
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
    def __init__(self, cfg, cfg_ds, cfg_vldt, net_pst_fwd, dvc, metrics, watching):
        self.cfg = cfg
        self.dvc = dvc
        self.metrics = metrics
        
        self.cfg_ds = cfg_ds
        self.cfg_vldt = cfg_vldt
        log.info(f'Validate w/\n\ncfg_ds:\n{cfg_ds}\n\ncfg_vldt:\n{cfg_vldt}\n')
        
        self.DL = get_testloader(cfg, cfg_ds)
        log.info(f'Validate w/ DL{self.DL}\n')
        
        self.gtfl = GTFL(cfg_ds)
        
        ## selects the net forward, as attnomil needs splitting to get SegmLev scores
        if 'ATTNOMIL' in cfg.NET.NAME: self.forward = self._forward_attnomil 
        elif 'TAD' in cfg.NET.NAME: raise NotImplementedError #self.forward = self._forward_tad
        else: 
            self.net_pst_fwd = net_pst_fwd
            ## here its assumed that net_pst_fwd isnt None
            ## as the input format 
            self.forward = self._forward_glob
        log.info(f'Validate w/ forward: {self.forward}\n')
        
        ## dict handlers
        self.vldt_info = VldtInfo(cfg_ds, cfg_vldt.PER_WHAT)
        self.watch_info = WatchInfo(cfg_ds, watching)
        
        ## sets the ret_att only if we watching 'attws' 
        ## so net_out['attws'] are saved/returned in watch_info.DATA
        if watching: self.ret_att = 'attws' in watching
        else: self.ret_att = False
        
        ## after self.forward() run, self.scores will have the SL scores acquired from chosen net forward
        self.scores, self.attws = [], [] #None, None
    
        
    def reset(self):
        ## reset the wtch/vldt_info.DATA
        ## so reruns of start method in train have empty lists up on the start
        ## as the metrics need the dict keys as arrays
        self.vldt_info.reset()
        self.watch_info.reset()
        
    @staticmethod
    def reshp_nd_fill(arr1, new_len):
        new_arr = torch.full(new_len, arr1[-1], dtype=arr1.dtype)
        new_arr[:arr1.shape[0]] = arr1
        return new_arr
    
    
    @torch.no_grad()
    def _forward_attnomil(self, net, feat):
        '''
        as one score per forward, segment&repeat to get len(scores)=len(feats)
        #if self.cfg.DATA.RGB.SEGMENTNFRAMES == 64: split_size = 8
        #elif self.cfg.DATA.RGB.SEGMENTNFRAMES == 32: split_size = 16 
        #elif self.cfg.DATA.RGB.SEGMENTNFRAMES == 16: split_size = 9 ## orignal
        '''
        split_size = self.cfg.NET.ATTNOMIL.VLDT_SPLIT
        
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
                #tmp_attw = np.repeat( out['attw'], self.cfg.DATA.RGB.SEGMENTNFRAMES, axis=0 )
                #log.debug(f'tmp_attw[{ci}] {tmp_attw.shape} {type(tmp_attw)}')
                self.attws.append( out['attw'].repeat( self.cfg.DATA.RGB.SEGMENTNFRAMES, dim=0 ) )

        self.scores = torch.cat((self.scores), dim=0)
        log.debug(f'scores: {self.scores.shape} {self.scores.ctx}')
        
        if self.ret_att: 
            self.attws = torch.cat((self.attws), dim=0)
            log.debug(f"attws: {self.attws.shape} {self.attws.ctx}") 
    
    '''
    @staticmethod
    def cos_sim(x1,x2,dim):
        ## cos_sim = np.sum(x1 * x2, axis=dim) / ( np.sqrt(np.sum(x1 ** 2, axis=dim)) * np.sqrt(np.sum(x2 ** 2, axis=dim)) + 1e-8 )
        return np.sum(x1 * x2, axis=dim) / ( np.sqrt(np.sum(x1 ** 2, axis=dim)) * np.sqrt(np.sum(x2 ** 2, axis=dim)) + 1e-8 )
    def _forward_tad(self, net, feat, origem=True):
        ## OFFLINE , NEEDS FULL LENGTH CONTEXT TO GET SL SCORES
        ## DONE BY THE INTERPOLATION
        if origem:
            ## interpolate
            seglen = 32
            nsegments = feat.shape[1]
            feat = feat.squeeze(axis=0)
            log.debug(f'{feat.shape=}')
            
            new_feat = np.zeros((seglen, feat.shape[1]), dtype=np.float32).copyto(self.dvc)
            r = numpy.linspace(0, len(feat), seglen+1, dtype=numpy.int32)
            for i in range(seglen):
                if r[i] != r[i+1]: new_feat[i, :] = np.mean(feat[r[i]:r[i+1], :], axis=0)
                else: new_feat[i, :] = feat[r[i], :]
            log.debug(f'{new_feat.shape=}')
            feat = np.expand_dims(new_feat,axis=0)
            
        ndata = net(feat)
        #ndata['slscores'] , ndata['feat']
        
        ## hoe much does adjacent (encoded temporal cnv ks7 enhacment) features look alike
        ano_score = np.zeros_like(ndata['feat'][0,:,0])
        ##ano_cos = cosine_similarity(ndata['feat'][0,:-1], ndata['feat'][0,1:], dim=1)
        ano_cos = self.cos_sim(x1=ndata['feat'][0,:-1], x2=ndata['feat'][0,1:], dim=1)
        log.debug(f"{ano_cos.shape=} {ano_cos}")
        
        ## if abrupt change in adjacent features ano_cos is ~0 > ano_score ~1
        ano_score[:-1] += 1-ano_cos
        ano_score[1:] += 1-ano_cos
        ano_score[1:-1] /= 2
        log.debug(f"{ano_score.shape=} {ano_score=}")
        
        tmpdy = ano_score
        ## ????
        #log.debug(f"{tmpdy.shape=}")
        #tmpdy = np.reshape(tmpdy,(-1,1))
        #log.debug(f"{tmpdy.shape=}")
        #tmpdy = np.mean(tmpdy,axis=1)
        log.debug(f"tmpdy: {tmpdy=}")
        
        tmpse = ndata['slscores'].squeeze() ## (t)
        ## ????
        #tmpse = np.reshape(tmpse,(-1,1))
        #log.debug(f"{tmpse.shape=}")
        #tmpse = np.mean(tmpse,axis=1)
        log.debug(f"tmpse: {tmpse=}")
        
        
        tmp = (tmpdy+tmpse)/2
        log.debug(f"{tmp.shape=} {tmp}")
        
        if origem:
            ## extrapolate
            extrapolated_outputs = []
            extrapolation_indicies = np.round(np.linspace(0, len(tmp) - 1, num=nsegments))
            for index in extrapolation_indicies:
                extrapolated_outputs.append(tmp[int(index)])
            self.scores = np.array(extrapolated_outputs)
        else:
            self.scores = tmp
            
        log.debug(f'scores post0 {self.scores.shape=}')
        self.scores = self.scores.squeeze() #axis=1
        if self.scores.shape == (): ## == (1,1)
            self.scores = np.expand_dims(self.scores, axis=0)
        log.debug(f'scores post1 {self.scores.shape=}') #
    '''
    @torch.no_grad()    
    def _forward_glob(self, net, feat):
        ndata = net(feat)
        
        self.scores = self.net_pst_fwd.infer(ndata)

        
        if ndata['id'] == 'zzz':
            
            if 'attw' in ndata:
                log.debug(f'slcores {ndata["slscores"]=}')
                log.debug(f'attw {ndata["attw"]=}')
                #self.scores = ndata['attw'] * ndata['slscores']
                self.scores = ndata['slscores']
            else:
                self.scores = ndata['slscores']
        
        ## every network outputs slscores
        #else: self.scores = ndata['slscores']
        
        ## edge cases    
        log.debug(f'scores post0 {self.scores.shape=}')
        self.scores = self.scores.squeeze() #axis=1
        if self.scores.shape == (): ## == (1,1)
            self.scores = np.expand_dims(self.scores, axis=0)
        log.debug(f'scores post1 {self.scores.shape=}') #
        
    def start(self, net):
        net.eval()
        
        gt_vl, scor_vl = [], []
        #gt_sl, scor_sl = [], []
        #gt_fl, scor_fl = [], []
        tic = time.time()
        
        log.info(f'$$$$ Validate starting')
        for i, data in enumerate(self.DL):
            log.debug(f'[{i}] ********************')

            feat=data[0][0]; label=data[1][0]; fn=data[2][0]
            feat = np.expand_dims(feat,axis=0).copyto(self.dvc)
            log.debug(f'[{i}] {feat.shape} , {fn} , {label}')
            self.forward(net, feat)
            ## self.scores is at segment level 
            
            ##############
            ## video-level
            ## topkmean ?
            scores_mean = self.scores.mean(axis=0)
            #scores_max = tmp_sl.max()
            #log.info(f'VL[{i}]: {label} {label_decod(label)} | mean:{scores_mean}  max:{scores_max})')
            #scor_vl.append(scores_mean.asnumpy())
            scor_vl.append(scores_mean.item() if scores_mean.size == 1 else scores_mean.asnumpy())
            gt_vl.append(0 if '000' in label[0] else 1)
            log.debug(f'scor_vl {scor_vl[-1]} , gt_vl {gt_vl[-1]}')
            
            #################
            ## segments-level 
            ## (1 score per segment)
            #scor_sl.append(self.scores)
            
            ##############
            ## frame-level
            ## base the length of GT/FL by number of frames in video 
            ##      -> reshp_nd_fill FL to match gt length
            ## base the length of GT/FL by length of scores (which are at segment level) * length of feat_ext wind
            ##      -> truncate the generated GT (which has video nframes length) to match the FL 
            
            tmp_gt = self.gtfl.get(osp.join(self.cfg_ds.VROOT,f'{fn}.mp4'))
            log.debug(f' tmp_gt: {len(tmp_gt)}')
                
            ## 1 segmnt = self.cfg.DATA.RGB.SEGMENTNFRAMES = (64 frames, slowfast mxnet) (32 frames, i3d mxnet) (16, i3dtorchdmil)
            tmp_fl = self.scores.repeat( self.cfg.DATA.RGB.SEGMENTNFRAMES, dim=0 )
            log.debug(f' tmp_fl: {len(tmp_fl)} {tmp_fl.shape} {type(tmp_fl)} {type(tmp_fl[0])}')
            
            ########
            ## as len of seqlen * self.cfg.DATA.RGB.SEGMENTNFRAMES != number frames original video
            ## fill tmp_fl w/ last value until len is the same as gt
            if len(tmp_fl) < len(tmp_gt):  tmp_fl = self.reshp_nd_fill(tmp_fl, len(tmp_gt))
            ## or truncate tmp_gt with len(tmp_fl), thhis is how RocNG/XDVioDet does
            #min_len = min(len(tmp_fl), len(tmp_gt))
            #tmp_fl = tmp_fl[:min_len]
            #tmp_gt = tmp_gt[:min_len]
            log.debug(f' tmp_fl_rshp: {len(tmp_fl)}')
            
            assert len(tmp_fl) == len(tmp_gt), f'{self.cfg.DATA.RGB.SEGMENTNFRAMES} cfg.DATA.RGB.SEGMENTNFRAMES * {self.scores.shape[0]} seqlen  != {len(tmp_gt)} orign video frames'
            
            self.vldt_info.updt(label, fn, tmp_gt, tmp_fl)
            self.watch_info.updt(fn, tmp_gt, tmp_fl, self.attws)
        
        
        self.vldt_info.log()        
        self.vldt_info.upgrade()
        self.vldt_info.log()
        
        self.watch_info.log()        
        self.watch_info.upgrade()
        self.watch_info.log()
        
        
        ####################################
        ## try to perform a mean with a window size == SEGMENTNFRAMES over the GT 
        ## with a step equal to half of that
        ## so lenght of GT equals tmp_sl (scores output of net)
        
        ## and perform metrics at the Seqment Level in anomalies
        ## to mke it more realistic/live use fwd_atm
        ##  in opossosed to feed full feats, does it differ ??
        
        
        ## VIDEO LEVEL
        #self.metrics.get_vl(gt_vl, scor_vl)

        ## FRAME LEVEL
        mtrc_info = self.metrics.get_fl(self.vldt_info)
        
        log.info(f'$$$$ VALIDATE completed in {hh_mm_ss(time.time() - tic)}')
        return self.vldt_info, self.watch_info.DATA, mtrc_info


###########
class GTFL:
    '''
        Retrieves the Ground Truth Frame Level for the specified vpath
        either for the XDV or UCF dataset
    '''
    def __init__(self, cfg_ds):
        #log.debug(f'GTFL:\n{cfg_ds}')
        self.gt_path = cfg_ds.GT
        self.vroot_path = cfg_ds.VROOT
        self.get_data()

    def get_data(self):
        with open(self.gt_path, 'r') as txt: txt_data = txt.read()
        self.vlines = [line.split() for line in txt_data.split('\n') if line]
        
    def get(self, vpath, nframes=0):
        vn = osp.basename(vpath)
        
        if not nframes:
            tmp_cv = cv2.VideoCapture( vpath )
            nframes = int(tmp_cv.get(cv2.CAP_PROP_FRAME_COUNT))
            tmp_cv.release()
            
        tmp_gt = [0 for _ in range(nframes)]
        
        vline = next((item for item in self.vlines if str(item[0]) in vn), None)
        ## if in annotations -> abnormal
        if vline is not None:
            log.debug(f'GTFL.get({vn}) found {vline}')
            
            ## ucf: Burglary005_x264.mp4 Burglary 4710 5040 -1 -1
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
        
        #else: its normal full 0's     
            
        log.debug(f'GTFL.get({vn}) gt {len(tmp_gt)}')
        return tmp_gt

