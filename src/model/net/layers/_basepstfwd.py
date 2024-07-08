## every netowrk has a NetPstFwd class
## which contains the pst processing depending if train / infer
## they inherit BasePstFwd so rsp_out is acessible w/o import or repeats
## relying on the cfg.TRAIN.BS and cfg.DATA.RGB.NCROPS 
## in pair with the creation of network in nets/_nets/get_net()
## while keepig the info regarding every post processing after net fwd 
## in the same file as the net def for both 
##  train, directly interact w net outs and inputs of chosen lossfx's
##      levarage that each lossfx returns a dict
##      furthermore the train_epo expects a dict of losses
##  infer, to adapt values 4 inference
## train_ep/train_epo (use netpstfwd.train)
## vldt/vldt/Validate (use netpstfwd.infer)
## both enabling the principal loop to stay static nd focus on architect
## if really needed realy on id key in ndata to post process both cases
import torch
from src.utils.logger import get_log
log = get_log(__name__)

class BasePstFwd:
    def __init__(self, _cfg):
        self.bs = _cfg.bs
        self.ncrops = _cfg.ncrops
        self._cfg = _cfg
        
    ## mod data['key']: reduce a 3d tensor to a 2d by crop mean / crop0 sel / or orig
    def uncrop(self, data, key, meth=""):
        if self.ncrops > 1:
            
            if data[key].ndim == 2: ## sl
                data[key] = data[key].view(self.bs, self.ncrops, -1)
                
                if meth == 'mean':
                    data[key] = torch.mean(data[key], dim=1)
                elif meth == 'crop0':
                    data[key] = data[key][:, 0, :]
                
            elif data[key].ndim == 3: ## fl
                _,t,f = data[key].shape
                data[key] = data[key].view(self.bs, self.ncrops, t, f)
                
                if meth == 'mean':
                    data[key] = torch.mean(data[key], dim=1)
                elif meth == 'crop0':
                    data[key] = data[key][:, 0, :, :]

    ## returns: reduce arr 3d tensor to a 2d b mean or crop0 sel
    ## additional 
    def uncrop2(self, arr, meth, force=False):
        if self.ncrops > 1:
            
            if arr.ndim == 2: ## sl
                arr = arr.view(self.bs, self.ncrops, -1)
                
                if meth == 'orig': pass
                if meth == 'mean':
                    arr = torch.mean(arr, dim=1)
                elif meth == 'crop0':
                    arr = arr[:, 0, :]
                        
            elif arr.ndim == 3: ## fl
                _,t,f = arr.shape
                arr = arr.view(self.bs, self.ncrops, t, f)
                
                if meth == 'orig': pass
                if meth == 'mean':
                    arr = torch.mean(arr, dim=1)
                elif meth == 'crop0':
                    arr = arr[:, 0, :, :]
                    
        elif force: ## nocrops
            if arr.ndim == 2: ## sl
                raise NotImplementedError
            
            elif arr.ndim == 3: ## fl
                _,t,f = arr.shape
                arr = arr.view(self.bs, 1, t, f)   
    
        return arr            
        

    def unbag2(self, arr, labels=None, view_crop=False):
        ## bs
        if labels == None: # assumes MIL
            abn = arr[self.bs//2:] 
            nor = arr[0:self.bs//2]
        else:
            abn = arr[ labels != 0]
            nor = arr[ labels == 0]
            
        if view_crop:
            if arr.ndim != 4: ## sl
                raise NotImplementedError
            
            abn = abn.permute(1, 0, 2, 3)# # (nc, bag, t, f)
            nor = nor.permute(1, 0, 2, 3) ## (nc, bag, t, f)                   
                
        return abn, nor 
        
    #def get_fmgnt(self, feats, labels):
    #    ## (bs*ncrops, t, f)
    #    feat_magn = torch.norm(feats, p=2, dim=2)  ## ((bs)*ncrops, t)
    #    feat_magn = self.uncrop2(feat_magn, 'mean') ## (bs, t)
    #    
    #    return feat_magn[labels != 0], feat_magn[labels == 0]
        

        
    ## agg
    def merge(self, *dicts):
        res = {}
        for d in dicts: res.update(d)
        return res
    
        
## exemplar
class NetPstFwdEx(BasePstFwd):
    def __init__(self, _cfg):
        super().__init__(_cfg)

    def train(self, ndata, ldata, lossfx):
        log.debug(f"")
        
        super().rshp_out(ndata, '', 'mean') ## crop0
        log.debug(f" pos_rshp: {ndata[''].shape}")
        
        ## every lossfx returns a dict
        L0 = lossfx[''](ndata[''])
        L1 = lossfx[''](ndata[''])
        
        ## merges indiv loss to be further summed to .backward 
        return super().merge(L0, L1)

    def infer(self, ndata):
        ## output is excepted to be clip level 
        log.debug(f"")
        log.debug(f"slscores: {ndata['slscores']=}")
        
        return ndata['slscores']    
    
    
