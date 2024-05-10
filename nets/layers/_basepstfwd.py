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
## train_ep/train_epo and vldt/vldt/Validate stays static nd focus on architect
## if really needed realy on id key in ndata o process in train_epo
import torch

class BasePstFwd:
    def __init__(self, bs, ncrops, dvc):
        self.bs = bs
        self.ncrops = ncrops
        self.dvc = dvc
        self.lbat = {} ## stores loss bat values, so they avaiable in train_epo
        
    ## mod data['key']: reduce a 3d tensor to a 2d by crop mean or crop0
    def rshp_out(self, data, key, meth):
        if self.ncrops > 1:
            
            if data[key].ndim == 2:
                data[key] = data[key].view(self.bs, self.ncrops, -1)
                if meth == 'mean':
                    data[key] = torch.mean(data[key], dim=1)
                elif meth == 'crop0':
                    data[key] = data[key][:, 0, :]
                    
            elif data[key].ndim == 3:
                b,t,f=data[key].shape
                data[key] = data[key].view(self.bs, self.ncrops, t, f)
                if meth == 'mean':
                    data[key] = torch.mean(data[key], dim=1)
                elif meth == 'crop0':
                    data[key] = data[key][:, 0, :, :]
    
    ## returns: reduce arr 3d tensor to a 2d by crop mean or crop0
    def rshp_out2(self, arr, meth):
        if self.ncrops > 1:
            
            if arr.ndim == 2:
                arr = arr.view(self.bs, self.ncrops, -1)
                if meth == 'mean':
                    arr = torch.mean(arr, dim=1)
                elif meth == 'crop0':
                    arr = arr[:, 0, :]
                    
            elif arr.ndim == 3:
                b,t,f=arr.shape
                arr = arr.view(self.bs, self.ncrops, t, f)
                if meth == 'mean':
                    arr = torch.mean(arr, dim=1)
                elif meth == 'crop0':
                    arr = arr[:, 0, :, :]
        return arr
                    
    def updt_lbat(self, key, value):
        self.lbat[key] = value
        
        
## exemplar
class NetPstFwdEx(BasePstFwd):
    def __init__(self, bs, ncrops, dvc):
        super().__init__(bs, ncrops, dvc)

    def train(self, ndata, ldata, lossfx):
        log.debug(f"")
        
        super().rshp_out(ndata, '', 'mean')
        #super().rshp_out(ndata, '', 'crop0')
        log.debug(f" pos_rshp: {ndata[''].shape}")
        loss0 = lossfx[''](ndata[''])
        self.updt_lbat('', loss0.item())
        
        return loss0 


    def infer(self, ndata):
        ## output is excepted to be segment level 
        log.debug(f"")
        log.debug(f"slscores: {ndata['slscores']=}")
        
        return ndata['slscores']    