import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import glob , os, os.path as osp, time
from hydra.utils import instantiate as instantiate

from src.data import get_trainloader, run_dl
from src.utils.logger import get_log
log = get_log(__name__)




class Debug():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug_data()
        
        
    def debug_data(self):
        traindl, trainfrmt = get_trainloader(self.cfg) 
        run_dl(traindl)
        
        
    
    def debug_loss(self):
        ## set debug.model=1
        from src.model.loss.mgnt import Rtfm
        
        nc = 2
        bag = 4
        t = 16
        dfeat = 512
        
        loss_cfg = {
            '_target_': 'src.model.loss.mgnt.Rtfm',
            '_cfg': {
                'k': 3,
                'alpha': 0.0001,
                'margin': 100,
            }
        }
        
        L = instantiate( loss_cfg )
        _ = L(
            abnr_fmagn=torch.randn(bag, t),
            norm_fmagn=torch.randn(bag, t),
            abnr_feats=torch.randn(nc, bag, t, dfeat),
            norm_feats=torch.randn(nc, bag, t, dfeat),
            abnr_sls=torch.randn(bag, t),
            norm_sls=torch.randn(bag, t),
            ldata=
            {
                'label': torch.cat( (torch.zeros(bag), torch.ones(bag)) )
            }
        )

    def debug_net(self, cfg):
        from src.model.net.rtfm import Network
        net_cfg = {
            '_target_': 'src.model.net.rtfm.Network',
            '_cfg': {
                'do': 0.7,
            }
        }
        
        dfeat = self.cfg.data.ds.frgb.dfeat + (self.cfg.data.ds.faud.dfeat if self.cfg.data.ds.get("faud") else 0)
        
        if self.cfg.model.net.get("cls"): 
            net = instantiate(cfg.model.net.main, dfeat=dfeat, _cls=self.cfg.model.net.cls, _recursive_=False)#.to(cfg.dvc)
        else:
            net = instantiate(cfg.model.net.main, dfeat=dfeat)#.to(cfg.dvc)
            


def aud_emb(cfg):
    from src.fe.hear_mn import mn10_all_b as mn10_MB
    from src.fe.hear_mn import mn10_all_b_all_se as mn10_MB_MSE

    model = mn10_MB_MSE.load_model()
    
    vids = 1
    secs = 32
    sr = 32000

    # n_sounds * n_samples
    audio = torch.ones((vids, int(sr * secs)))
    
    ## (1, t, d) // (1, t)
    embed, time_stamps = model.get_timestamp_embeddings(audio) 
    log.info(f"{embed.shape} {time_stamps.shape} {type(embed)}")

    #fps = 24
    #vid_len = secs * fps
    #clip_len = cfg.datatrnsfrm.clip_len
    #seg_len = int(vid_len / clip_len) #3
    #for vid in range(vids):
    #    lin = np.linspace(0, len(time_stamps[vid]), seg_len, dtype="int")
    #    log.info(lin)
    #    
    #    feat = np.zeros( (len(lin)-1, embed.shape[-1]), dtype="float32")
    #    #feat = []
    #    for i in range(len(lin)-1):
    #        feat[i] = torch.mean( embed[vid][lin[i]:lin[i+1]], dim=0).numpy()
    #        #feat.append( torch.mean( embed[vid][lin[i]:lin[i+1]], dim=0).numpy() )
    #    #feat = np.array(feat)
    #    log.info(feat.shape)

    ## get_timestamp_embeddings
    import torch.nn.functional as F
    window_size=160
    hop=50 
    pad = window_size // 2
    audio = audio.cpu()
    n_sounds, n_samples = audio.shape
    audio = audio.unsqueeze(1)  # n_sounds, 1, (n_samples+pad*2)
    log.info(f"{audio.shape}")
    padded = F.pad(audio, (pad, pad), mode='reflect')
    log.info(f"{padded.shape}")
    padded = padded.unsqueeze(1)  # n_sounds, 1, (n_samples+pad*2)
    log.info(f"{padded.shape}")
    segments = F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop)).transpose(-1, -2).transpose(0, 1)
    log.info(f"{segments.shape}")
    timestamps = []
    embeddings = []
    for i, segment in enumerate(segments):
        timestamps.append(i)
        embeddings.append(model.forward(segment).cpu())
    log.info(f"{segments.shape}")
    timestamps = torch.as_tensor(timestamps) * hop * 1000. / sr

    embeddings = torch.stack(embeddings).transpose(0, 1)  # now n_sounds, n_timestamps, timestamp_embedding_size
    timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)


    #embed = model.get_scene_embeddings(audio)
    #print(embed.shape)
    #print(embed[0]) 
    # tensor([-0.3774,  0.9759, -1.4932,  ..., -0.2288,  0.1544, -0.0292])


    #embed_norm = torch.norm(embed, p=2, dim=0)
    #print(f" {embed_norm.shape} {max(embed_norm)} ")
    #print(embed_norm)
    #
    #embed_relu = torch.relu(embed)
    #print(f" {embed_relu.shape} {max(embed_relu)} ")
    #print(embed_relu)