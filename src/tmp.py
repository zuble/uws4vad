import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import glob , os, os.path as osp, time
from hydra.utils import instantiate as instantiate

from src.data import *
from src.utils import get_log, Visualizer
log = get_log(__name__)


def mpcs(cfg):
    from pytorch_metric_learning.samplers import MPerClassSampler
    from src.data import LBL, FeaturePathListFinder
    from torch.utils.data import BatchSampler
    
    labels_id = cfg.data.lbls.info[:-2]
    log.info(labels_id)
    
    #traindl, trainfrmt = get_trainloader(self.cfg) 
    
    rgbfplf = FeaturePathListFinder(cfg, 'train', 'rgb')
    argbfl, nrgbfl = rgbfplf.get('ANOM', cfg.dataproc.culum), rgbfplf.get('NORM')
    rgbfl = argbfl + nrgbfl

    lbl_mng = LBL(ds=cfg.data.id, cfg_lbls=cfg.data.lbls)
    
    lbls=[]
    for path in rgbfl: lbls.extend( lbl_mng.encod(osp.basename(path)) )
    
    m = len(labels_id)  ## xdv ~ 7
    ## 128 -> 126 | 64 -> 63 | 32 -> 28
    bs = cfg.dataload.bs - cfg.dataload.bs // m
    niters = len(rgbfl)
    log.info(f"{m=} {bs=} {niters=}   ")
    sampa = MPerClassSampler(lbls, m, bs, niters)
    
    
    bsampler = BatchSampler(sampa, bs, True)
    log.info(f" {len(sampa)=}  {len(bsampler)=}")
    
    
    #for bi, batch_idxs in enumerate(bsampler):
    #    for batch_idx in batch_idxs:
    #        log.info(f"[{bi}] {lbls[batch_idx]}   ")
    
    #a={ f'MPerClassSampler': bsampler}
    #analyze_sampler(a, lbls, cfg.data.id, iters=1,vis=None)
    
    
def embeds(cfg):
    vis = Visualizer('TMP_EMBEDS', 
            restart=cfg.xtra.vis.restart, 
            delete=cfg.xtra.vis.delete
            )
    
    #trn_inf = {
    #    'epo': 0,
    #    'bat': 0,
    #    'step': 0,
    #    'ttic': None,
    #    'dvc': cfg.dvc
    #}
    #dataloader, collator = get_trainloader(cfg, vis)
    #for batch in dataloader:
    #    feat, ldata = collator(batch, trn_inf)
    #    log.info(f"{feat.shape}")
    #    break
    
    DL = get_testloader(cfg)
    for i, data in enumerate(DL):
        feat=data[0][0]; label=data[1][0]; fn=data[2][0]
        break
    log.info(f"{fn}  {feat.shape}   {label}  ")
    
    #label = label
    
    net, inferator = build_net(cfg)
    
    #label = ldata["label"].repeat_interleave(cfg.dataproc.crops2use.train)
    #vis.embeddings()
    
    
    
class Debug():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def hydra(self):
        log = utils.get_log(__name__, self.cfg)
        log.debug(f"Working dir : {os.getcwd()},\nOriginal dir : {hydra.utils.get_original_cwd()} ")
        log.debug(utils.collect_random_states())
        utils.xtra(self.cfg)
    
    def testset(self):
        get_testxdv_info(self.cfg)
        
    def data(self):
        traindl, trainfrmt = get_trainloader(self.cfg) 
        run_dl(traindl)
        
        #testdl = get_testoader(self.cfg) 
        #run_dl()
        
    def loss(self):
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

    def net(self, cfg):
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
            

def aud_len_mat():
    len_rgb = 12
    len_aud = 11
    
    idxs = np.linspace(0, len_aud - 1, len_rgb, dtype=np.int32) 
            
    idxs2 = np.linspace(0, len_aud-1, len_rgb, endpoint=True, dtype=np.int32)
    idxs3 = np.round(np.arange(len_rgb) * (len_aud) / (len_rgb)).astype(np.int32)
    
    
    repeat_counts = np.ones(len_aud, dtype=int)
    # Distribute the required repetitions at the end of the array
    for i in range(len_rgb - len_aud):
        repeat_counts[-(i % len_aud) - 1] += 1
    idxs4 = np.repeat(np.arange(len_aud), repeat_counts)
    
    log.warning(f'idxs {idxs.shape} \n {idxs}')
    log.warning(f'idxs2 {idxs2.shape} \n {idxs2}')
    log.warning(f'idxs3 {idxs3.shape} \n {idxs3}')
    log.warning(f'idxs4 {idxs4.shape} \n {idxs4}')

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