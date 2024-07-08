#import clip
import torch
import timm
from timm.data import resolve_data_config, create_transform
import numpy as np
import os, os.path as osp, glob, time, random


from src.fe.handler import get_aud_model, get_vid_model
from src.fe.dlvid import get_vidloader
from src.fe.dlaud import get_audloader

from src.utils import get_log
log = get_log(__name__)


## https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py
class VisFeatExtract():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data
        self.out_dir = cfg.path.fext.out_dir
        log.info(f"SAVING @ {self.out_dir}")

        self.fwd = {'full': self._fwd_full, 
                    'iter': self._fwd_iter}[self.cfg.fwd_feed]
        self.start()
        
    def start(self):
        """
        Extracts audio features and saves them to .npy files.

        This function handles different scenarios for specifying input paths:
        1. If cfg.ext_dir is a string (single path), it extracts features from all .mp4 files in that path.
        3. If cfg.ext_dir is None, it extracts features from all .mp4 files in each subfolder of cfg.data.vroot.
        """
        
        if isinstance(self.cfg.ext_dir, str):
            folder2proc = [self.cfg.ext_dir]
        elif self.cfg.ext_dir is None:
            #folder2proc = [osp.join(self.cfg.data.vroot, folder) for folder in os.listdir(self.cfg.data.vroot) if osp.isdir(osp.join(self.cfg.data.vroot, folder))]
            folder2proc = [osp.join(self.cfg.data.vroot, 'TRAIN'), osp.join(self.cfg.data.vroot, 'TEST')]
        else:
            raise ValueError("Invalid cfg.ext_dir type. Must be list, str, or None.")
        
        ## MODEL
        model, trnsfrm, cfg_model = get_vid_model(self.cfg)
        try: assert model.training == False
        except: pass ## openai
        
        for folder in folder2proc:
            log.info(f"Processing folder: {folder}")
            
            ## PATHS            
            vpaths = glob.glob(osp.join(folder, "*.mp4"))
            vpaths.sort()
            if not vpaths:
                log.warning(f"No .mp4 files found in {folder}. Skipping.")
                continue
            if self.cfg.get("debug"): vpaths = vpaths[:1]

            dataloader = get_vidloader(self.cfg, trnsfrm, cfg_model, vpaths)
            
            ## OUT_FOLDER
            tmp = f"{cfg_model.id}"
            if cfg_model.get("vrs"):
                tmp += f"__{cfg_model.vrs}"
            model_name = tmp.upper()
            out_folder = osp.join(osp.join(self.out_dir, f"{model_name}") , osp.basename(folder))
            os.makedirs(out_folder, exist_ok=True)
            log.info(f"out_folder: {out_folder}")

            self._proc_dataloader(dataloader, out_folder, model)
            
    def _proc_dataloader(self, data_loader, out_folder, model): 
        """Processes a single DataLoader and saves extracted features."""
        extracted = 0; tic = time.time()
        
        for vidx, data in enumerate(data_loader):
            video = data[0][0]; vn = data[1][0]  
            log.info(f'[{vidx}] {vn} {video.shape} ')
            
            
            out_npy = osp.join(out_folder, f'{vn}.npy')
            if osp.exists(out_npy):
                log.info(f"Feature file for video {vn} already exists. Skipping.")
                continue
            #out_npys = [osp.join(self.out_dir, f'{vn}' + (f'__{i}' if self.cfg_trnsfrm.ncrops > 1 else '') + '.npy') for i in range(self.cfg_trnsfrm.ncrops)]
            #if all([osp.exists(out_npy) for out_npy in out_npys]):
            #    continue
            
            feats = self.fwd(video, model) 
            log.info(f'{video.shape} {video.dtype} -> {feats.shape} {type(feats)} {feats.dtype}')
            
            og_nclips = self._get_og_nclips(vn, osp.basename(out_folder))
            if og_nclips == -1:
                log.error(f"No matching original feature file found for {vn}. Skipping.")
                continue  # Skip to the next video
            elif feats.shape[0] == og_nclips:
                log.debug(f"OG check successful for {vn}: {og_nclips} clips.")
            elif feats.shape[0] > og_nclips:
                log.warning(f"More new clips ({feats.shape[0]}) than original ({og_nclips}) for {vn}. Truncating new features.")
                feats = feats[:og_nclips]
            else:  # feats.shape[0] < og_nclips
                log.error(f"Fewer new clips ({feats.shape[0]}) than original ({og_nclips}) for {vn}. This is unexpected! Skipping.")
                continue  # Skip to the next video
            
            np.save(out_npy, feats)
            extracted += 1

        elapsed = time.time() - tic
        log.info(f"Extracted features for {extracted} videos in {elapsed:.2f} seconds.")
        
    def _fwd_full(self, clips, model):
        with torch.no_grad():
            return model(clips.to(self.cfg.dvc)).cpu().numpy()

    def _fwd_iter(self, clips, model):
        feats = []
        for i, clip in enumerate(clips):
            with torch.no_grad():
                feat = model(clip.to(self.cfg.dvc).unsqueeze(0))
            feats.append(feat.cpu().numpy())
        return np.vstack(feats)

    def _get_og_nclips(self, vn, folder_name, source='I3DROCNG'):
        """
        Checks the length (number of features) of the corresponding RGB feature file.
        Args: vn (str): The base name of the video (e.g., "Abuse001").
        Returns: int: The length of the RGB feature file, or -1 if not found.
        """
        seapat = self.cfg.data.froot + f"/RGB/{source}/{folder_name}/{vn}*.npy"
        matching_files = glob.glob(seapat)
        log.debug(f"{seapat} -> {matching_files}")
        
        if matching_files: 
            return np.load(matching_files[0]).shape[0] #crop0
        else:
            log.error(f"No matching RGB feature file found for {vn}")
            return -1


class AudFeatExtract():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data
        self.out_dir = cfg.path.fext.out_dir
        log.info(f"SAVING @ {self.out_dir}")
        
        self.model, self.cfg_model= get_aud_model(cfg)
        assert self.model.net.training == False ## hear_mn
        
        if self.cfg_model.hear:
            self.fwd = self._fwd_hear
        else: raise NotImplementedError
        
        self.start()

    def _fwd_hear(self, aud_arr, fps):
        with torch.no_grad():
            embeds, tstmps = self.model.get_timestamp_embeddings(aud_arr.to(self.cfg.dvc))
        embeds = embeds[0]
        log.info(f"{embeds.shape=} {tstmps.shape=} {embeds}")
        
        ## pst prc to match seglen as frgb
        clip_dur = self.cfg_model.clip_len / fps
        samples_per_clip = clip_dur * self.cfg_model.sr
        tstmps_per_clip = int(samples_per_clip / self.model.timestamp_hop)
        log.info(f"{tstmps_per_clip}")
        
        clip_starts = range(0, tstmps.shape[-1], tstmps_per_clip)
        tmp = [embeds[s:s+tstmps_per_clip].mean(dim=0) for s in clip_starts]
        sl_embeds = torch.stack(tmp)
        log.info(f"{sl_embeds.shape} {sl_embeds}")
        
        return sl_embeds.cpu().numpy()
        

    def start(self):
        """
        Extracts audio features and saves them to .npy files.

        This function handles different scenarios for specifying input paths:
        1. If cfg.ext_dir is a string (single path), it extracts features from all .mp4 files in that path.
        3. If cfg.ext_dir is None, it extracts features from all .mp4 files in each subfolder of cfg.data.vroot.
        """
        
        if isinstance(self.cfg.ext_dir, str):
            folder2proc = [self.cfg.ext_dir]
        elif self.cfg.ext_dir is None:
            #folder2proc = [osp.join(self.cfg.data.vroot, folder) for folder in os.listdir(self.cfg.data.vroot) if osp.isdir(osp.join(self.cfg.data.vroot, folder))]
            folder2proc = [osp.join(self.cfg.data.vroot, 'TRAIN'), osp.join(self.cfg.data.vroot, 'TEST')]
        else:
            raise ValueError("Invalid cfg.ext_dir type. Must be str, or None.")
        
        for folder in folder2proc:
            log.info(f"in_folder: {folder}")
            
            ## PATHS
            vpaths = glob.glob(osp.join(folder, "*.mp4"))
            vpaths.sort()
            if not vpaths:
                log.warning(f"No .mp4 files found in {folder}. Skipping.")
                continue
            if self.cfg.get("debug"): vpaths = vpaths[:1]
                
            dataloader = get_audloader(self.cfg, self.cfg_model, vpaths)
            
            ## OUT_FOLDER
            tmp = f"{self.cfg_model.id}"
            if self.cfg_model.get("vrs"):
                tmp += f"_{self.cfg_model.vrs}"
            model_name = tmp.upper()
            out_folder = osp.join(osp.join(self.out_dir, f"{model_name}") , osp.basename(folder))
            os.makedirs(out_folder, exist_ok=True)
            log.info(f"out_folder: {out_folder}")
            
            self._proc_dataloader(dataloader, out_folder)
            
        
    def _proc_dataloader(self, dataloader, out_folder):
        """Processes a single DataLoader and saves extracted features."""
        extracted = 0; tic = time.time()
        log.info(f"starting extraction...")
        
        for aidx, data in enumerate(dataloader):
            audio = data[0][0]; vn = data[1][0]; fps = data[2].item()  # Extract data from the batch
            log.info(f'[{aidx}] {vn} {audio.shape=} {fps=}')
            
            out_npy = osp.join(out_folder, f'{vn}.npy')
            if osp.exists(out_npy):
                log.info(f"Feature file for audio {vn} already exists. Skipping.")
                continue
            
            feats = self.fwd(audio, fps)
            log.info(f'{audio.shape} {audio.dtype} -> {feats.shape} {type(feats)} {feats.dtype}')
            
            og_nclips = self._get_fog(vn, osp.basename(out_folder))
            if og_nclips == -1: log.error(f"mismatch betwen og {og_nclips} and new {feats.shape[0]}"); continue
            log.info(f"OG {vn}: {og_nclips=}")
            
            np.save(out_npy, feats)
            extracted += 1
            
        elapsed = time.time() - tic
        log.info(f"Extracted features for {extracted} audio files in {elapsed:.2f} seconds.")
        
    def _get_fog(self, vn, folder_name, source='VGGISH'):
        """
        Checks the number of clips from rocng files
        Args: vn (str): The base name of the video (e.g., "Abuse001").
        Returns: int: The length of the RGB feature file, or -1 if not found.
        """
        seapat = self.cfg.data.froot + f"/AUD/{source}/{folder_name}/{vn}*.npy"
        log.info(seapat)
        matching_files = glob.glob(seapat)
        log.debug(f"{seapat} -> {matching_files}")
        
        if matching_files: 
            tmp = np.load(matching_files[0])
            return tmp.shape[0] 
        else:
            log.warning(f"No matching RGB feature file found for {vn}")
            return -1