#import clip
import torch
import timm
from timm.data import resolve_data_config, create_transform
import numpy as np
import os, os.path as osp, glob, time, random, gc


from src.fe.handler import get_aud_model, get_vid_model
from src.fe.dlvid import get_vidloader, VideoDS
from src.fe.dlaud import get_audloader

from src.utils import get_log
log = get_log(__name__)


## https://github.com/OpenGVLab/VideoMAEv2/blob/master/extract_tad_feature.py
class VisFeatExtract():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg_dsinf = cfg.data
        if self.cfg.get("debug"): self.out_dir = cfg.path.fext.out_dir
        else: self.out_dir = osp.join(cfg.data.froot,"RGB") ## set accordinfly ds
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
        
        confirmation = input(f"all good ? (y) ")
        if confirmation.lower() != 'y': return
        
        for folder in folder2proc:
            log.info(f"{'*'*10}\nProcessing folder: {folder} / {len(folder2proc)}\n")
            
            ## PATHS            
            vpaths = glob.glob(osp.join(folder, "*.mp4"))
            vpaths.sort()
            if not vpaths:
                log.warning(f"No .mp4 files found in {folder}. Skipping.")
                continue
            if self.cfg.get("debug"): vpaths = vpaths[:1]

            ## OUT_FOLDER
            tmp = f"{cfg_model.id}"
            if cfg_model.get("vrs"):
                tmp += f"__{cfg_model.vrs}"
            model_name = tmp.upper()
            out_folder = osp.join(osp.join(self.out_dir, f"{model_name}") , osp.basename(folder))
            os.makedirs(out_folder, exist_ok=True)
            log.info(f"out_folder: {out_folder}")

            ## find duplicates
            vpaths_done = []
            for vpath in vpaths:
                vn = osp.splitext(osp.basename(vpath))[0]
                out_npy = osp.join(out_folder, f'{vn}.npy')
                if osp.exists(out_npy):
                    log.info(f"Feature 4 visual {vn} already exists. Skipping.")
                    vpaths_done.append(vpath)
            if vpaths_done:
                log.info(f"Found {(len(vpaths_done))} videos already processed out of {len(vpaths)}")
                vpaths = [a for a in vpaths if a not in vpaths_done]
            log.info(f"Processing {len(vpaths)} videos")
            if self.cfg.get("debug"): vpaths = vpaths[:2]
            
            
            #dataloader = get_vidloader(self.cfg, trnsfrm, cfg_model, vpaths)
            dataloader = VideoDS(self.cfg, trnsfrm, cfg_model, vpaths)
            self._proc_data(dataloader, out_folder, model)
            
    def _proc_data(self, data_loader, out_folder, model): 
        """Processes a single DataLoader and saves extracted features."""
        extracted = 0; tic = time.time()
        
        log.info(f"starting extraction...")
        confirmation = input(f"all good ? (y) ")
        if confirmation.lower() != 'y': return
        
        for vidx, data in enumerate(data_loader):
            #video = data[0][0]; vn = data[1][0]; safe = data[2][0]
            video = data[0]; vn = data[1]; safe = data[2]
            log.info(f"{'*'*33}")
            if not safe: log.warning(f"\n\n\tSomething wrong with {vn}. Skipping\n\n"); continue ## v=8cTqh9tMz_I__#1_label_A.mp4
            log.info(f'[{vidx}] {vn} {video.shape} ')
            
            out_npy = osp.join(out_folder, f'{vn}.npy')
            #if osp.exists(out_npy):
            #    log.info(f"Feature file for video {vn} already exists. Skipping.")
            #    continue
            #out_npys = [osp.join(self.out_dir, f'{vn}' + (f'__{i}' if self.cfg_trnsfrm.ncrops > 1 else '') + '.npy') for i in range(self.cfg_trnsfrm.ncrops)]
            #if all([osp.exists(out_npy) for out_npy in out_npys]):
            #    continue
            
            feats = self.fwd(video, model) 
            log.info(f'[{vidx}] {video.shape} {video.dtype} -> {feats.shape} {type(feats)} {feats.dtype}')
            
            ## -----
            og_nclips = self._get_og_nclips(vn, osp.basename(out_folder))
            if og_nclips == -1:
                log.error(f"[{vidx}] No matching original feature file found for {vn}. Skipping.")
                continue  # Skip to the next video
            elif feats.shape[0] == og_nclips:
                log.debug(f"[{vidx}] OG check successful for {vn}: {og_nclips} clips.")
            elif feats.shape[0] > og_nclips:
                log.warning(f"[{vidx}] More new clips ({feats.shape[0]}) than original ({og_nclips}) for {vn}. Truncating new features.")
                feats = feats[:og_nclips]
            else:  # feats.shape[0] < og_nclips
                log.error(f"[{vidx}] Fewer new clips ({feats.shape[0]}) than original ({og_nclips}) for {vn}. This is unexpected! Skipping.")
                continue  # Skip to the next video
            ## -----
            
            np.save(out_npy, feats)
            extracted += 1
            if extracted % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        log.info(f"Extracted features for {extracted} videos in {(time.time() - tic):.2f} seconds.")
    
    
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
        if self.cfg.get("debug"): self.out_dir = cfg.path.fext.out_dir
        else: self.out_dir = osp.join(cfg.data.froot,"AUD")
        log.info(f"SAVING @ {self.out_dir}")
        
        self.model, self.cfg_model= get_aud_model(cfg)
        assert self.model.net.training == False ## hear_mn
        
        if self.cfg_model.hear:
            self.fwd = self._fwd_hear    
        else: raise NotImplementedError
        
        self.start()

    def _fwd_hear(self, aud_arr, fps, vid_tframes):
        with torch.no_grad():
            embeds_og, tstmps = self.model.get_timestamp_embeddings(aud_arr.to(self.cfg.dvc))
        embeds_og = embeds_og[0]
        log.debug(f"{embeds_og.shape=}") #{embeds_og}
        log.debug(f" {tstmps.shape=} ") #{tstmps}
        
        new_step = 1
        idx_vid = list(range(0, vid_tframes, new_step))
        if vid_tframes < self.cfg_model.clip_len: 
            r = self.cfg_model.clip_len // vid_tframes + 1
            idx_vid = (idx_vid * r)[:self.cfg_model.clip_len]
        nclips = int(len(idx_vid) / self.cfg_model.clip_len)
        clip_dur = self.cfg_model.clip_len / fps

        vid_tsecs_og_frgb = nclips*clip_dur ##  Target audio duration to match RGB features 
        vid_tsecs_og_faud = embeds_og.shape[0] * self.cfg_model.hop / 1000  ## Actual audio duration after processing
        
        time_diff_seconds = vid_tsecs_og_faud - vid_tsecs_og_frgb
        num_embeds_to_remove = int(time_diff_seconds / self.cfg_model.hop * 1000)
        if num_embeds_to_remove > 0:
            embeds = embeds_og[:-num_embeds_to_remove]
        else: embeds = embeds_og
        
        idxs = np.linspace(0, embeds.shape[0], nclips+1, dtype=np.int32) ## Create segment boundaries using linspace
        sl_embeds_list = []
        for i in range(nclips):
            sl_embeds_list.append( embeds[idxs[i]:idxs[i+1]].mean(0) )
        sl_embeds = torch.stack(sl_embeds_list)
        
        log.info(f"OG VID {(vid_tframes / fps)}s | OG FRGB {vid_tsecs_og_frgb} {nclips=} {clip_dur=} | OG FAUD {vid_tsecs_og_faud}  {tstmps[0,-1]/1000}")
        log.info(f"from {embeds_og.shape} -> trunc to {embeds.shape} -> itp to {nclips=} {sl_embeds.shape=} ")
        
        ###########################
        #clip_dur = self.cfg_model.clip_len / fps
        #tstmps_per_clip = clip_dur / self.cfg_model.hop * 1000
        #log.info(f"{clip_dur=}   {tstmps_per_clip=}")
        
        #clip_dur = self.cfg_model.clip_len / fps
        #samples_per_clip = clip_dur * self.cfg_model.sr
        #tstmps_per_clip = int(samples_per_clip / self.model.timestamp_hop)
        #log.info(f"{clip_dur=} {samples_per_clip=} {tstmps_per_clip=}  {self.model.timestamp_hop=}")
        #
        #clip_starts = range(0, tstmps.shape[-1], tstmps_per_clip)
        #tmp = [embeds[s:s+tstmps_per_clip].mean(dim=0) for s in clip_starts]
        #sl_embeds = torch.stack(tmp)
        #log.info(f"{sl_embeds.shape=}") #{sl_embeds}
        
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
        
        confirmation = input(f"\nall good ? (y) ")
        if confirmation.lower() != 'y': return
        
        for folder in folder2proc:
            log.info(f"in_folder: {folder}")
            
            ## PATHS
            vpaths = glob.glob(osp.join(folder, "*.mp4"))
            vpaths.sort()
            if not vpaths:
                log.warning(f"No .mp4 files found in {folder}. Skipping.")
                continue
            
            ## OUT_FOLDER
            tmp = f"{self.cfg_model.id}"
            if self.cfg_model.get("vrs"):
                tmp += f"_{self.cfg_model.vrs}"
            model_name = tmp.upper()
            out_folder = osp.join(osp.join(self.out_dir, f"{model_name}") , osp.basename(folder))
            os.makedirs(out_folder, exist_ok=True)
            log.info(f"out_folder: {out_folder}")
            
            ## find duplicates
            vpaths_done = []
            for vpath in vpaths:
                vn = osp.splitext(osp.basename(vpath))[0]
                out_npy = osp.join(out_folder, f'{vn}.npy')
                if osp.exists(out_npy):
                    log.info(f"Feature 4 audio {vn} already exists. Skipping.") 
                    vpaths_done.append(vpath)
            if vpaths_done:
                log.info(f"Found {(len(vpaths_done))} videos already processed out of {len(vpaths)}")
                vpaths = [a for a in vpaths if a not in vpaths_done]
            log.info(f"Processing {len(vpaths)} videos")
            if self.cfg.get("debug"): vpaths = vpaths[:2]
            
            ## this is loading into mem full
            dataloader = get_audloader(self.cfg, self.cfg_model, vpaths)
            self._proc_data(dataloader, out_folder)
            
        
    def _proc_data(self, dataloader, out_folder):
        """Processes a single DataLoader and saves extracted features."""
        extracted = 0; tic = time.time()
        log.info(f"starting extraction...")
        confirmation = input(f"all good ? (y) ")
        if confirmation.lower() != 'y': return
        
        ## pass 2 try/except
        for aidx, data in enumerate(dataloader):
            audio = data[0][0]; vn = data[1][0]; fps = data[2].item(); tframes = data[3].item() 
            log.info(f"{'*'*33}")
            if fps == tframes == -1: log.warning(f"Something wrong with {vn}. Skipping"); continue ## v=8cTqh9tMz_I__#1_label_A.mp4
            log.info(f'[{aidx}] {vn} {audio.shape=} {fps=} {tframes=}')
            
            feats = self.fwd(audio, fps, tframes)
            log.info(f'{audio.shape} {audio.dtype} -> {feats.shape} {type(feats)} {feats.dtype}')
            
            #if self._compare_fog(vn, osp.basename(out_folder),  feats.shape[0]) == -1: continue
            out_npy = osp.join(out_folder, f'{vn}.npy')
            np.save(out_npy, feats)
            extracted += 1
        
        log.info(f"Extracted audio features 4 {extracted} videos in {(time.time() - tic):.2f} seconds.")
    
    
    def _compare_fog(self, vn, folder_name, nclips, source='VGGISH'):
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
            og_nclips = np.load(matching_files[0]).shape[0]
            if og_nclips != nclips: 
                log.error(f"mismatch betwen OG {og_nclips} and new {nclips}")
                return -1
            else: return 1
        else:
            log.warning(f"No matching RGB feature file found for {vn}")
            return 1