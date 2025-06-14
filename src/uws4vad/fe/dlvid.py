import numpy
import torch
from torch.utils.data import DataLoader, Dataset

import decord
from decord import VideoReader, cpu
# decord.bridge.set_bridge('torch')

import cv2
import os, os.path as osp, glob, time, random

from uws4vad.utils import get_log
log = get_log(__name__)

## it might come handy
## https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/preprocess/compress.py


def get_vidloader(cfg, trnsfrm, cfg_model, vpaths):
    #cfg_loader = cfg.dataload.test
    return DataLoader(
        VideoDS(cfg, trnsfrm, cfg_model, vpaths),
        batch_size=1,
        shuffle=False,
        num_workers=5,  # cfg_loader.nworkers,
        pin_memory=False,
    )


class VideoDS(Dataset):
    def __init__(self, cfg, trnsfrm, cfg_model, vpaths):
        self.cfg = cfg
        self.cfg_model = cfg_model
        self.trnsfrm = trnsfrm

        self.vpaths = vpaths
        assert len(self.vpaths) > 0, "No video found in the provided paths"
        log.info(f"{len(self.vpaths)=}")

        # self.records = [VideoRecord(vp, self.cfg_model) for vp in self.vpaths]
        # log.info(f"{len(self.records)=}")

    def _get(self, vrec):
        ## actually decodes frames
        clips = []
        for k, cidx in enumerate(vrec.cidxs_bat):
            # log.warning(f"clip[{k}] frame[{cidx}]")

            if len(cidx) == 1:  ## means that clip only has 1 frame idx
                frame = vrec.vid[cidx[0]].numpy()  ## (H, W, 3)
                log.debug(f"dcord[{k}] {frame.shape} {frame[0].dtype} {type(frame)}")

                # frame = to_pil_image(frame.permute(2,0,1), 'RGB') ## -> c * h * w - expected per ToPILImage
                # log.debug(f'pil[{k}] {type(frame)}')

                frame = self.trnsfrm(frame)
                log.debug(f"trnsfrm[{k}] {frame.shape} {frame[0].dtype} {type(frame)}")

                # if cfg.viewer: view_crops_trnsf(frame,self.cfg_model.ncrops)
                # clips.extend(frame)
                clips.append(frame)

            else:  ## normal input to cnn-based
                frames = vrec.vid.get_batch(cidx)  ## (new_length, H, W, 3)
                log.debug(f"dcord[{k}] {frames.shape} {frames[0].dtype} {type(frames)}")
                ## not tested yet.....

        return torch.stack(clips, dim=0)
        # return clips

    """
    ## missin files : 
    v=8cTqh9tMz_I__#1_label_A 
    
    v=9eME1y6V-T4__#01-12-00_01-18-00_label_A
        https://github.com/dmlc/decord/issues/145
        decord._ffi.base.DECORDError: 
        [02:55:19] /github/workspace/uws4vad/video/ffmpeg/threaded_decoder.cc:292: 
        [02:55:19] /github/workspace/uws4vad/video/ffmpeg/threaded_decoder.cc:218: 
            Check failed: avcodec_send_packet(dec_ctx_.get(), pkt.get()) >= 0 (-1094995529 vs. 0) 
            Thread worker: Error sending packet
    """

    def __getitem__(self, idx):
        p = self.vpaths[idx]
        log.debug(p)

        # vrec = self.records[idx]
        vrec = VideoRecord(self.vpaths[idx], self.cfg_model)
        if vrec.vid is None or "v=9eME1y6V-T4__#01-12-00_01-18-00_label_A" in p:
            clips = -1
            safe = 0
        else:
            clips = self._get(vrec)
            safe = 1

        return clips, vrec.vname, safe

    def __len__(self):
        return len(self.vpaths)


class VideoRecord:
    def __init__(self, vpath, cfg_model):
        self.vpath = vpath
        self.vname = osp.splitext(osp.basename(vpath))[0]
        log.info(f"{'*' * 4} {self.vname} {'*' * 4}")

        ###############
        # vid2 = cv2.VideoCapture(vpath)
        # if not vid2.isOpened(): raise ValueError(f"Failed to open video {vpath}")
        # self.fps = int(vid2.get(cv2.CAP_PROP_FPS))
        # self.tframes = int(vid2.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.w = vid2.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.h = vid2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # vid2.release()
        # log.info(f'{str(self.fps)} fps | {str(self.tframes)} frames {self.h}*{self.w}')
        ##############
        try:
            # self.vid = VideoReader(vpath, width=self.w, height=self.h, num_threads=1, ctx=cpu(0))
            self.vid = VideoReader(vpath, num_threads=1, ctx=cpu(0))  #
            # log.warning(self.vid.get_avg_fps())
            # log.warning(len(self.vid))
            self.vid_len = len(self.vid)
            # assert self.vid_len == self.tframes, f'{self.vid_len} != {self.tframes}'

            self.frame_step = cfg_model.frame_step
            self.clip_len = cfg_model.clip_len
            self.frame_sel = cfg_model.frame_sel  ## CLIP only

            self.get_vidxs()
        except:
            self.vid = None

    def get_vidxs(self):
        vidxs = list(
            range(0, self.vid_len, self.frame_step)
        )  ## grab vidxs by frame step
        if self.vid_len < self.clip_len:
            r = self.clip_len // self.vid_len + 1
            vidxs = (vidxs * r)[: self.clip_len]

        nclips = int(len(vidxs) / self.clip_len)
        log.info(
            f"from {len(vidxs)} yielding {nclips * self.clip_len} frames in {nclips=} {self.clip_len=}@{self.frame_step})"
        )

        if nclips * self.clip_len < len(vidxs):
            log.info(f"discarting {len(vidxs) - (nclips * self.clip_len)} ")

        self.cidxs_bat = []
        for k in range(nclips):
            cidxs = vidxs[k * self.clip_len : (k + 1) * self.clip_len]
            log.debug(f"[{k}]  pre {cidxs}")

            if self.frame_sel == "rnd":  ## grab random from each clip
                tmp = [cidxs[random.randint(0, len(cidxs) - 1)]]
            elif self.frame_sel == "mid":  ## grab middle frame from each clip
                tmp = [cidxs[(len(cidxs) // 2) - 1]]
            else:  ## rest need full clip as inpt
                tmp = cidxs

            self.cidxs_bat.append(tmp)
            log.debug(f"[{k}]  pst {tmp}")
