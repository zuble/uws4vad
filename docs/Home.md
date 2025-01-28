

## Pipeline 

The overall pipeline adapts most commom blocks used in WVAD methods, as well defined class/fx's to enable combination and experimentation.
 
<p align="center"><img src="img/vadpipe4.png" width="100%" alt='vadpipe'> </p>

- **Segmentation** transforms each video clip-level feature into the desirable fixed-length number of feature segments, which are input to the network during training.  

- **Segment Selection** aims to prioritise relevant segments to be used in the training, under a specific selection criteria as confident cues to select and optimize (e.g. L2 norm, more commonly known as feature magnitude, Divergence of Feature from Mean (DFM), generated scores from the regressor network,..). The selection can be made either from a sample pool of all video segments (static top-k) or a batch gathering of both abnormal and normal class segment videos (dynamic SBS).


## Data

| Videos |  FRGB   | FAUDIO  |
|---------|--------|------|
| [UCF](https://www.crcv.ucf.edu/projects/real-world/) | [I3DROCNG](https://github.com/Roc-Ng/DeepMIL) <br> [CLIPTSA](https://github.com/joos2010kj/CLIP-TSA) | - |
| [XDV](https://roc-ng.github.io/XD-Violence/) | [VADCLIP](https://github.com/nwpu-zxr/VadCLIP) <br> [CLIPTSA](https://github.com/joos2010kj/CLIP-TSA) | [I3DROCNG (CNNAED_VGG-E)](https://roc-ng.github.io/XD-Violence/) |


Under the set folder at *path.data_root* the following `tree -d` is excepted

```bash
DS ## videos
├── UCF
│   ├── TEST
│   └── TRAIN
└── XDV
    ├── TEST
    └── TRAIN
FEAT ## features
├── UCF
│   └── RGB 
│       ...
│       └── I3DROCNG ## each folder name match the value of data.frgb.id
│           ├── TEST
│           └── TRAIN
└── XDV
    ├── AUD 
    │   ...
    │   └── VGGISH ## each folder name match the value of data.faud.id
    │       ├── TEST
    │       └── TRAIN
    └── RGB 
        ...
        └── CLIPTSA ## each folder name match the value of data.frgb.id
            ├── TEST
            └── TRAIN
```


