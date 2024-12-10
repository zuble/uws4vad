

## Pipeline 

<embed src="docs/img/vadpipe4.pdf" type="application/pdf">


## Data

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
│   └── RGB ## each folder name match the value of data.frgb.id
│       ├── CLIPTSA
│       │   ├── TEST
│       │   └── TRAIN
│       └── I3DROCNG
│           ├── TEST
│           └── TRAIN
└── XDV
    ├── AUD ## each folder name match the value of data.faud.id
    │   ├── EFAT_10BSE
    │   │   ├── TEST
    │   │   └── TRAIN
    │   └── VGGISH
    │       ├── TEST
    │       └── TRAIN
    └── RGB ## each folder name match the value of data.frgb.id
        ├── CLIPTSA 
        │   ├── TEST
        │   └── TRAIN
        ├── I3DROCNG
        │   ├── TEST
        │   └── TRAIN
        ├── SLOWFAST_4X16_RN50_K400_C1
        │   ├── TEST
        │   └── TRAIN
        ├── SLOWFAST_4X16_RN50_K400_C5
        │   ├── TEST
        │   └── TRAIN
        ├── TIMM__VITAMIN_BASE_224.DATACOMP1B_CLIP
        │   ├── TEST
        │   └── TRAIN
        └── VADCLIP
            ├── TEST
            └── TRAIN
```
videos ucf [UCFC](https://www.crcv.ucf.edu/projects/real-world/)
features rgb ucf [I3DROCNG](https://github.com/Roc-Ng/DeepMIL)

videos xdv & features rgb xdv & features aud xdv [XDV](https://roc-ng.github.io/XD-Violence/) 
features rgb xdv [VADCLIP](https://github.com/nwpu-zxr/VadCLIP)
features rgb ucf/xdv [CLIPTSA](https://github.com/joos2010kj/CLIP-TSA)
