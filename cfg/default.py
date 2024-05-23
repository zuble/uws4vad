## NOTES 
## theres a few caveats here, mainly because of how the yacs deals with cfg
## using Hydra would be the best solution
from yacs.config import CfgNode as CN

_C = CN()
## number of gpus to use. Use -1 for CPU
_C.GPUID = [3]
_C.GPUSETON = False ## sets all gpus in os var CUDA_VISIBLE_DEVICES
## if None and train: generates trough os see more utils/misc.py
## else: use seed at fn .pkl 
_C.SEED = None
_C.DETERMINISTIC = False

#########
## root dir to save logs /ckpt / model
#_C.EXPERIMENTDIR = '/media/jtstudents/T77/TH/zutc_vad/.params'
#_C.EXPERIMENTDIR = '/mnt/t77/TH/zutc_vad/.params'
_C.EXPERIMENTDIR = ''
## based on fn.yml, a fn folder will be created in the experiment dir (see utils/setup.py)
_C.EXPERIMENTID = ''
_C.EXPERIMENTPROJ = ''
_C.EXPERIMENTPATH = ''
## wheter to log cfg after merge with default
_C.LOG_CFG_INFO = True

## quick experimmt: jumps train/test call in main and runs tmp.py
## better use with --exp cfg/.tmp.yml
## it has a cfg with all parameters to experiment
_C.TMP = False 
_C.PROFILE = 0 ## epochs to stop/dump mxnet.profiler
_C.FCOMP = False

##############################
## DEBUG
## 0: logging.INFO, 1: logging.DEBUG
## controls the max level for both console and file logger
_C.DEBUG = CN()
## sets debug root/master level for both logger handlers
_C.DEBUG.ROOT = 0
## sets log level for modules 
_C.DEBUG.DATA = 0
_C.DEBUG.LOSS = 0
_C.DEBUG.METRIC = 0
_C.DEBUG.MISC = 0
_C.DEBUG.NETS = 0
_C.DEBUG.TEST = 0
_C.DEBUG.TRAIN = 0
_C.DEBUG.VLDT = 0
_C.DEBUG.TMP = 1


#############
## NETWORK 
## it serves both train/test
_C.NET = CN()
_C.NET.LOG_INFO = ['blck'] ## blck / prmt
_C.NET.WGHT_INIT = 'xavier' ## weights~xavier / bias~0
###############
## keep consistent cfg.NET entrie with nets/filename.py
## so @ _nets/get_net automatcly loads nets/{NET.NAME}/Network 
## if .VERSION present loads nets/{.VERSION}/Network 
## otherwise its excepted each net.py to have the main module as Network
_C.NET.NAME = ''

## nets/attnomil
_C.NET.ATTNOMIL = CN()
_C.NET.ATTNOMIL.VERSION = 'VCls' ## VCls , SAVCls , SAVCls_lstm
_C.NET.ATTNOMIL.DA = 64
_C.NET.ATTNOMIL.R = 3
_C.NET.ATTNOMIL.DROP_RATE = 0.3
_C.NET.ATTNOMIL.CLS_NEURONS = [32,1]
_C.NET.ATTNOMIL.LSTM_DIM = 0 ## SAVCls_lstm set to 0 for no use
## if true softmax over the feature axis (normalizing across the feats)
## false over the temporal axis (normalizing across the time steps)
_C.NET.ATTNOMIL.SPAT = False 
## used by Vldt/Validate when in test 
## and if 'attws' in TEST.WATCH.FRMT
## will be auto set prior to model creation 
## thus enable the model forward to return attention weights
## not used in train
_C.NET.ATTNOMIL.RET_ATT = False
## controls the split size to divide the input feats to achieve finner temporal precision
## divides N feature vectors into m (= N/VLDT_SPLIT) bags during inference, each bag contains VLDT_SPLIT feature vectors.
## affects vldt/Validade/_forward_attnomil
_C.NET.ATTNOMIL.VLDT_SPLIT = 9 ## 0 could one global score (not implemented)

## nets/claws
_C.NET.CLAWS = CN()

## nets/cmala
_C.NET.CMALA = CN()
_C.NET.CMALA.CLS_VRS = 'CONV' ## MLP CONV LSTM

## nets/mindspore
_C.NET.MINDSPORE = CN()
_C.NET.MINDSPORE.CLS_VRS = 'MLP' ## MLP CONV LSTM
## seg/seq len impact
#T    32      64      100     200     320     400 
#UCF 85.24   84.93   84.40   84.75   86.19   85.49 
#XD  78.91   80.68   83.59   83.31   83.17   81.53

## nets/dtr
_C.NET.DTR = CN()
_C.NET.DTR.DTR_NEURONS = [512,128]
_C.NET.DTR.CLS_VRS = 'MLP' ## MLP CONV LSTM

## nets/rtfm
_C.NET.RTFM = CN()
_C.NET.RTFM.CLS_VRS = 'MLP' ## MLP CONV LSTM


########
## classfication/head netowrks
## each main net can have one of those by specifing CFG.NET.***.CLS
_C.NET.CLS = CN()
## this defaults are the same as class parameters
_C.NET.CLS.MLP = CN()
_C.NET.CLS.MLP.NEURONS = [512,32]
_C.NET.CLS.MLP.ACTIVA = 'relu'
_C.NET.CLS.MLP.DO = 0.7

_C.NET.CLS.CONV = CN()
_C.NET.CLS.CONV.KS = 7

_C.NET.CLS.LSTM = CN()
_C.NET.CLS.LSTM.HIDDIM = 256
_C.NET.CLS.LSTM.BD = False



################
## TRAIN
_C.TRAIN = CN()
_C.TRAIN.ENABLE = True
_C.TRAIN.EPOCHS = 1
_C.TRAIN.BS = 1
_C.TRAIN.EPOCHBATCHS = 0 ## batch per epoch assigneded in data.py after get len ds
_C.TRAIN.DROPLAST = True ## 4 BatchSampler, True otherwise irregular length

#######
## following one of the provided _C.DS
## ds.modality.ftype  ;  0:rgb 1:aud
## therefore loading rgb/aud features and the gt file from it
_C.TRAIN.DS = ['UCF.RGB.CLIPTSA']


## MUST BE SET RIGHT
## 0 : feats in use must have no crops whatsoever, so fn are in format 'fn.npy'  
## 1 : feats in use have crop augm and only center crop is used, eg. 'fn__0.npy'
## 2 : feats in use have crop augm and center + topright (or wtv) eg. 'fn__0.npy' && 'fn__1.npy' ...
_C.TRAIN.CROPS2USE = 1
_C.TRAIN.CROPASVIDEO = False ## each crop with be treated as a video it self
_C.TRAIN.RGBL2N = 0 ## if 1 l2norm apllied right after load
_C.TRAIN.AUDL2N = 0

## each batch is a equal number of abnormal/normal features
## mainly used per UCF, but can be used per XDV aswell
_C.TRAIN.MIL = True

## determines how to segment/process input features
## so the temporal dimension is static
## it works indepently of MIL
_C.TRAIN.FRMT = 'ITPLT'

## use sultani interpolate
## select seg.len evenly spaced feats from each video nfeats
_C.TRAIN.ITPLT = CN()
_C.TRAIN.ITPLT.LEN = 32 ## interpolation of original feat arr (ts,nfeat) into (nsegments,nfeat)
_C.TRAIN.ITPLT.L2N = 0 ## l2norm 0-none / 1-pre / 2-post segmentation
## jit: add a jitter to the linspace idxs
## glob: rnd temporal order of feat
_C.TRAIN.ITPLT.RND = ['']
## a list of lossfx is created in utils/loss/get_loss
## latter passed in trainep/train_epo to chosen net_pst_fwd.train
_C.TRAIN.ITPLT.LOSS = [''] 

## use mode introduced by xdv DS: XDVioDet@gh
## video features are treated as a fixed size sequence, either padded or truncated
## then each seq len is used to in loss to find the topk, if set 
_C.TRAIN.SEQ = CN()
_C.TRAIN.SEQ.LEN = 200 ## sequence of original feat arr (ts,nfeat) into (nsegments,nfeat)
_C.TRAIN.SEQ.L2N = 0 ## l2norm 0-none / 1-pre / 2-post segmentation
## frist pos of list must be one of these 2: 
#   'uni': chooses idxs as linspace(0, len(feat)-1, SEQ.LEN)
#   'rnd': enables to choose SEQ.LEN features randomly
## jit: rnd betwen adjacent interval of linspace idxs
## glob: rnd temporal order of feat
_C.TRAIN.SEQ.RND = ['uni']
_C.TRAIN.SEQ.LOSS = ['']

#######
## monitor conditions
_C.TRAIN.PLOT_LOSS = True
_C.TRAIN.LOG_PERIOD = 0 ## batch relative , if 0 = EPOCHBATCHS/2
_C.TRAIN.CKPT_PERIOD = 10 ## epoch relative , if 0 = EPOCH/10

########
_C.TRAIN.VLDT = CN()
_C.TRAIN.VLDT.PERIOD = 10 ## epoch relative , if 0 = EPOCH/10
## glob (global) :metrics performed over all normal&&anomalies
## lbl (label) : metrics performed over each anomalie label aswell as all anomalies
_C.TRAIN.VLDT.PER_WHAT = 'lbl'
## if False saves time in vldt by setting MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 before start vldt
_C.TRAIN.VLDT.CUDDNAUTOTUNE = True
## updates lines after each vldt with metrics values (dependant of )
_C.TRAIN.VLDT.VISPLOT = False


########
## https://github.com/sdjsngs/XEL-WSAD/tree/main
## hard hinge loss usable in NetPstFwd at lossfx['xel']
_C.TRAIN.XEL = CN()
_C.TRAIN.XEL.ENABLE = True
_C.TRAIN.XEL.MEM_LEN = 0 ## auto set as cfg.TRAIN.EPOCHBATCHS * (cfg.TRAIN.BS//2)
_C.TRAIN.XEL.HIB_UPD = False ## dont change
_C.TRAIN.XEL.WARMUP = 1 ## epo rel
_C.TRAIN.XEL.MARGSTEP = 10 ## epo rel
_C.TRAIN.XEL.MARGLIST = [0.5,0.6,0.7,0.8,0.9,1.0]
## warmup+(step*6) = epochs
## epochbatchs * (bs//2)
## ucf 101


################
## LOSS
_C.LOSS = CN()
## RNKG (RankingLoss) ( Sultani MIL) with 2 versions deepmil and milbert 
_C.LOSS.RNKG = CN()
## deepmil
## milbert
## Motion-Aware Feature for Improved Video Anomaly Detection
## tempatt (Attention-based Temporal MIL Ranking)
_C.LOSS.RNKG.VERSION = 'deepmil' 
_C.LOSS.RNKG.LAMBDA12 = [8e-5,8e-5]

_C.LOSS.BCE = CN()

## BCE of mean topk/full/wind scores
_C.LOSS.CLAS = CN()
## fx to obtain video level scores from each sequence
_C.LOSS.CLAS.FX = 'topk' ## topk wink full
## will be set to cfg.DATA.RGB.SEGMENTNFRAMES in get_loss
_C.LOSS.CLAS.TOPK = 16
_C.LOSS.CLAS.WINK = 3
_C.LOSS.CLAS.WINKCONV = False

_C.LOSS.RTFM = CN()
_C.LOSS.RTFM.K = 3
_C.LOSS.RTFM.ALPHA = 0.0001
_C.LOSS.RTFM.MARGIN = 100


################
## OPTIMIZER
## https://pytorch.org/docs/1.8.1/optim.html?highlight=optimizer#torch.optim.Optimizer
_C.OPTIMA = CN()
_C.OPTIMA.TYPE = ''

_C.OPTIMA.ADAM = CN()
_C.OPTIMA.ADAM.LR = 0.001
_C.OPTIMA.ADAM.WD = 0.0
_C.OPTIMA.ADAM.BETAS = [0.9, 0.999]
_C.OPTIMA.ADAM.EPS = 1e-8
_C.OPTIMA.ADAM.AMSGRAD = False

_C.OPTIMA.ADAMW = CN()
_C.OPTIMA.ADAMW.LR = 0.001
_C.OPTIMA.ADAMW.WD = 0.01
_C.OPTIMA.ADAMW.BETAS = [0.9, 0.999]
_C.OPTIMA.ADAMW.EPS = 1e-8

_C.OPTIMA.ADAMAX = CN()
_C.OPTIMA.ADAMAX.LR = 0.002
_C.OPTIMA.ADAMAX.WD = 0.0
_C.OPTIMA.ADAMAX.BETAS = [0.9, 0.999]
_C.OPTIMA.ADAMAX.EPS = 1e-8

_C.OPTIMA.SGD = CN()
_C.OPTIMA.SGD.LR = 0.001
_C.OPTIMA.SGD.WD = 0.0
_C.OPTIMA.SGD.MOMENTUM = 0.0
_C.OPTIMA.SGD.DAMPENING = 0.0
_C.OPTIMA.SGD.NESTEROV = False

_C.OPTIMA.ADADELTA = CN()
_C.OPTIMA.ADADELTA.LR = 0.01
_C.OPTIMA.ADADELTA.LR_DECAY = 0.0
_C.OPTIMA.ADADELTA.WD = 0.0
_C.OPTIMA.ADADELTA.EPS = 1e-10

_C.OPTIMA.ASGD = CN()
_C.OPTIMA.ASGD.LR = 0.01
_C.OPTIMA.ASGD.WD = 0.0
_C.OPTIMA.ASGD.LAMBD = 0.0001
_C.OPTIMA.ASGD.ALPHA = 0.75
_C.OPTIMA.ASGD.TI = 100000.0

_C.OPTIMA.ADABEL = CN()
_C.OPTIMA.ADABEL.LR = 1e-3

########
_C.OPTIMA.LRS = CN()
_C.OPTIMA.LRS.VERBOSE = True ## prints a message to stdout for each update
_C.OPTIMA.LRS.TYPE = ''  ## step / multistep / cosanlg 
## Reduce the LR by a FACTOR for every n STEPS.
_C.OPTIMA.LRS.STEP = CN()
_C.OPTIMA.LRS.STEP.SIZE = 1 ## epoch relative, assigned later to cfg.TRAIN.EPOCHSTEPS * itself 
_C.OPTIMA.LRS.STEP.GAMMA = 0.1
## Reduce the LR by a FACTOR for every n STEPS.
_C.OPTIMA.LRS.MULTISTEP = CN()
_C.OPTIMA.LRS.MULTISTEP.MILESTONES = [10,20]
_C.OPTIMA.LRS.MULTISTEP.GAMMA = 0.1
## CosineAnnealingLR
_C.OPTIMA.LRS.COSANLG = CN()
_C.OPTIMA.LRS.COSANLG.T_MAX = 60 
_C.OPTIMA.LRS.COSANLG.ETA_MIN = 0


##############
_C.TEST = CN()
_C.TEST.ENABLE = True
## if test enabled, must be set
## load model from .params/cfg_in_use/this.json/params
## cfg.NET.NAME must be same used 2 gen those params
_C.TEST.LOADFROM = ''
## net: from a torch.save(net) | dict: from a torch.save(net.state_dict())
_C.TEST.LOADMODE = 'net'

## same logic as per train
_C.TEST.DS = ['UCF.RGB.CLIPTSA']
_C.TEST.BS = 1 
_C.TEST.CROPS2USE = 1 ## many use all crops, other dont 
_C.TEST.L2NORM = 0  ## 0 off / 1 on

########
_C.TEST.VLDT = CN()
## saves vldt_info returned per Validate into .pkl format
_C.TEST.VLDT.SAVEPKL = True
## jumps Validate and uses vldt_info.pkl if already generated
_C.TEST.VLDT.FROMPKL = False

## per_what perspectives to show validation results
## glob: one (AP , AU_PRC , AUC_ROC) 4 whole test ds && only anom class
## lbl: one (AP , AU_PRC , AUC_ROC) as previous && each of anom classes
## vid: one (AP , AU_PRC , AUC_ROC) per video, grouped per labels
_C.TEST.VLDT.PER_WHAT = 'lbl'
## further adds the F1 score curve for the different thresholds returned by sklearn.p_r_curve
## and the optimal values of precision and recall based on the thresholds that yields the maximum f1
_C.TEST.VLDT.XTRA_MTRCS = False
## send barplot per metric to visdom server (xaxis is labels)
## useful for compare same metrics per multiples experiments in visdom
_C.TEST.VLDT.MTRC_VISPLOT = True
## sends metrics table as png to visdom server
_C.TEST.VLDT.MTRC_VISTABLE = False
## saves metrics table as png
_C.TEST.VLDT.MTRC_SAVETABLE = False
## sends plotly figs roc/pr curves 2 visdom
_C.TEST.VLDT.CURV_VISPLOT = True
## send GT/FL from (label) ALL videos in test ds
_C.TEST.VLDT.GTFL_VISPLOT = True


########
_C.TEST.WATCH = CN()
## saves watch_info returned per Validate into .pkl format
_C.TEST.WATCH.SAVEPKL = False
## jumps Validate and uses watch_info.pkl if already generated
_C.TEST.WATCH.FROMPKL = False

## vizualitation frontend
## how are the seleted formats watched
## wnshow:  uses opencv.imshow for asp | matplt.show for gtfl/attws
## visdom: sends everything to visdom utils/misc.py/Visualizer
_C.TEST.WATCH.FRTEND = 'visdom'
## FORMAT to watch the results
## []: disabled
## asp (anomaly score player): 
#   video player with GT and anomaly scores overlayed
## gtfl (grount-truth frame-level): 
#   xaxis: frames // yaxis: gt and fl
## attws (attention-weights): colormap xaxis: 
#   frames / yaxis: ATTNOMIL.R att_weights 
#   usable for NET.NAME==attnomil
_C.TEST.WATCH.FRMT =  ['asp', 'gtfl']
## 1 or multiples numbers of selected dataset LBLS_INFO
## eg: UCF abuse and vandalism ['1','13']
## if None selected , it will be prompted after getting the results of validation
_C.TEST.WATCH.LABEL = ['111'] 
## interactively selection of vpath to watch
## enables a terminal diving menu into vldt_info dict
## if false cycles trough videos of seleted WATCH.LABEL
_C.TEST.WATCH.GUIVIDSEL = False

########
_C.TEST.WATCH.ASP = CN()
## frames 2 skip in hte overlayed video to save time :)
_C.TEST.WATCH.ASP.FS = 1
## float: scores in text/numbers
## overlay: half machine other half gt , green-normal / red-abn
_C.TEST.WATCH.ASP.CVPUTFX = 'float'
## anomaly score [0:1[ thrashold to trigger red/green colors when cvputfx is overlay
_C.TEST.WATCH.ASP.TH = 0.5


################
_C.DATA = CN()
## parameters of both train and test data.DataLoader
_C.DATA.LOADIN2MEM = False
_C.DATA.PINMEM = True
_C.DATA.PERSISTWRK = True
_C.DATA.NWORKERS = 1 ## if 0 , = CPU_CORES / 4


####################
## DATASET
_C.DS = CN()
#   holds the default cfg for each of the dataset
#   affeting where to load files 
#   and how they are structered 
## LBLS/LBLS_INFO : 
#   labeling in data/testdataset, process in vldt/VldtInfo // metric/Metrics
#   LBLS: represents the labels ids to look for in filenames
#   LBLS_INFO:  represent the correspondent enconded labels created in data/TestDataset and 
#               further used as keys in both vldt/Validade/VldtInfo and WtchInfo dict
## GT/VROOT : 
#   vldt/GTFL for creation of arrays based on total frames in original videos , test/Watch for watching of test results
## FROOT:
#   feature root path
#   each added entry represents a different feature type holding both train/test features
#   with atleast 2 children folders with '*train*' / '*test*' in its name, e.g.:
#       FROOT/I3DROCNG/RGB/train/...
#       FROOT/I3DROCNG/RGB/test/...
#   moreover @ _data/_data/FeaturePathListFinder
## FSTEP :  number of frames used in each clip when forward trough vision/fe model
##          so this * ts of feature arr = vid tot frames

########
## UCF
_C.DS.UCF = CN()
_C.DS.UCF.LBLS = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
_C.DS.UCF.LBLS_INFO = ['000.NORM', '1.ABUSE', '2.ARREST', '3.ARSON', '4.ASSAULT', '5.BURGLARY', '6.EXPLOS', '7.FIGHT', '8.ROADACC', '9.ROBBER', '10.SHOOT', '11.SHOPLIFT', '12.STEAL', '13.VANDAL', '111.ANOM', 'ALL']
_C.DS.UCF.GT = 'data/gt/ucf.txt' ## del normal lines
_C.DS.UCF.VROOT = '/mnt/t77/DS/UCF/test'#'/raid/DATASETS/anomaly/UCF_Crimes/DS/test'
_C.DS.UCF.FROOT = '/mnt/t77/FEAT/UCF/' #'/raid/DATASETS/anomaly/UCF_Crimes/features/'

#####
_C.DS.UCF.RGB = CN()
##  'I3DDEEPMIL' : https://github.com/Roc-Ng/DeepMIL , 16 fstep , 10 crop
_C.DS.UCF.RGB.I3DROCNG = CN()
_C.DS.UCF.RGB.I3DROCNG.FROOT = ''
_C.DS.UCF.RGB.I3DROCNG.FSTEP = 16
_C.DS.UCF.RGB.I3DROCNG.NFEATS = 1024
_C.DS.UCF.RGB.I3DROCNG.NCROPS = 10
##  'CLIPTSA' : https://github.com/joos2010kj/CLIP-TSA , 8 fstep , nocrop
_C.DS.UCF.RGB.CLIPTSA = CN()
_C.DS.UCF.RGB.CLIPTSA.FROOT = ''
_C.DS.UCF.RGB.CLIPTSA.FSTEP = 8
_C.DS.UCF.RGB.CLIPTSA.NFEATS = 512
_C.DS.UCF.RGB.CLIPTSA.NCROPS = 0
##  'I3DRTFM' : https://github.com/tianyu0207/RTFM

########
## XDV
_C.DS.XDV = CN()
_C.DS.XDV.LBLS = ['label_A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']
_C.DS.XDV.LBLS_INFO = ['000.NORM', 'B1.FIGHT', 'B2.SHOOT', 'B4.RIOT', 'B5.ABUSE', 'B6.CARACC', 'G.EXPLOS', '111.ANOM', 'ALL']
_C.DS.XDV.GT = 'data/gt/xdv.txt'
_C.DS.XDV.VROOT = '/raid/DATASETS/anomaly/XD_Violence/testing_copy'
_C.DS.XDV.FROOT = '/raid/DATASETS/anomaly/XD_Violence/features/'
## search FROOT children folders for this folder name, in its children folders must have 2 folders with ...train.. ...test... in the names
##  i3d-features-rgb : original , 16 fstep , 5 CROP
##  (.mx/)i3d_nl5_resnet50_v1_kinetics400-xdviol_c1
##  (.mx/)i3d_resnet50_v1_kinetics400-xdviol_c1
##  (.mx/)slowfast_4x16_resnet50_kinetics400-xdviol_c5


#_C.DS.XDV.RGBFNAME = ''
### vggish-features : original
### (.tf/)vgg42-tlpf-aps
#_C.DS.XDV.AUDFNAME = ''




# ---------------------------------------------------------------------------- #
def get_cfg(p):
    """
    Get the experiment config.
    """
    cfg = _C.clone()
    cfg.merge_from_file(p)
    return cfg