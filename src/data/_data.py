import torch
import numpy as np, random

import glob , os, os.path as osp, math, time, itertools
from collections import OrderedDict, Counter, defaultdict
import matplotlib.pyplot as plt

from src.utils import mp4_rgb_info, logger
log = logger.get_log(__name__)


###########
## DEBUGGER
def run_dl(dl, iters=2, vis=False):
    """
    """
    import matplotlib.pyplot as plt
    try:
        for ii in range(iters):
            tic = time.time()
            log.info(f"Iter {ii + 1} / {iters}")
            log.info(f"cfeat | seqlen | label")
            epo_counts = []  # Store counts for each class across batches
            epo_class_counts = [] # Store all the counts in the epoch

            for b_idx, (cfeat, seqlen, label) in enumerate(dl):  # Unpack seqlen here
                log.debug(f'B[{b_idx}] (seq) {cfeat.shape=} {cfeat.dtype} | {label=} {label.shape=} {label.dtype} | {seqlen.shape=} {seqlen.dtype}')
                
                ################3
                ## seqlen batch calculues
                #og_sl = torch.sum(torch.max(torch.abs(cfeat[:,0,:,:]), dim=2)[0] > 0, 1)
                #assert torch.all(og_sl.eq(seqlen)) == True, f"{og_sl.shape} != {seqlen.shape}"
                #
                #inputs = cfeat[:, :torch.max(seqlen), :]
                #assert torch.all(inputs.eq(cfeat)) == True, f"{inputs.shape} != {cfeat.shape}"
                #log.error(f"{inputs.shape=} {torch.max(seqlen)=}")
                ##################
                
                bat_counts = label.tolist() if isinstance(label, torch.Tensor) else label
                log.info(bat_counts)
                # Calculate and log per-batch statistics
                bat_dist = Counter(bat_counts)
                batch_total = len(bat_counts)
                log.info(f"B[{b_idx + 1}] {bat_dist}  "
                        f"({', '.join([f'{class_id}:{count} ({(count / batch_total) * 100:.2f}%)' for class_id, count in sorted(bat_dist.items())])})")

                # Append counts for each class to epo_counts 
                for class_id in bat_dist:
                    while len(epo_counts) <= int(class_id):
                        epo_counts.append([])  # Add lists for new classes
                    epo_counts[int(class_id)].append(bat_dist[class_id])
                epo_class_counts.extend(bat_counts)

                log.debug(f'[{ii}] time to load {b_idx} batches {time.time()-tic}')
            
            # --- Log Class Distribution Summary ---
            total_samples = len(epo_class_counts)
            class_distribution = Counter(epo_class_counts)
            log.info("Epoch Class Distribution:")
            for class_id, count in sorted(class_distribution.items()):
                log.info(f"  Class {class_id}: {count} samples ({((count / total_samples) * 100):.2f}%)")
                
    except ValueError as e: 
        log.error(f'Error unpacking variables in batch {b_idx}: {e}')



def analyze_sampler(bsampler, labels, dataset_name, iters=2, vis=True):
    """Analyzes BatchSampler behavior over multiple iterations."""
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    
    for ii in range(iters):
        all_indices = []
        epo_counts = []  # Store counts for each class across batches
        epo_class_counts = [] # Store all the counts in the epoch
        
        for batch_idx, batch_indices in enumerate(bsampler):
            all_indices.extend(batch_indices)

            bat_counts = [labels[idx] for idx in batch_indices]  # Get labels for the batch
            bat_dist = Counter(bat_counts)
            for class_id in bat_dist:
                while len(epo_counts) <= int(class_id):
                    epo_counts.append([])
                epo_counts[int(class_id)].append(bat_dist[class_id])
            epo_class_counts.extend(bat_counts)

        print(f"\n\n\t  Dataset: {dataset_name}, Epoch: {ii + 1}")
        print(f"\t    Samples per Epoch: {len(all_indices)}")

        # --- Log Class Distribution Summary ---
        class_distribution = Counter(labels[idx] for idx in all_indices)
        total_samples = len(all_indices)
        print(f"\t    Epoch Class Distribution")
        for class_id, count in sorted(class_distribution.items()):
            print(f"\t  Class {class_id}: {count} samples ({((count / total_samples) * 100):.2f}%)")

        #####################################
        #unique_indices = set(all_indices)
        #repeated_indices = len(all_indices) - len(unique_indices)
        #print(f"\t    Repeated Indices: {repeated_indices}")
        ## Analyze repeated indices per class
        index_counts = Counter(all_indices)
        repeated_indices_per_class = defaultdict(list)
        for idx, count in index_counts.items():
            if count > 1:
                label = labels[idx]
                repeated_indices_per_class[label].append(idx)
        print("\t    Repeated Indices per Class:")
        for label, indices in sorted(repeated_indices_per_class.items()):
            print(f"\t      Label {label}: {len(indices)} ") #indices ({indices})
        #####################################

        # --- Visualization ---
        if vis:
            num_batches = len(epo_counts[0])
            for class_id, class_counts_per_batch in enumerate(epo_counts):
                plt.plot(range(1, num_batches + 1), class_counts_per_batch, label=f'Class {class_id}')
            plt.xlabel('Batch Number')
            plt.ylabel('Number of Samples')
            plt.title(f'Per-Batch Class Distribution (Epoch {ii + 1})')
            plt.legend()
            plt.show()



def debug_cfg_data(cfg):
    """
    """
    #if 'dltrain' not in cfg.debug.tag: return
    
    ID_DL = cfg.data.id
    
    log.debug("DEBUG CFG DATA RGB")
    cfg_frgb = cfg.data.frgb
    ID_RGB = cfg_frgb.id
    assert cfg.data.id in cfg_frgb.ds  
    
    ########## 
    ## RGB
    MODE = "TRAIN"
    ## TRAIN TEST FOLDERS
    root_rgb = cfg.data.froot+"/RGB/"+ID_RGB
    assert osp.exists(root_rgb), f"{root_rgb} does not exist"
    assert osp.exists(f"{root_rgb}/{MODE}")
    assert osp.exists(f"{root_rgb}/TEST")
    #log.debug(f"[{ID_DL}] {root_rgb} w/ TRAIN + TEST folders")
    
    ## train
    dbg_train_fps = glob.glob(f"{root_rgb}/{MODE}/*.npy")
    dbg_train_fns = [ osp.splitext(osp.basename(p))[0] for p in dbg_train_fps]
    dbg_train_fns.sort()
    log.debug(f"[{ID_DL}] found {len(dbg_train_fns)} .npy files {dbg_train_fns[0]}")
    
    tmp = cfg.path.data_dir+"gt/"
    with open(tmp+f"{cfg.data.id}_flist_train.txt", 'r') as txt: data = txt.read()
    train_fns = [line for line in data.split('\n') if line]
    train_fns.sort()
    log.debug(f"[ORIGINAL LEN {len(train_fns)} {train_fns[0]}")
    
    ## NUMBER OF .npy FEATS NCROPS ASSERT
    if cfg_frgb.ncrops:
        
        if len(dbg_train_fns) != len(train_fns) * cfg_frgb.ncrops:
            dbg_train_unqs = list(OrderedDict.fromkeys([f[:-3] for f in dbg_train_fns]))
            
            ## check miss file
            notin = [ f for f in train_fns if f not in dbg_train_unqs]
            if len(notin):
                log.error(f"[{ID_DL}] {len(notin)} .npy files not in {ID_RGB} train")
            
            ## check miss crop
            for idx, train_fn in enumerate(train_fns):
                
                for crop in range(cfg_frgb.ncrops):
                    this = f"{train_fn}__{crop}.npy" 
                    assert osp.exists(f"{root_rgb}/{MODE}/{this}"), f"{this} does not exist"
                    
        if cfg.dataproc.crops2use.train == 0: ## aslong as itsused the dyn_crops2use no error here (def in datatrnsfrm._globo)
            log.error(f"[{ID_DL}] wrong {cfg.dataproc.train.crops2use=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception 
        
        #assert len(dbg_train_fns) == (cfg.data.train.normal + cfg.data.train.abnormal) * cfg_frgb.ncrops
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg.data.froot}/RGB/{MODE}/{ID_RGB}")
        
        #if cfg.dataproc.crops2use.test == 0:
        #    log.error(f"[{ID_DL}] wrong {cfg.dataproc.crops2use.test=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
        #    raise Exception
        #assert len(dbg_test_fns) == (cfg.data.test.normal + cfg.data.test.abnormal) * cfg_frgb.ncrops    
        #log.debug(f"[{ID_DL}] {ID_RGB}.NCROPS match number of files in {cfg.data.froot}/RGB/TEST/{ID_RGB}")
        
    else:
        
        #assert len(dbg_train_fns) == (cfg.data.train.normal + cfg.data.train.abnormal)
        if len(dbg_train_fns) != len(train_fns):
            
            notin = [ f for f in train_fns if f not in dbg_train_fns]
            if len(notin): 
                log.warning(f"[{ID_DL}] {len(notin)} missing file in {ID_RGB}train")
                log.warning(notin)
        else: 
            log.debug(f"[{ID_DL}] {ID_RGB} match number of files in {cfg.data.froot}/RGB/{MODE}/{ID_RGB}")
        
        if cfg.dataproc.train.crops2use > 0: ## aslong as itsused the dyn_crops2use no error here (def in datatrnsfrm._globo)
            log.error(f"[{ID_DL}] wrong {cfg.dataproc.train.crops2use=} while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
            raise Exception 
        
        ## always used one per now
        #if cfg.dataproc.crops2use.test > 0:
        #    log.error(f"[{ID_DL}] wrong {cfg.dataproc.crops2use.test=}  while {ID_RGB}.NCROPS {cfg_frgb.ncrops}")
        #    raise Exception
        
        if cfg.dataproc.cropasvideo: 
            log.error(f"[{ID_DL}] {cfg_frgb.ncrops=} while cropasvideo is True")
            raise Exception 
        
        
    ## FEAT LEN ASSERT
    log.debug(f"\nFEAT LEN ASSERT")
    tmp = np.load(dbg_train_fps[random.randint(0,len(dbg_train_fps)-1)])
    assert tmp.shape[1] == cfg_frgb.dfeat
    log.debug(f"[{ID_DL}] {cfg_frgb.dfeat=} match {tmp.shape[1]}")
    
    ## FEATUREATHLISTFINDER
    log.debug(f"\nFEATUREATHLISTFINDER")
    rgbfplf = FeaturePathListFinder(cfg, 'train', 'rgb')
    rgbfltrain = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    log.debug(f'{MODE}: RGB {len(rgbfltrain)} feats  {rgbfltrain[0]}')
    
    if cfg.data.get("faud"):
        log.debug(f"\nAUD")
        audfplf = FeaturePathListFinder(cfg, 'train', 'aud', auxrgbflist=rgbfltrain)
        aaudfl, naudfl = audfplf.get('ANOM'),  audfplf.get('NORM')
        log.debug(f'{MODE}: AUD on {len(naudfl)} normal, {len(aaudfl)} abnormal')    
        
        #raise NotImplementedError
    
    
    
    '''
    ## test
    dbg_test_fps = glob.glob(f"{root_rgb}/TEST/*.npy")
    dbg_test_fns = [ osp.basename(p) for p in dbg_test_fps]
    dbg_test_fns.sort()
    log.debug(f"[{ID_DL}] found {len(dbg_test_fns)} .npy files")
    
    with open(tmp+f"{cfg.data.id}_flist_test.txt", 'r') as txt: data = txt.read()
    test_fns = [ line for line in data.split('\n') if line ]
    test_fns.sort()
    log.debug(f"ORIGINAL LEN {len(test_fns)}")
    
    ## FEAT LEN ASSERT
    #assert len(dbg_test_fns) == (cfg.data.test.normal + cfg.data.test.abnormal)
    tmp = np.load(dbg_test_fps[random.randint(0,len(dbg_test_fps))])
    assert tmp.shape[1] == cfg_frgb.dfeat
    log.debug(f"[{ID_DL}] {cfg_frgb.dfeat=} match test .npy files")
    
    rgbfplf = FeaturePathListFinder(cfg, 'test', 'rgb')
    rgbfltest = rgbfplf.get('ANOM') + rgbfplf.get('NORM')
    log.debug(f'TEST: RGB {len(rgbfltest)} feats  {rgbfltest[0]}')
    
    '''
    

####################
## PATHS AND SUCH
class FeaturePathListFinder:
    """
        From cfg_data.ds.info.froot/cfg_feat.id/ finds a folder with mode in it (train / test)
        then based on cfg_data procedes to filter the features paths
        so it retrieves accurate features list to use
        used in data/get_trainloader && data/get_testloader
    """
    def __init__(self, cfg, mode:str, modality:str, auxrgbflist:list = None):
        self.listANOM , self.listNORM = [], []
        
        cfg_lbls = cfg.data.lbls
        crops2use = cfg.dataproc.crops2use.get(mode)
        cfg_feat = cfg.data.get(f"f{modality}")

        mode = mode.upper()
        modality = modality.upper()
        
        ID_FEAT = cfg_feat.id
        fpath = cfg.data.froot+'/'+modality+"/"+ID_FEAT+"/"+mode
        flist = glob.glob(fpath + '/*.npy')
        if not len(flist): 
            raise Exception (f'{fpath} has NADA')
        flist.sort()
        log.debug(f" {mode} {modality} feat flist pre-filt {len(flist)}")
        
        #################        
        ## if cropasvideo
        ##      if train, 
        ##          if cfg_data.ds.frgb.ncrops == to the amount of crops in the flist 
        ##              all features will treatead as a video so leave all fn in list
        ##          if != load only correspondant crop based cfg_data.trnsfrm.train.crops2use , 
        ##              eg if cfg_data.ds.frgb.ncrops == 1 and the flist contains 5 crops for each vid, form flist with flist excluding the __1.npy __2.npy __3.npy __4.npy crops for each video 
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## elif cfg_data,DATA.ncrops is set means that features files have crops even if only 1 is used
        ##      if train, take unique basename from crop fns -> feed each crop feat -> mean scores over crop dimension
        ##      if test, take unique basename from crop fns -> it'll use only center crop __0.npy
        ## else means that all files are a feature of video full view

        if modality == 'RGB' and mode == 'TRAIN':

            if cfg.dataproc.get("cropasvideo"):
                
                if crops2use == cfg_feat.ncrops: ## get full list w/o ".npy"
                    flist = [f[:-4] for f in flist] 
                else: ## get only rigth crop idx
                    flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                    flist = [f"{f}__{i}" for f in flist for i in range(cfg.dataproc.crops2use.train)]
            
            elif crops2use: ## >= 1
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
            else: ## goes without .npy
                flist = [osp.splitext(f)[0] for f in flist]
                
        elif modality == 'RGB' and mode == 'TEST': 
            
            if crops2use == 1: ## == 1
                ##feature fn from features crop folder without duplicates (__0, __1...) 
                flist = list(OrderedDict.fromkeys([osp.splitext(f)[0][:-3] for f in flist]))
                
            else: ## no crops -> wo.npy
                flist = [f[:-4] for f in flist]
        
        ## aud 1 .npy per vid
        ## acepts an additional rgblist to have matched multimodal pairs
        else: 
            flist = [f[:-4] for f in flist] ## wo .npy
            log.debug(f"AUD FLIST {len(flist)}  {flist[0]}")
            ## /mnt/t77/FEAT/XDV/AUD/VGGISH/TRAIN/A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A
            
            if len(flist) != len(auxrgbflist):
                if not auxrgbflist: raise Exception
                log.debug(f" auxerrgbflist {len(auxrgbflist)}")
                ## /mnt/t77/FEAT/XDV/RGB/CLIPTSA/TRAIN/Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
                
                def core_filename(path):
                    return osp.splitext(osp.basename(path))[0]
                
                auxrgb_filenames = set(core_filename(f) for f in auxrgbflist)
                
                # Filter audflist to include only those items with a core filename that exists in auxrgb_filenames
                flist = [f for f in flist if core_filename(f) in auxrgb_filenames]
                log.warning(f" {mode} {modality} flist_filtered {len(flist)} ") #{flist[0]}
                

        log.debug(f"{mode} {modality} feat flist pos-filt {len(flist)}")
        #log.warning(f"{flist}")
        
        ######
        ## filters into anom and norm w/ listNORM and listANOM
        ## filters for watching purposes w/ fn_label_dict
        self.fn_label_dict = {lbl: [] for lbl in cfg_lbls.info[:-1]}
        fist = list(self.fn_label_dict.keys())[0]  ## 000.NORM
        last = list(self.fn_label_dict.keys())[-1] ## 111.ANOM

        for fp in flist:
            fn = osp.basename(fp)
            
            ## v=Z12t5h2mBJc__#1_label_B1-0-0
            if 'label_' in fn: xearc = fn[fn.find('label_'):]
            else: xearc = fn
            
            #log.debug(f'{fp} {xearc} ')
            
            if cfg_lbls.id[0] in xearc: ## Assault018
                self.listNORM.append(fp)
                self.fn_label_dict[fist].append(fn)
                #log.debug(f'{xearc} {fn}')
            else:
                self.listANOM.append(fp)
                self.fn_label_dict[last].append(fn)

                for key in self.fn_label_dict.keys():
                    if key in xearc: 
                        self.fn_label_dict[key].append(fn)
                        #log.debug(f'{key} {fn}')
                        
        #for label, lst in self.fn_label_dict.items(): 
        #    log.debug(f'[{label}]: {len(self.fn_label_dict[label])}  ') ##{self.fn_label_dict[label]}
                
                
    def get(self, mode, watch_list=[]):
        if mode == 'ANOM': l = self.listANOM
        elif mode == 'NORM': l = self.listNORM
        elif mode == 'watch': 
            l = []
            for lbl2wtch in watch_list:
                ## find the labels that match the ones provided which need to be the nummberss prior to dot/.
                lbl2wtch = [lbl for lbl in list(self.fn_label_dict.keys()) if lbl.split('.')[0] == lbl2wtch][0]
                log.debug(f'{lbl2wtch} {len(self.fn_label_dict[lbl2wtch])}')
                l.extend(self.fn_label_dict[lbl2wtch])
        else: log.error(f'{mode} not found')
        return l
    


##############
## XDV/UCF DS
def get_testxdv_info(txt_path):
    txt = open(txt_path,'r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    for vidx in range(len(video_list)):
        log.info(video_list[vidx])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[vidx])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[vidx][nota_i])
                dif_aux = aux2-int(video_list[vidx][nota_i-1])
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
        log.info(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[vidx][-1]),'max anom frame\n')
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    log.info("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")

def get_xdv_stats():
    folders = {"train":"/raid/DATASETS/anomaly/XD_Violence/training_copy", "test":"/raid/DATASETS/anomaly/XD_Violence/testing_copy"}
    for key, folder in folders.items():
        paths = glob.glob(f"{folder}/*.mp4")
        total = [0,0] ## normal , anom
        for p in paths:
            dur, tframes, fps = mp4_rgb_info(p)
            if "label_A" in p: total[0] += tframes
            else: total[1] += tframes
            
        log.info(f"XDV {key} STATS")
        log.info(f"{len(paths)} videos")
        log.info(f"TOTAL NORMAL FRAMES: {total[0]} {(total[0]/(total[0]+total[1]))*100:2f}%")
        log.info(f"TOTAL ANOMALY FRAMES: {total[1]} {(total[1]/(total[0]+total[1]))*100:2f}% \n\n")

def get_ucf_stats():
    folders = {"train":"/raid/DATASETS/anomaly/UCF_Crimes/DS/train", "test":"/raid/DATASETS/anomaly/UCF_Crimes/DS/test"}
    for key, folder in folders.items():
        paths = glob.glob(f"{folder}/*.mp4")
        total = [0,0] ## normal , anom
        for p in paths:
            dur, tframes, fps = mp4_rgb_info(p)
            if "Normal" in p: total[0] += tframes
            else: total[1] += tframes
            
        log.info(f"UCF {key} STATS")
        log.info(f"{len(paths)} videos")
        log.info(f"TOTAL NORMAL FRAMES: {total[0]} {(total[0]/(total[0]+total[1]))*100:2f}%")
        log.info(f"TOTAL ANOMALY FRAMES: {total[1]} {(total[1]/(total[0]+total[1]))*100:2f}%\n\n")

