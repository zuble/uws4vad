import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
import numpy as np

#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#torch.set_default_tensor_type(torch.FloatTensor)

import os, random, time

import nets ## can use nets.dtr.dtr eg
from nets import get_net, save
from utils import LoggerManager, get_optima, Visualizer, hh_mm_ss, mp4_rgb_info
#from vldt import Metrics, Validate 
from data import get_trainloader, get_xdv_stats, get_ucf_stats
from loss import get_loss

log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)

# LOSS / NETS TEST CALLS
def run_net_loss(cfg, dvc):
    #A = ['BCE','RNKG','CLAS', 'MBS']
    #A = ['RNKG']
    #cfg.TRAIN.SEQ.LOSS = A
    lossfx, ldata = get_loss(cfg, dvc)
    for lfx in lossfx:
        log.info(f"{lfx}")
    #ldata = {}
    #lossfx = {}
    
    #A = ['CMA','DTR','ATTNOMIL', 'MINDSPORE', 'ZZZ', 'TAD', 'CLAWS']
    A = ['rtfm'] #RTFM
    
    for a in A:
        cfg.NET.NAME = a
        log.info(f"\n\n{a}\n\n")
        
        net, net_pst_fwd = get_net(cfg, dvc)
        

        bs = cfg.TRAIN.BS
        nc = cfg.DATA.RGB.NCROPS
        seqlen = cfg.TRAIN.SEG.LEN
        rgbnf = cfg.DATA.RGB.NFEATURES
        #feat = torch.randint(0,1,(seqlen,rgbnf),dtype=torch.float32)#.to(dvc)
        feat = torch.randint(0,1,(bs*nc,seqlen,rgbnf),dtype=torch.float32)#.to(dvc)
        
        ######
        ndata = net(feat)
        loss_glob = net_pst_fwd.train(ndata, ldata, lossfx)
        ######
        
        
        
        #if a == 'CLAWS':
        #    cfg.TRAIN.SEQ.LOSS = 'CLAWS'
        #    l, ldata = get_loss(cfg, dvc)
        #    log.info(f"{a} {l}")
        #    loss = l(ndata, ldata)

        
        ######
        for k,v in ndata.items():
            if k != "id": log.info(f"{k} {v.shape}")
            else: log.info(f"{k} {v}")
        log.info(f"\n\n")
        ######
        
class Zuader:
    ## encapsulates the dataloader independtly of elements yielded
    def __init__(self, frmt='SEG', *loaders):
        self.frmt = frmt
        self.loaders = loaders
    def __iter__(self):
        self.loader_iters = [iter(loader) for loader in self.loaders]
        return self
    def __next__(self):
        if self.frmt == 'SEG': return next(self.loader_iters[0]), next(self.loader_iters[1])
        elif self.frmt == 'SEQ': return next(self.loader_iters[0])
        else: raise StopIteration

class RndDS(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = np.arange(0, length, 1)
        #print(f"{self.data = }")
        
    def __getitem__(self, index):
        #print(f"__getitem__ {index}")
        #print(f"{torch.utils.data.get_worker_info()}") 
        #print(f"__getitem__ {np.random.randint(0,20)}")
        tmp = self.data[index] * np.random.randint(0,20) #np.random.randn(100, 100)
        #print(f"{tmp.shape = }")
        return tmp

    def __len__(self):
        return self.len


def main(cfg, dvc):
    
    run_net_loss(cfg, dvc); return
    
    allow_tf32()
    init_seed(22)
    assert torch.initial_seed() == int(os.environ['PYTHONHASHSEED']), f"{torch.initial_seed()} {os.environ['PYTHONHASHSEED']}"
    
    device = torch.device(f'cuda:0')
    
    ds1 = RndDS(10)
    ds2 = RndDS(10)

    maxlends = max(ds1.len,ds2.len)

    
    rsampler = RandomSampler(ds1, replacement=True,  num_samples=maxlends)
    bsampler = BatchSampler(rsampler, batch_size=2 , drop_last=True)
    #for epo in range(8): print(f"{list(bsampler)}")
    loader1 = DataLoader( ds1, batch_sampler=bsampler, 
                            num_workers=8 , worker_init_fn=seed_sade,
                            pin_memory=True , prefetch_factor=2 , persistent_workers=True ) 
    
    #print()
    #rsampler = RandomSampler(ds2)
    #bsampler = BatchSampler(rsampler, batch_size=2 , drop_last=True)
    #for epo in range(8): print(f"{list(bsampler)}")
    #loader2 = DataLoader( ds2, batch_size=3, shuffle=True, drop_last=True)


    ## dataset (Dataset) 
    ##     dataset from which to load the data.
    ## batch_size (int, optional) 
    ##     how many samples per batch to load (default: 1).
    ## shuffle (bool, optional) 
    ##     set to True to have the data reshuffled at every epoch (default: False).
    ## sampler (Sampler or Iterable, optional)
    ##     defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.
    ## batch_sampler (Sampler or Iterable, optional) 
    ##     like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
    ## num_workers (int, optional) 
    ##     how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    ## collate_fn (callable, optional) 
    ##     merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
    ## pin_memory (bool, optional) 
    ##     If True, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
    ## drop_last (bool, optional) 
    ##     set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
    ## timeout (numeric, optional) 
    ##     if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)
    ## worker_init_fn (callable, optional) 
    ##     If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)
    ## prefetch_factor (int, optional, keyword-only arg) 
    ##     Number of samples loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)
    ## persistent_workers (bool, optional) 
    ##     If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: False)

    #loader = Zuader('SEG', loader1)
    tic = time.time()
    for epo in range(20):
        print(f"{epo = }")
        for data in loader1:
            _data = data.to(device)
            #continue
            print(f"{data.dtype} {data}")
            #_data = torch.cat((data), 0)
            #_data = _data.cuda()
            #print(f"{type(_data)} {_data.dtype} {_data}")
    print(f"{time.time() - tic}")


if __name__ == "__main__":
    
    main()