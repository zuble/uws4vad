import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
import numpy as np
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_tensor_type(torch.FloatTensor)

import os, random, time


import nets ## can use nets.dtr.dtr eg
from nets import get_net, save
from utils import LoggerManager, get_optima, Visualizer, hh_mm_ss, FeaturePathListFinder, mp4_rgb_info
#from vldt import Metrics, Validate 
from data import get_trainloader, get_xdv_stats, get_ucf_stats
#from loss import get_loss

log = None
def init():
    global log
    log = LoggerManager.get_logger(__name__)


#########################
## SEED
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.
    https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom( RAND_SIZE )  ## Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def init_seed(seed = None):
    ## https://pytorch.org/docs/1.8.1/notes/randomness.html
    ## https://github.com/henrryzh1/UR-DMU/blob/master/utils.py#L34
    ## https://github.com/Roc-Ng/DeepMIL/blob/master/main.py#L9
    ## https://github.com/Lightning-AI/pytorch-lightning/blob/baeef935fb172a5aca2c84bff47b9b59d8e35b8a/src/lightning/fabric/utilities/seed.py#L37 
    
    print(f'cfg.SEED: {seed}')
    if seed is None: ## cfg.SEED
        seed = os.environ.get('PYTHONHASHSEED', None)
        print(f'PYTHONHASHSEED: {seed}')
        if seed is None: ## os.var
            seed = get_truly_random_seed_through_os()
            print(f'new os.seed: {seed}')
            
    ## change the cfg file itself to the seed or simple 
    print(f'SEEDIT~WEEDIT {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) ## cpu
    ## gpus
    torch.cuda.manual_seed(seed) ## single
    torch.cuda.manual_seed_all(seed) ## multiple
    ## https://pytorch.org/docs/1.8.1/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #return np.random.seed(seed)

def seed_sade(id):
    tmp = torch.initial_seed() % 2**32
    np.random.seed(tmp)
    random.seed(tmp)   


#########################
## allow_tf32
## https://pytorch.org/docs/1.8.1/notes/cuda.html#tf32-on-ampere
def allow_tf32():
    
    a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
    b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
    tic = time.time()
    ab_full = a_full @ b_full
    print(f"time: {time.time() - tic}")
    mean = ab_full.abs().mean()  # 80.7277

    a = a_full.float()
    b = b_full.float()

    # Do matmul at TF32 mode.
    print(f"{torch.backends.cuda.matmul.allow_tf32=} {torch.backends.cudnn.allow_tf32=}")
    tic = time.time()
    ab_tf32 = a @ b  # takes 0.016s on GA100
    print(f"time: {time.time() - tic}")
    error = (ab_tf32 - ab_full).abs().max()  # 0.1747
    relative_error = error / mean  # 0.0022
    print(f"{error=} {relative_error=}")
    
    # Do matmul with TF32 disabled
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"{torch.backends.cuda.matmul.allow_tf32=} {torch.backends.cudnn.allow_tf32=}")
    
    tic = time.time()
    ab_fp32 = a @ b  # takes 0.11s on GA100
    print(f"time: {time.time() - tic}")
    error = (ab_fp32 - ab_full).abs().max()  # 0.0031
    relative_error = error / mean  # 0.000039
    print(f"{error=} {relative_error=}")

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


def main():
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


def ola(cfg,dvc):
    print(cfg)
    
if __name__ == "__main__":
    
    main()