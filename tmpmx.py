import mxnet as mx
from mxnet import np , npx
npx.set_np()
from mxnet.gluon.data import RandomSampler, BatchSampler, Dataset

#import numpy as np
import os, random, time


class RndDS(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = np.arange(0, length, 1)
        print(f"{self.data = }")
        
    def __getitem__(self, index):
        print(f"__getitem__ {index}")
        return self.data[index]

    def __len__(self):
        return self.len


def main():
    #allow_tf32()
    
    ds1 = RndDS(10)
    ds2 = RndDS(12)

    #maxlends = max(ds1.len,ds2.len)
    maxlends = max(len(ds1),len(ds2))
    
    rsampler = RandomSampler(maxlends)
    bsampler = BatchSampler(rsampler, 2, 'discard')
    for epo in range(2): print(f"{list(bsampler)}")
    


if __name__ == "__main__":

    main()

    