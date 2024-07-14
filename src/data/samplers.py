from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
import torch

from src.utils import logger
log = logger.get_log(__name__)


def get_labels_to_indices(labels):
    """
    Creates labels_to_indices, which is a dictionary mapping each label
    to a numpy array of indices that will be used to index into self.dataset
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for k, v in labels_to_indices.items():
        labels_to_indices[k] = np.array(v, dtype=int)
    return labels_to_indices

def safe_random_choice(input_data, size):
    """
    Randomly samples without replacement from a sequence. It is "safe" because
    if len(input_data) < size, it will randomly sample WITH replacement
    Args:
        input_data is a sequence, like a torch tensor, numpy array,
                        python list, tuple etc
        size is the number of elements to randomly sample from input_data
    Returns:
        An array of size "size", randomly sampled from input_data
    """
    replace = len(input_data) < size
    return np.random.choice(input_data, size=size, replace=replace)

# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
class MPerClassSampler(Sampler):
    def __init__(self, labels, m, batch_size, length_before_new_iter=100000, generator=None):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        #log.debug(f"{self.labels_to_indices} {self.labels}")
        
        # Determine the smaller and larger classes
        self.smaller_class = min(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        self.larger_class = max(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        
        self.smaller_len = len(self.labels_to_indices[self.smaller_class])
        self.larger_len = len(self.labels_to_indices[self.larger_class])
        #log.debug(f"{self.smaller_len=} {self.larger_len=}")
        
        ## use to give more samples equally addint into the num_batches calculation
        ## and maybe provide a better parameters that this function givve , with better names, 
        ## or add other to improve sampler design
        self.length_before_new_iter = length_before_new_iter
        #log.error(f"{self.length_before_new_iter=}")
        assert self.length_before_new_iter >= self.batch_size
        assert (self.batch_size % (self.m_per_class * len(self.labels))) == 0, "batch_size must be divisible by m * (number of unique labels)"
        #self.length_before_new_iter -= self.length_before_new_iter % self.batch_size
        ##################

        # Calculate the number of batches required for each class
        self.num_batches_smaller = self.smaller_len // self.m_per_class
        self.num_batches_larger = self.larger_len // self.m_per_class + \
                                (self.larger_len  % self.m_per_class > 0)  # Add 1 if there's a remainder

        self.num_batches = max(self.num_batches_smaller, self.num_batches_larger)

    def __len__(self):
        #return self.length_before_new_iter
        return self.num_batches * self.batch_size
    
    def __iter__(self):
        idx_list = []
        
        ## Shuffle indices within each class beforehand
        for label in self.labels:
            np.random.shuffle(self.labels_to_indices[label])

        ## Pointers to track the current position within each class's indices
        class_pointers = {label: 0 for label in self.labels}
        #log.error(class_pointers)
        
        for batch in range(self.num_batches):
            batch_indices = []
            for label in self.labels:
                t = self.labels_to_indices[label]
                for _ in range(self.m_per_class):  ## Add m samples per class
                    pointer = class_pointers[label]

                    if pointer >= len(t): ## end of a class's indices, reset the pointer
                        pointer = 0
                        class_pointers[label] = 0
                        np.random.shuffle(t)  # Reshuffle the indices for this class
                        log.debug(f"Reshuffling indices for label {label} in batch {batch}")  # This should now be triggered

                    batch_indices.append(t[pointer])
                    class_pointers[label] = pointer + 1
                    #log.info(f"Indexing {t[pointer]} for label {label} in batch {batch}")  
                    
            #log.error(len(batch_indices))
            idx_list.extend(batch_indices)

        return iter(idx_list)
