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
class AbnormalBatchSampler(Sampler):
    """
    Sampler that returns a fixed number of samples per class at each iteration.
    """
    def __init__(self, labels, bal_abn_bag, bs, bal_abn_set, generator):
        """
        Args:
            labels: A list or tensor of class labels for the dataset 
            bal_abn_bag (float): The ratio of abnormal samples in each batch (0.0 to 1.0).
            bs (int): The total batch size.
            len_abn_set (int, optional): The desired length of anomaly set. 
                If None, it will be inferred from the labels.
        """
        self.bs = int(bs)
        self.gen = generator
        if isinstance(bal_abn_bag, (float, int)):
            if 0.5 <= bal_abn_bag < 1:
                len_bag_abn = int(self.bs*bal_abn_bag)
                len_nor_bag = self.bs - len_bag_abn
                self.smp_per_bag = (len_bag_abn, len_nor_bag)
            else: raise ValueError
        else: raise ValueError("'bal_abn_bag' must be an single value in [0.5:1[.")
        
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        #log.info(f"{self.labels}") #{self.labels_to_indices} 
        
        labels_len = {}
        for l in self.labels: labels_len[l] = len(self.labels_to_indices[l])
        len_abn_set = labels_len[1]
        self.len_nor_set = labels_len[1]
        
        log.info(f"ABS: batch w/ A{self.smp_per_bag[0]} N{self.smp_per_bag[1]} {self.labels=}")

        ## calculate the possible new abnormal set length
        log.info(f"PRE {len_abn_set}")
        if 1 <= bal_abn_set < 2:
            len_abn_set = int(bal_abn_set * len_abn_set)
        ## want to make sure that the number of added bats are not to drasticall
        elif bal_abn_set * self.bs < (len_abn_set+self.len_nor_set)//self.bs//5:
            len_abn_set = len_abn_set + int(bal_abn_set)*self.bs
        else: 
            raise ValueError("'bal_abn_set' must be [1:2[ or be the number of additional batchs")
        
        smaller_class = min(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        larger_class = max(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        smaller_len = len(self.labels_to_indices[smaller_class])
        larger_len = len(self.labels_to_indices[larger_class])
        log.info(f"    ¬ {smaller_class}:{smaller_len}  {larger_class}:{larger_len}")
        
        self.len_abn_set = len_abn_set + \
                        ((len_abn_set) % self.smp_per_bag[0] > 0) * self.bs
        log.info(f"POST {len_abn_set} ")                     
        
        self.num_batches = self.len_abn_set // self.smp_per_bag[0] + \
                            ((self.len_abn_set) % self.smp_per_bag[0] > 0)
        log.info(f"NBATS {self.num_batches=}  ")   
        #self.num_batches = (self.len_abn_set*2) // self.bs
        #log.info(f"21 {self.num_batches}  ")   
        
    def __len__(self):
        return self.num_batches * self.bs
    
    def __iter__(self):
        idx_list = []
        total_yielded = 0 
        
        # Shuffle indices within each class beforehand
        for label in self.labels: 
            self.gen.shuffle(self.labels_to_indices[label]) ## !!!! how can i check the seed here
            
        # Pointers to track the current position within each class's indices
        class_pointers = {label: 0 for label in self.labels}
        for batch in range(self.num_batches):
            batch_indices = []
            
            for label in self.labels:
                t = self.labels_to_indices[label]
                num_samples = self.smp_per_bag[label]
                #log.info(f"{num_samples} {label}")
                
                for _ in range(num_samples):
                    pointer = class_pointers[label]
                    if pointer >= len(t):
                        pointer = 0
                        class_pointers[label] = 0
                        self.gen.shuffle(t)  # Reshuffle the indices for this class
                        log.debug(f"\tReshuffling indices for label {label}")
                    batch_indices.append(t[pointer])
                    pointer += 1
                    class_pointers[label] = pointer
                #log.info(f"{label}  {batch_indices}")
                
            idx_list.extend(batch_indices)
            total_yielded += len(batch_indices)
            
        #log.info(f"{total_yielded=}")
        return iter(idx_list)

'''
class DummyDataset(Dataset):
    def __init__(self,labels):
        self.len = len(labels)
        self.labels = labels
    def __getitem__(self, idx):
        if idx >= self.len:
            log.info("olaolaolaolaolaoalaoaoaooa")
            ## case for sult to oversampl
            idx = np.random.randint(0,len(self.len))
            
        features = np.random.rand(100, 1024).astype(np.float32)
        return features[:32], self.labels[idx], idx
    def __len__(self):
        return self.len

class DummyArgs:
    def __init__(self):
        self.bs = 32
        self.workers = 0 
        self.max_epoch = 1
        self.ds = ['ucf','xdv']
        self.len_normal = {'ucf':800,'xdv':2049}
        self.len_abnormal = {'ucf':810,'xdv':1905}
        self.droplast = False
        ## 4 randomsampelr
        self.replacement = False 
        self.os_rate = 1.05
        

def dummy_train_loop(args):
    for ds_name in args.ds:
        log.info(f"\n\n”{'*-'*7} {ds_name} {'*-'*7}\n\n")
        len_normal = args.len_normal[ds_name]
        len_abnormal = args.len_abnormal[ds_name]
        log.info(f"\t Normal/0:{len_normal}  Abnormal/1:{len_abnormal} Total:{len_normal+len_abnormal}")
        
        labels_nor = [0]*len_normal
        labels_abn = [1]*len_abnormal
        labels = labels_nor + labels_abn
        
        DS = DummyDataset(labels)
        
        BSAMPLERS = {}
        
        
        ###################
        ## UCF
        ## OG
        #ds_nor = DummyDataset(labels_nor)
        #ds_abn = DummyDataset(labels_abn)
        #nsampler = RandomSampler(ds_nor, 
        #                    replacement=args.replacement,
        #                    #num_samples=samples_per_epoch
        #                    )
        #asampler = RandomSampler(ds_abn, 
        #                    replacement=args.replacement,
        #                    #num_samples=samples_per_epoch, # !!!! set intern to len(ds)
        #                    )
        #nbsampler = BatchSampler(nsampler, args.bs//2, args.droplast)
        #absampler = BatchSampler(asampler, args.bs//2, args.droplast)
        #log.info("\nUCF 2 DL (Original):")
        ##log.info(f"{len(nbsampler)} {len(absampler)} ")
        #analyze_sampler_pair(nbsampler,absampler,labels_nor, labels_abn, ds_name)
        
        ## this how original does
        #nloader = DataLoader(
        #    ds_nor,
        #    batch_size=args.bs//2,shuffle=True,drop_last=args.droplast,
        #    num_workers=args.workers,
        #    )
        #aloader = DataLoader(
        #    ds_abn,
        #    batch_size=args.bs//2, shuffle=True,drop_last=args.droplast,
        #    num_workers=args.workers,
        #    )
        #niters =  min(len_normal,len_abnormal) // (args.bs//2)
        #log.info(niters)
        #n_all_indices = []
        #a_all_indices = []
        #for epoch in range(args.max_epoch):
        #    for i in range(niters): 
        #        _,_,idxn = next(iter(nloader))
        #        _,_,idxa = next(iter(aloader))
        ###################
        
        
        ###################
        ## MPerClassSampler
        ## a replica in per batch distr
        #sampler1 = MPerClassSampler(
        #                        labels, 
        #                        m=args.bs//2, 
        #                        batch_size=args.bs, 
        #                        length_before_new_iter=int(args.os_rate*len(DS))
        #                        ) 
        #bsampler1 = BatchSampler(sampler1 , args.bs , args.droplast)
        #naloader1 = DataLoader(
        #    DS,
        #    batch_sampler=bsampler1,
        #    num_workers=args.workers,
        #    )
        #log.info(f"MPCS {len(sampler1)} {len(bsampler1)} {len(naloader1)} ")
        #BSAMPLERS.update({'mpcs':bsampler1})
        
        ###################
        ## AbnormalBatchSampler
        ## assures that only after the dataset with
        max_len = max( len_normal , len_abnormal )
        #samples_per_epoch = int(args.os_rate * max_len )
        #samples_per_epoch = max_len + args.bs * 3
        samples_per_epoch = max_len
        sampler2 = AbnormalBatchSampler(
                                labels, 
                                bal_abn_bag=(args.bs//2-2, args.bs//2+2), 
                                batch_size=args.bs, 
                                abg_set_bal=samples_per_epoch 
                                ) 
        bsampler2 = BatchSampler(sampler2 , args.bs , args.droplast)
        naloader2 = DataLoader(
            DS,
            batch_sampler=bsampler2,
            num_workers=args.workers,
            )
        #log.info(f"ABS {len(sampler2)=} {len(bsampler2)} {len(naloader2)} ")
        BSAMPLERS.update({'abs':bsampler2})
        
        
        ###################
        ## XDV
        sampler3 = RandomSampler(DS, 
                            replacement=args.replacement,
                            num_samples=int(args.os_rate*len(DS))
                            )
        bsampler3 = BatchSampler(sampler3, args.bs , drop_last=False) #args.droplast
        naloader3 = DataLoader(
            DS,
            #batch_size=args.bs,shuffle=True,drop_last=args.droplast,
            batch_sampler=bsampler3,
            num_workers=args.workers,
            )
        #log.info(f"XDV {len(sampler3)=} , batxs {len(bsampler3)} {len(naloader3)} ")
        BSAMPLERS.update({'xdv':bsampler3})
        
        ####################
        ## WeightedRandomSampler
        #rebal_wgh = 0.5 - ( 1 - (len_normal/len_abnormal) )
        #log.info(rebal_wgh)
        #weights = [1-rebal_wgh]*len_normal + [rebal_wgh]*len_abnormal
        #sampler22 = WeightedRandomSampler(weights, 
        #                    replacement=True, #args.replacement,
        #                    num_samples=int(args.os_rate*len(DS)) , 
        #                    #generator=TCRNG
        #                    )
        #bsampler22 = BatchSampler(sampler22 , args.bs , args.droplast)
        #naloader22 = DataLoader(
        #    DS,
        #    batch_sampler=bsampler22,
        #    num_workers=args.workers,
        #    )
        #log.info("\nWeightedRandomSampler:")
        #log.info(f"{len(sampler22)=} {len(bsampler22)} {len(naloader22)} ")
        #BSAMPLERS.update({'wrs':bsampler2})
        
        #########
        analyze_sampler(BSAMPLERS, labels, ds_name, iters=args.max_epoch, vis=True)
        #########
        
'''        
def analyze_sampler(bsamplers, labels, dataset_name, iters=2, vis=False):
    #Analyzes multiple BatchSampler behaviors over multiple iterations.
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict

    dataset_size = len(labels)
    all_epo_counts = {}  # Store epo_counts for all samplers, keyed by sampler name

    for sampler_name, bsampler in bsamplers.items():  # Iterate through dictionary
        log_output = f"\n\n\tSampler: {sampler_name}\n"

        for ii in range(iters):
            all_indices = []
            epo_counts = []  # Store counts for each class across batches
            epo_class_counts = [] # Store all the counts in the epoch

            for batch_idx, batch_indices in enumerate(bsampler):
                all_indices.extend(batch_indices)

                bat_counts = [labels[idx] for idx in batch_indices]
                bat_dist = Counter(bat_counts)
                for class_id in bat_dist:
                    while len(epo_counts) <= int(class_id):
                        epo_counts.append([])
                    epo_counts[int(class_id)].append(bat_dist[class_id])
                epo_class_counts.extend(bat_counts)

            log_output += f"\t  Dataset: {dataset_name}, Epoch: {ii + 1} , {len(bsampler)} Batchs\n"
            log_output += f"\t    Samples per Epoch: {len(all_indices)}\n"

            # --- Log Class Distribution Summary ---
            class_distribution = Counter(labels[idx] for idx in all_indices)
            total_samples = len(all_indices)
            log_output += f"\t    Epoch Class Distribution\n"
            for class_id, count in sorted(class_distribution.items()):
                log_output += f"\t  Class {class_id}: {count} samples ({((count / total_samples) * 100):.2f}%)\n"

            unique_indices = set(all_indices)
            repeated_indices = len(all_indices) - len(unique_indices)
            log_output += f"\t    Repeated Indices: {repeated_indices}\n"

            # Analyze repeated indices per class
            index_counts = Counter(all_indices)
            repeated_indices_per_class = defaultdict(list)
            for idx, count in index_counts.items():
                if count > 1:
                    label = labels[idx]
                    repeated_indices_per_class[label].append(idx)
            log_output += "\t    Repeated Indices per Class:\n"
            for label, indices in sorted(repeated_indices_per_class.items()):
                log_output += f"\t      Label {label}: {len(indices)} \n"

            # --- Identify unsampled indices ---
            unsampled_indices = set(range(dataset_size)) - unique_indices
            log_output += f"\t    Unsampled Indices: {len(unsampled_indices)} \n"
            class_distribution = Counter(labels[idx] for idx in unsampled_indices)
            for class_id, count in sorted(class_distribution.items()):
                log_output += f"\t  Class {class_id}: {count} samples \n"

            all_epo_counts[sampler_name] = epo_counts

        log.info(log_output)

    # --- Visualization ---
    if vis:
        for sampler_name, epo_counts in all_epo_counts.items():
            num_batches = len(epo_counts[0])
            for class_id, class_counts_per_batch in enumerate(epo_counts):
                plt.plot(range(1, num_batches + 1), class_counts_per_batch, label=f'{sampler_name}, Class {class_id}')
        plt.xlabel('Batch Number')
        plt.ylabel('Number of Samples')
        plt.title(f'Per-Batch Class Distribution (Overlapping)')
        plt.legend()
        plt.show()



'''def analyze_sampler_pair(nbsampler, absampler, labels_nor, labels_abn, dataset_name, iters=1, vis=False):
    """Analyzes BatchSampler behavior, simulating paired iteration like the original setup."""
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict

    # Create a dictionary mapping indices to labels
    index_to_label = {idx: 0 for idx in range(len(labels_nor))}  # Normal indices
    index_to_label.update({idx + len(labels_nor): 1 for idx in range(len(labels_abn))})  # Abnormal indices
    
    for ii in range(iters):
        n_all_indices = []
        a_all_indices = []
        epo_counts = []  # Store counts for each class across batches
        epo_class_counts = [] # Store all the counts in the epoch

        # Iterate through normal and abnormal BatchSamplers in a paired fashion
        batchs=0
        for n_batch_indices, a_batch_indices in zip(nbsampler, absampler):
            #log.info(f"{batchs} {len(n_batch_indices)}")
            #log.info(f"{a} {n_batch_indices}")
            
            #log.info(f"{batchs} {len(a_batch_indices)}")
            #log.info(f"{a} {a_batch_indices}")
            batchs+=1
            n_all_indices.extend(n_batch_indices)
            a_all_indices.extend(a_batch_indices)

            # --- Simulate DataLoader behavior for class distribution per batch ---
            bat_counts = [0] * len(n_batch_indices) + [1] * len(a_batch_indices)  # Combine labels
            bat_dist = Counter(bat_counts)
            for class_id in bat_dist:
                while len(epo_counts) <= int(class_id):
                    epo_counts.append([])
                epo_counts[int(class_id)].append(bat_dist[class_id])
            epo_class_counts.extend(bat_counts)

            
        log.info(f"\t  Dataset: {dataset_name}, Epoch: {ii + 1}, {batchs} Batchs of {args.bs}")
        log.info(f"\t    Samples per Epoch: {len(n_all_indices)+len(a_all_indices)}")

        # --- Log Class Distribution Summary ---
        total_samples = len(n_all_indices) + len(a_all_indices)
        log.info(f"\t    Epoch Class Distribution")
        log.info(f"\t      Class 0: {len(n_all_indices)} samples ({((len(n_all_indices) / total_samples) * 100):.2f}%)")
        log.info(f"\t      Class 1: {len(a_all_indices)} samples ({((len(a_all_indices) / total_samples) * 100):.2f}%)")

        # --- Analyze missing indices ---
        n_missing_indices = len(labels_nor) - len(set(n_all_indices))
        a_missing_indices = len(labels_abn) - len(set(a_all_indices))
        log.info(f"\t    Normal Missing Indices: {n_missing_indices}")
        log.info(f"\t    Abnormal Missing Indices: {a_missing_indices}")
        
        # --- Analyze repeated indices ---
        n_unique_indices = set(n_all_indices)
        n_repeated_indices = len(n_all_indices) - len(n_unique_indices)
        a_unique_indices = set(a_all_indices)
        a_repeated_indices = len(a_all_indices) - len(a_unique_indices)
        log.info(f"\t    Normal Repeated Indices: {n_repeated_indices}")
        log.info(f"\t    Abnormal Repeated Indices: {a_repeated_indices}")

        # Analyze repeated indices per class (separately for each class)
        if n_repeated_indices:
            n_index_counts = Counter(n_all_indices)
            n_repeated_indices_per_class = defaultdict(list)
            for idx, count in n_index_counts.items():
                if count > 1:
                    n_repeated_indices_per_class[0].append(idx)
            log.info("\t    Normal Repeated Indices per Class:")
            for label, indices in sorted(n_repeated_indices_per_class.items()):
                log.info(f"\t      Label {label}: {len(indices)} ")
        if a_repeated_indices:
            a_index_counts = Counter(a_all_indices)
            a_repeated_indices_per_class = defaultdict(list)
            for idx, count in a_index_counts.items():
                if count > 1:
                    a_repeated_indices_per_class[1].append(idx)
            log.info("\t    Abnormal Repeated Indices per Class:")
            for label, indices in sorted(a_repeated_indices_per_class.items()):
                log.info(f"\t      Label {label}: {len(indices)} ")

        
        # Analyze repeated indices per class using index_to_label
        repeated_indices_per_class = defaultdict(list)
        for idx in all_indices:
            if all_indices.count(idx) > 1:  # Check if the index is repeated
                label = index_to_label[idx]  # Get the label from the dictionary
                if idx not in repeated_indices_per_class[label]:  # Avoid adding duplicates
                    repeated_indices_per_class[label].append(idx)

        log.info("\t    Repeated Indices per Class:")
        for label, indices in sorted(repeated_indices_per_class.items()):
            log.info(f"\t      Label {label}: {len(indices)} ")
        
        # --- Visualization ---
        if vis:
            num_batches = len(epo_counts[0])
            for class_id, class_counts_per_batch in enumerate(epo_counts):
                plt.plot(range(1, num_batches + 1), class_counts_per_batch, label=f'Class {class_id}')
            plt.xlabel('Batch Number')
            plt.ylabel('Number of Samples')
            plt.title(f'Per-Batch Class Distribution (Epoch {ii + 1})')
            plt.legend()
            plt.show()'''

if __name__ == "__main__":
    args = DummyArgs()
    dummy_train_loop(args)
