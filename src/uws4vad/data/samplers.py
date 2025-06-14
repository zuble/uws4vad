from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np
import torch

from uws4vad.utils import logger
log = logger.get_log(__name__)

'''
def get_sampler(cfg, ds=None, tc_gen=None, np_gen=None):
    
    cfg_ds = cfg.data
    cfg_dload = cfg.dataload
    
    bal_abn_bag = cfg_dload.balance.bag
    bal_abn_set = cfg_dload.balance.set
    
    if bal_abn_bag == -1: ## mperclass
        
    elif bal_abn_bag == 0:
        assert (ds and gen) is not None
        ## same as batch_size=cfg_dload.bs, shuffle=True, drop_last=True
        sampler = RandomSampler(ds, 
                    #replacement=False,
                    #num_samples=samples_per_epoch, # !!!! set intern to len(ds)
                    generator=tc_gen
                    )
    else: 
        sampler = AbnormalBatchSampler(
                    labels,
                    bal_abn_bag=bal_abn_bag,
                    bs=cfg_dload.bs,
                    bal_abn_set=bal_abn_set,
                    generator=np_gen
                )
    
    
    
    return
'''


#######################
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
    def __init__(self, labels, bal_abn_bag, bs, bal_abn_set=1, generator=None):
        """
        Args:
            labels: A list or tensor of class labels for the dataset 
            bal_abn_bag (int or tuple): The bag balance in relation to abnormal
                the be drawn in each batch. If an integer, its set
                based on batch_size. If a tuple, it should 
                have two elements specifying the number of samples 
                for ABNORMAL and NORMAL classes, respectively.
            bs (int): The total batch size.
            len_abn_set (int, optional): The desired length of anomaly set. 
                If None, it will be inferred from the labels.
        """
        self.bs=int(bs)
        self.GEN = generator
        if self.GEN is None:
            log.warning("no generator")
            self.GEN = np.random
            
        if isinstance(bal_abn_bag, (float, int)):
            if 0.3 <= bal_abn_bag < 1:
                len_bag_abn = int(self.bs*bal_abn_bag)
                len_nor_bag = self.bs - len_bag_abn
                #self.smp_per_bag = (len_bag_abn, len_nor_bag)
                self.smp_per_bag = {
                    1: len_bag_abn,
                    0: len_nor_bag
                }
            else: raise ValueError
        else: raise ValueError("'bal_abn_bag' must be an single value in [0.5:1[.")
        
        self.labels_to_indices = get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        #log.info(f"{self.labels}") #{self.labels_to_indices} 
        
        #smaller_class = min(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        #larger_class = max(self.labels, key=lambda label: len(self.labels_to_indices[label]))
        #smaller_len = len(self.labels_to_indices[smaller_class])
        #larger_len = len(self.labels_to_indices[larger_class])
        #log.info(f"    ¬ {smaller_class}:{smaller_len}  {larger_class}:{larger_len}")
        
        labels_len = {}
        for l in self.labels: labels_len[l] = len(self.labels_to_indices[l])
        len_abn_og_set = labels_len[1]
        self.len_nor_set = labels_len[0]
        log.info(f"{labels_len=}")

        log.debug(f"ABS: batch w/ {bal_abn_bag} A{self.smp_per_bag[1]} N{self.smp_per_bag[0]} {self.labels=}")
        
        
        ## calculate the possible new abnormal set length
        log.debug(f"PRE {len_abn_og_set}")
        if 1 <= bal_abn_set < 2:  #assert  1 <= bal_abn_set < 2, f"'bal_abn_set'not in [1:2["
            len_abn_set = int(bal_abn_set * len_abn_og_set)
        else:  
            # Check if the additional batches are within the allowed limit
            assert bal_abn_set <= (len_abn_og_set + self.len_nor_set) // self.bs // 5 , \
                "'bal_abn_set' set as too many additional batches."
            len_abn_set = len_abn_og_set + bal_abn_set * self.smp_per_bag[1] 
        log.debug(f"MID {len_abn_set}")

        len_abn_set = -(-len_abn_set // self.smp_per_bag[1]) * self.smp_per_bag[1]
        assert len_abn_set % self.smp_per_bag[1] == 0
        log.debug(f"POST {len_abn_set}")               
        
        self.num_batches = len_abn_set // self.smp_per_bag[1] 
        log.debug(f"NBATS {self.num_batches=}  ")
        
    def __len__(self):
        return self.num_batches * self.bs
    
    def __iter__(self):
        idx_list = []
        total_yielded = 0 
        
        ## Shuffle indices within each class beforehand
        for label in self.labels: 
            self.GEN.shuffle(self.labels_to_indices[label]) ## !!!! how can i check the seed here
            
        ## Pointers to track the current position within each class's indices
        class_pointers = {label: 0 for label in self.labels}
        for batch in range(self.num_batches):
            batch_indices = []
            
            for label in self.labels:
                t = self.labels_to_indices[label]
                num_samples = self.smp_per_bag[label]
                #log.debug(f"{num_samples} {label}")
                
                for _ in range(num_samples):
                    pointer = class_pointers[label]
                    if pointer >= len(t):
                        pointer = 0
                        class_pointers[label] = 0
                        self.GEN.shuffle(t)  ## Reshuffle the indices for this class
                        #log.debug(f"\tReshuffling indices for label {label}")
                    batch_indices.append(t[pointer])
                    pointer += 1
                    class_pointers[label] = pointer
                #log.debug(f"{label}  {batch_indices}")
                
            idx_list.extend(batch_indices)
            total_yielded += len(batch_indices)
            
        #log.debug(f"{total_yielded=}")
        return iter(idx_list)
########################


def analyze_sampler(bsamplers, labels, dataset_name, iters=2, vis=None):
    #Analyzes multiple BatchSampler behaviors over multiple iterations.
    import matplotlib.pyplot as plt
    from collections import Counter, defaultdict
    import plotly.graph_objs as go, plotly.express as px
    from plotly.subplots import make_subplots

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
        #for sampler_name, epo_counts in all_epo_counts.items():
        #    num_batches = len(epo_counts[0])
        #    for class_id, class_counts_per_batch in enumerate(epo_counts):
        #        plt.plot(range(1, num_batches + 1), class_counts_per_batch, label=f'{sampler_name}, Class {class_id}')
        #plt.xlabel('Batch Number')
        #plt.ylabel('Number of Samples')
        #plt.title(f'Per-Batch Class Distribution')
        #plt.legend()
        #plt.show()

        colors = px.colors.qualitative.Plotly
        fig = go.Figure() 
        #fig = make_subplots(rows=len(bsamplers), cols=1, subplot_titles=[sampler_name for sampler_name in bsamplers.keys()])

        for i, (sampler_name, epo_counts) in enumerate(all_epo_counts.items()):
            num_batches = len(epo_counts[0])

            for class_id, class_counts_per_batch in enumerate(epo_counts):
                # Unique color for each class within each sampler
                line_color = colors[(i * len(epo_counts) + class_id) % len(colors)]  
                
                # Line plot for class counts
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, num_batches + 1)),
                        y=class_counts_per_batch,
                        mode='lines',
                        line=dict(color=line_color, width=2),
                        name=f'{sampler_name}, Class {class_id}',
                        #name=f'Class {class_id}',
                        #legendgroup=f'{sampler_name}',  # Group lines by sampler in the legend
                        #showlegend=i == 0  # Show legend only for the first sampler
                    ),
                    #row=i + 1,  # Add trace to the correct subplot
                    #col=1
                )
        fig.update_xaxes(title_text="Batch Number")
        fig.update_yaxes(title_text="Number of Samples")
        fig.update_layout(height=800, showlegend=True, title_text="Per-Batch Class Distribution")
        #fig.update_layout(height=400 * len(bsamplers), showlegend=True, title_text="Per-Batch Class Distribution")
        
        vis.potly(fig)


###########################
## --------------------- ##    
def dummy_train_loop(vis):
    from torch.utils.data import Dataset, BatchSampler, RandomSampler
    class DummyDataset(Dataset):
        def __init__(self,labels):
            self.len = len(labels)
            self.labels = labels
        def __getitem__(self, idx): return [], self.labels[idx], idx
        def __len__(self): return self.len
        
    class DummyArgs:
        def __init__(self):
            self.bs = 32
            self.workers = 0 
            self.max_epoch = 1
            self.ds = ['ucf', 'xdv'] #,
            self.len_normal = {'ucf':800,'xdv':2049}
            self.len_abnormal = {'ucf':810,'xdv':1905}
            self.droplast = False
            ## 4 randomsampelr
            self.replacement = False 
            self.bal_abn_bag = [0.5,0.55,0.45,0.7]
            self.os_rate = [1]
    
    args = DummyArgs()
    for ds_name in args.ds:
        print(f"\n\n”{'*-'*7} {ds_name} {'*-'*7}\n\n")
        len_normal = args.len_normal[ds_name]
        len_abnormal = args.len_abnormal[ds_name]
        print(f"\t Normal/0:{len_normal}  Abnormal/1:{len_abnormal} Total:{len_normal+len_abnormal}")
        
        labels_nor = [0]*len_normal
        labels_abn = [1]*len_abnormal
        labels = labels_abn + labels_nor
        
        DS = DummyDataset(labels)
        
        BSAMPLERS = {}
        for bag_rate in args.bal_abn_bag:
            for set_rate in args.os_rate:
                ###################
                ## AbnormalBatchSampler
                #max_len = max( len_normal , len_abnormal )
                #samples_per_epoch = int(args.os_rate * max_len )
                #samples_per_epoch = max_len + args.bs * 3
                #samples_per_epoch = max_len
                #if ds_name == 'xdv':
                
                sampler22 = AbnormalBatchSampler(
                                        labels, 
                                        bal_abn_bag=bag_rate, 
                                        bs=args.bs, 
                                        bal_abn_set=set_rate
                                        ) 
                bsampler22 = BatchSampler(sampler22 , args.bs , args.droplast)
                BSAMPLERS.update({f'abs_{bag_rate}_{set_rate}':bsampler22})
        
        ###################
        ## XDV
        sampler3 = RandomSampler(DS, 
                            replacement=args.replacement,
                            num_samples=int(set_rate*len(DS))
                            )
        bsampler3 = BatchSampler(sampler3, args.bs , drop_last=False) #args.droplast
        BSAMPLERS.update({f'org_xdv_{bag_rate}_{set_rate}':bsampler3})

        ####################
        ## WeightedRandomSampler
        #rebal_wgh = 0.5 - ( 1 - (len_normal/len_abnormal) )
        #print(rebal_wgh)
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
        #print("\nWeightedRandomSampler:")
        #print(f"{len(sampler22)=} {len(bsampler22)} {len(naloader22)} ")
        #BSAMPLERS.update({'wrs':bsampler2})
        
        #########
        analyze_sampler(BSAMPLERS, labels, ds_name, iters=args.max_epoch, vis=vis)
        #########
        