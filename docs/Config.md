The config is managed trough [hydra](https://hydra.cc/docs/intro/), following a hierarchical configuration by composition and can be overrided through config files and the command line. 

To get a view of the config and hydra-related arguments.
 
```bash
python main.py --help
python main.py --hydra-help
```

The final config follows the order provided in the entry cfg file, which is stated in main.py decorator. One can override with cmd configname parameter ```-cn ucf/xdv```

### Structure Overview

Refer to *ucf/xdv.yaml* as they are the entry and *000.yaml* (which serves as base for both), to better understand the parameters and its use. The same applies for others, as the .yml's are fairly commented, especially the dflt.yaml in each folder.

```bash
data: metadata for dataset
data/faud: defaults parameters used for each audio feature extractor
data/frgb: defaults parameters used for each visual feature extractor
dataload: controls the batchloader and dataloader settings 
dataproc: controls segmentation and crops
model: constructs the modelling using all the below 
model/loss: loss functions targetting both scores & features
model/lrs: leraning rate schedulers
model/optima: optimizer
model/pstfwd: post-forward reference parameters
net: each network can be construced using some predifined modules
net/cls: classifier (regressor)
net/fm: feature modulator
vldt: validation parameters for both train/test, as both use the same class
path: dflt
xtra: dflt
exp: recommended way to override all the above cfg parameters, serving as experiments
debug: enables to set log level to debug to specific .py's, and dry_run dataloaders/validation 
```

