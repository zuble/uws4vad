The config is managed trough [hydra](https://hydra.cc/docs/intro/), following a hierarchical configuration by composition and can be overrided through config files and the command line. 

To get a view of both config/hydra-related arguments.
 
```bash
python main.py --help
python main.py --hydra-help ## while on it add tab completation (bash/zsh/fish shells), it helps
python main.py -c=job/hydra/all 
```

The final config follows the order provided in the entry *.yamls* files, under the cfg folder, which can be stated in main.py decorator. One can override with --config-name / -cn parameter.


## Structure Overview

It tries to mimic much as possible the src code in terms of levels/folders and its main use. Although some logic is separated, whenever its needed for both trian and test, for example the model
Some aditional ones (path, )

Refer to *ucf/xdv.yaml* as they are the entry configs under  folder and *000.yaml* (which serves as base for both, with some shared/sane defaults), to better understand the parameters and its use.
The same applies for others, as the .yml's are fairly commented, especially the *_dflt/dflt.yaml* in each folder.

- **data:** metadata for each dataset
**data/faud:** defaults parameters used for each audio feature extractor
**data/frgb:** defaults parameters used for each visual feature extractor
**dataload:** controls the batchloader and dataloader settings 
**dataproc:** controls segmentation and crops
- **model:** constructs the modelling using all the below 
**model/loss:** necessaire params to contruct each available loss function under src.model.loss.feat/score.py (targetting both scores & features)
**model/lrs:** learning rate schedulers
**model/optima:** optimizer
**model/pstfwd:** post-forward reference parameters
- **net:** each network can be construced using the predifined modules
**net/cls:** classifier (regressor)
**net/fm:** feature modulator
- **vldt:** validation parameters for both train/test, as both use the same class
- **path:** dflt
- **xtra:** controls cfg logging and visdom env 
- **exp:** recommended way to override all the above cfg parameters to experiment combinations  
- **debug:** enables to set log level to debug to specific .py's, and dry_run dataloaders/validation


## Usage Examples

TODO: add valuable examples

- Basic training with RTFM:
`python main.py model=rtfm`

- deletes visdom envs matching specified string or all
'all' and others strings expect 'debug' ask for confirmation
`python main.py xtra.vis.delete='all'`

- prints only essential cfg blocks
`python main.py xtra.cfg=1`

- Debug mode with visualization:
`python main.py model=rtfm debug=vis`

- Custom experiment setup:
`python main.py exp=cmala dataload=high`