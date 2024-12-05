So (almost) everything is configured trough [hydra](https://hydra.cc/docs/intro/). It follows a hierarchical configuration by composition and can be overrided through config files and the command line. 

To get a view of the config and hydra-related arguments.
 
    python main.py --help
    python main.py --hydra-help


The final config follows the order provided in the entry cfg, which is stated in main.py decorator. One can override with ```-cn ucf/xdv```

### Structure Overview

Refer to *000.yaml*, (which serves as base for both) and mostly to *ucf/xdv.yaml* as they are the entry, the configs are fairly commented

```bash
cfg/
├── 000.yaml                # Default base configuration
├── _feaud.yaml             # Audio feature extraction config
├── _fergb.yaml             # RGB feature extraction config
├── ucf.yaml                # UCF dataset specific config
├── xdv.yaml                # XD-Violence dataset specific config
│
├── data/                   # Dataset 
│   ├── faud/              # Audio feature
│   ├── frgb/              # RGB feature
│   ├── ucf.yaml           # UCF dataset params
│   └── xdv.yaml           # XD-Violence params
│
├── dataload/              # Data loading 
│   ├── dflt.yaml          
│   ├── high.yaml          # High-performance loading
│   └── low.yaml           # Low-resource loading
│
├── dataproc/              # Data processing 
│   ├── _dflt.yaml         # Default processing
│   ├── itp.yaml           # Interpolation settings
│   ├── seq.yaml           # Sequence processing
│   └── uni.yaml           # Uniform sampling
│
├── debug/                 # Debug 
├── exp/                   # Experiment 
│
├── model/                 # Model 
│   └── 
│
├── net/                   # Network 
│   └─ mir.yaml            # the simplest case (mlp)
```

## Key Configuration Components

### 1. Model Configurations (`model/`)
- Contains model-specific hyperparameters
- Learning rates and optimizer settings
- Loss function 
- Post-processing settings

### 2. Network Architectures (`net/`)
- Different backbone architectures
- Model-specific network 
- Feature manipulation settings
- Default architectural parameters

### 3. Data Handling (`dataload/` & `dataproc/`)
- Data loading strategies
- Processing pipelines
- Sampling 
- Augmentation settings

### 4. Experiment Configurations (`exp/`)
- Specific experimental setups
- Model-specific training 
- Data loading variations
- Processing variations

### 5. Debug Settings (`debug/`)
- Component-wise debugging
- Visualization settings
- Validation 
- Performance monitoring


## Usage Examples

1. Basic training with RTFM:
```bash
python main.py model=rtfm data=xdv
```

2. Debug mode with visualization:
```bash
python main.py model=rtfm data=xdv debug=vis
```

3. Custom experiment setup:
```bash
python main.py model=rtfm data=xdv exp=cmala dataload=high
```
