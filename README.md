<p align="center"><img src="docs/img/logo.png" width="40%" alt='uws4vad'> </p>

## <p align="center"> Unified WorkStation 4 Video Anomaly Detection </p>

---

UWS4VAD is an attempt to unify common pratices in VAD methods, with support for both UCFC and XDV datasets, configured trough [hydra](https://hydra.cc/docs/intro/), in a modular and experimental pipeline, setting ground for a centralised experimental playground and benchmark. Includes feature extraction for both visual (trough [timm]() models) and audio (trough HEAR-based models).~


> [!important]
> **Looking for contributions/suggestions of any kind.** *If you have interest in the project, please dont hesitate to contact, will be more than grateful for such.*



---
### Installation

```bash
conda env create -f environment.yml && conda activate uws4vad
```



---
### Usage

Basic overview of configuration setup
```bash
python main.py --help
```
Refer to [wiki/config](https://github.com/zuble/uws4vad/wiki/Config) for a broader view of both configuration and usage.



---
### Development/Contributions

Refer to [wiki/Dev](https://github.com/zuble/uws4vad/wiki/Dev) for instructions on how to contribute or help with development, as well as an updated to-do list.



---
### Acknowledgments

Gratzie to author's works that are either part of this project, served as inspiration or contributed to VAD/VAU. 

Refer to [wiki/methods](https://github.com/zuble/uws4vad/wiki/Meth) for a complete and updated list. 



--- 
### Citation

Give a shout if used <3 

```bibtex
@misc{uws4vad,
    author = {Zuble Barbas},
    title = {A Unified WorkStation for Video Anomaly Detection.},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/zuble/uws4vad}},
    year = {2024},
}
```