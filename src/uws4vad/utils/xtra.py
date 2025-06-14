from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from uws4vad.utils.logger import get_log
log = get_log(__name__)


def xtra(cfg):
    """
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    if not cfg.get("xtra"):
        log.warning("Extras config not found! <cfg.xtra=null>")
        return

    # disable python warnings
    if cfg.xtra.get("no_warns"):
        log.info(
            "Disabling python warnings! <cfg.xtra.no_warns=True>"
        )
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    #if cfg.xtra.get("enforce_tags"):
    #    log.info("Enforcing tags! <cfg.xtra.enforce_tags=True>")
    #    enforce_tags(cfg, save_to_file=True)

    if cfg.xtra.get("log_cfg") != 0:
        # pretty print config tree using Rich library
        log.info( "Printing config tree with Rich! <cfg.xtra.log_cfg=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "path",
        "model",
        "net",
        "hydra",
        "exp"
    ),
    resolve: bool = False,
    save_to_file: bool = False,
    ) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config
            components are printed.
        resolve (bool, optional): Whether to resolve reference fields of
            DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra
            output folder.
    """
    
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    if cfg.xtra.get("log_cfg") > 1:
        # add all the other fields to queue (not specified in `print_order`)
        for field in cfg:
            if field not in queue:
                queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
            
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.path.out_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in
    config."""
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning(
            "No tags provided in config. Prompting user to input tags..."
        )
        tags = Prompt.ask(
            "Enter a list of comma separated tags", default="dev"
        )
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.path.out_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
