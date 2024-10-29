"""Configuration helpers."""

from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def initialize_config(cfg_dir: Path, cfg_name: str, overrides: Optional[list[str]] = None) -> DictConfig:
    """Initialize and return hydra config."""

    with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None):
        return compose(config_name=cfg_name, overrides=overrides)
