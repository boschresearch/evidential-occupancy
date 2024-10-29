"""Common constants."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from ..core.config import initialize_config

CFG_NAME = typer.Option(help="Configuration name.")
CFG_DIR = typer.Option(envvar="SR_CONFIG_DIR", dir_okay=True, readable=True, help="Configuration root directory.")
CFG_OVERRIDES = typer.Argument(show_default=False, help="Additional configuration overrides.")

OptionalOverrides = Optional[list[str]]


def make_cfg(
    ctx: typer.Context,
    cfg_dir: Annotated[Path, CFG_DIR],
    cfg_name: Annotated[str, CFG_NAME],
    cfg_overrides: Annotated[OptionalOverrides, CFG_OVERRIDES] = None,
):
    """Create configuration."""
    ctx.meta["cfg"] = initialize_config(cfg_dir=cfg_dir, cfg_name=cfg_name, overrides=cfg_overrides)
