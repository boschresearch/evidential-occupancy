"""Commands for data export."""


import typer
from hydra.utils import instantiate

from scene_reconstruction.cli.config import make_cfg
from scene_reconstruction.eval.lidar_depth import LidarDistanceEval

app = typer.Typer(name="eval", callback=make_cfg, help="Various export commands.", no_args_is_help=True)


@app.command(name="render-lidar-depth")
def lidar_depth(ctx: typer.Context) -> None:
    """Eval rendered lidar depth."""
    cfg = ctx.meta["cfg"]

    transmission_and_reflections: LidarDistanceEval = instantiate(cfg.eval.render_lidar_depth)
    results = transmission_and_reflections.eval()
    print("Config yaml:")
    print(cfg.eval)
    for k, v in results.items():
        print(f"{k}: {v}")
