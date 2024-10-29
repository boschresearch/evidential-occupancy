"""CLI entry point."""

import typer

from scene_reconstruction.cli import eval, export

app = typer.Typer(
    name="cli",
    help="CLI entry point.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(typer_instance=export.app)
app.add_typer(typer_instance=eval.app)


if __name__ == "__main__":
    app()
