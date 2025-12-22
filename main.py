from pathlib import Path

import typer

from sara_thermal_reading.plotting import plot_fff_from_path

app = typer.Typer()


@app.command()
def plot_fff(file_path: Path):
    plot_fff_from_path(file_path)


if __name__ == "__main__":
    app()
