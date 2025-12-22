from pathlib import Path

import matplotlib.pyplot as plt

from .fff_loader import load_fff_from_bytes


def plot_thermal_image(image, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap="jet")
    plt.colorbar(label="Temperature")
    plt.title(title)
    plt.show()


def plot_fff_from_path(file_path: Path):
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    image = load_fff_from_bytes(file_bytes)
    plot_thermal_image(image, f"Thermal Image: {file_path}")
