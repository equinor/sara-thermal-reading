import matplotlib.pyplot as plt
import numpy as np

from sara_thermal_reading.config.settings import ImageAlignmentMethod
from sara_thermal_reading.dev_utils.synthetic_data import (
    generate_synthetic_alignment_data,
)
from sara_thermal_reading.image_alignment import get_alignment_strategy
from sara_thermal_reading.visualization.plotting import plot_thermal_image


def compare_alignments_synthetic(num_samples: int = 5) -> None:
    """
    Runs comparison of alignment methods on synthetic data.

    Args:
        num_samples: Number of synthetic samples to generate (starting from seed 0).
    """
    methods = list(ImageAlignmentMethod)

    for seed in range(num_samples):
        print(f"Processing Seed {seed}...")

        # Generate data
        # Note: generate_synthetic_alignment_data returns (ref, src, poly, translation)
        ref_img, src_img, ref_polygon, (tx, ty) = generate_synthetic_alignment_data(
            seed
        )

        # Prepare Plot
        # 1 column for Reference, then 1 column for each method
        cols = 1 + len(methods)
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))

        if cols == 1:
            axes = [axes]  # type: ignore

        # --- Plot 1: Reference Image ---
        plot_thermal_image(ref_img, f"Reference (Seed {seed})", ref_polygon, ax=axes[0])

        # --- Plot Methods ---
        for i, method in enumerate(methods):
            ax = axes[i + 1]

            # Run Alignment
            strategy = get_alignment_strategy(method)
            warped_polygon = strategy.align(ref_img, src_img, ref_polygon)

            # Draw Source Image with Result Polygon (Green by default in plot_thermal_image)
            plot_thermal_image(
                src_img,
                f"Method: {method.value}",
                warped_polygon,
                ax=ax,
                polygon_label="Predicted (Aligned)",
            )

            # Draw Ground Truth Polygon (Red Dashed)
            # Apply known translation to original polygon
            gt_polygon = [(x + tx, y + ty) for x, y in ref_polygon]
            gt_points = np.array(gt_polygon)
            gt_points = np.vstack([gt_points, gt_points[0]])  # Close loop

            ax.plot(
                gt_points[:, 0],
                gt_points[:, 1],
                "r--",
                linewidth=2,
                label="Ground Truth",
            )

            # Draw Original Non-Shifted Polygon (White Dotted)
            orig_points = np.array(ref_polygon)
            orig_points = np.vstack([orig_points, orig_points[0]])  # Close loop
            ax.plot(
                orig_points[:, 0],
                orig_points[:, 1],
                "w:",
                linewidth=2,
                label="Original (Unshifted)",
            )

            ax.legend()

        plt.tight_layout()
        plt.show()
