from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from numpy.typing import NDArray

LabelAction = Literal["accept", "unmatchable", "skip", "quit"]
ReferenceAction = Literal["accept", "next", "quit"]


@dataclass(frozen=True)
class LabelingDecision:
    action: LabelAction
    gt_polygon: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class ReferenceDrawDecision:
    action: ReferenceAction
    polygon: list[tuple[float, float]] | None = None


def draw_reference_polygon(
    candidate_image: NDArray[np.float64],
    candidate_label: str,
    allow_next_candidate: bool = True,
) -> ReferenceDrawDecision:
    """
    Interactively draw a reference polygon on a thermal image.
    Controls:
      - Left click: add vertex
      - Right click / Backspace: remove last vertex
      - Y / Enter: accept polygon (requires >= 3 vertices)
      - N / S: try next image as reference (when available)
      - Q: save progress and quit
    """
    points: list[tuple[float, float]] = []
    decision: dict[str, ReferenceDrawDecision] = {
        "value": ReferenceDrawDecision(
            action="next" if allow_next_candidate else "quit"
        )
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    next_hint = " | N/S: next frame" if allow_next_candidate else ""
    fig.suptitle(
        f"Draw REFERENCE polygon: {candidate_label}\n"
        f"Left-click: add point | Right-click/Backspace: undo | "
        f"Y/Enter: accept (>=3 pts){next_hint} | Q: quit",
        fontsize=9,
    )
    ax.imshow(candidate_image, cmap="RdYlBu_r")
    ax.axis("off")

    (line,) = ax.plot([], [], color="lime", marker="o", linewidth=2)
    preview_patch = MplPolygon(
        np.empty((0, 2), dtype=np.float64),
        closed=True,
        facecolor=(0.0, 0.8, 0.3, 0.3),
        edgecolor="lime",
        linewidth=2,
    )
    preview_patch.set_visible(False)
    ax.add_patch(preview_patch)

    def refresh() -> None:
        line.set_data([p[0] for p in points], [p[1] for p in points])
        if len(points) >= 3:
            preview_patch.set_xy(np.asarray(points, dtype=np.float64))
            preview_patch.set_visible(True)
        else:
            preview_patch.set_visible(False)
        ax.set_title(f"Reference vertices: {len(points)}")
        fig.canvas.draw_idle()

    def on_press(event: Any) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            points.append((float(event.xdata), float(event.ydata)))
            refresh()
        elif event.button == 3 and points:
            points.pop()
            refresh()

    def on_key(event: Any) -> None:
        key = event.key
        if key is None:
            return
        if key == "backspace" and points:
            points.pop()
            refresh()
        elif key in ("y", "enter"):
            if len(points) < 3:
                ax.set_title("Need at least 3 vertices before accepting!")
                fig.canvas.draw_idle()
                return
            decision["value"] = ReferenceDrawDecision(
                action="accept", polygon=list(points)
            )
            plt.close(fig)
        elif key in ("n", "s") and allow_next_candidate:
            decision["value"] = ReferenceDrawDecision(action="next")
            plt.close(fig)
        elif key == "q":
            decision["value"] = ReferenceDrawDecision(action="quit")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    plt.tight_layout()
    plt.show()
    return decision["value"]


def label_thermal_sample(
    reference_image: NDArray[np.float64],
    reference_polygon: Sequence[tuple[float, float]],
    source_image: NDArray[np.float64],
    initial_polygon: Sequence[tuple[float, float]],
    sample_label: str,
    confidence: float | None = None,
) -> LabelingDecision:
    """
    Interactively adjust or redraw a ground-truth polygon on a source thermal image.
    Controls:
      - Arrows / Shift+Arrows or mouse drag: shift polygon
      - d: enter redraw mode (left-click vertices, right-click or d to finish)
      - r: reset polygon to initial alignment
      - y / enter: accept ground-truth polygon
      - u: mark sample as unmatchable
      - s: skip sample
      - q: save progress and quit
    """
    base_polygon = [(float(x), float(y)) for x, y in initial_polygon]
    state: dict[str, Any] = {
        "polygon": list(base_polygon),
        "decision": LabelingDecision(action="skip"),
        "dragging": False,
        "drag_start": None,
        "drag_origin_polygon": None,
        "redraw_mode": False,
        "redraw_points": [],
    }

    fig, (ax_ref, ax_src) = plt.subplots(1, 2, figsize=(15, 6))
    confidence_text = f" | auto conf={confidence:.2f}" if confidence is not None else ""
    fig.suptitle(
        f"{sample_label}{confidence_text}\n"
        "Arrows/Drag: move | Shift+Arrows: 10px | D: redraw | R: reset | "
        "Y/Enter: accept | U: unmatchable | S: skip | Q: quit",
        fontsize=9,
    )

    ax_ref.set_title("Reference image + polygon")
    ax_ref.imshow(reference_image, cmap="RdYlBu_r")
    ax_ref.add_patch(
        MplPolygon(
            np.asarray(reference_polygon, dtype=np.float64),
            closed=True,
            facecolor=(0.0, 0.8, 0.3, 0.35),
            edgecolor="lime",
            linewidth=2,
        )
    )
    ax_ref.axis("off")

    ax_src.set_title("Source image (editable GT polygon)")
    ax_src.imshow(source_image, cmap="RdYlBu_r")
    editable_patch = MplPolygon(
        np.asarray(state["polygon"], dtype=np.float64),
        closed=True,
        facecolor=(1.0, 0.2, 0.2, 0.35),
        edgecolor="white",
        linewidth=2,
    )
    ax_src.add_patch(editable_patch)
    (redraw_line,) = ax_src.plot([], [], color="yellow", marker="o", linewidth=2)
    ax_src.axis("off")

    def refresh() -> None:
        if state["redraw_mode"]:
            pts = state["redraw_points"]
            redraw_line.set_data([p[0] for p in pts], [p[1] for p in pts])
            editable_patch.set_visible(False)
            ax_src.set_title(
                f"REDRAW MODE ({len(pts)} pts): left-click to add, right-click/D to close"
            )
        else:
            redraw_line.set_data([], [])
            editable_patch.set_xy(np.asarray(state["polygon"], dtype=np.float64))
            editable_patch.set_visible(True)
            ax_src.set_title("Source image (editable GT polygon)")
        fig.canvas.draw_idle()

    def shift_polygon(dx: float, dy: float) -> None:
        state["polygon"] = [(x + dx, y + dy) for x, y in state["polygon"]]
        refresh()

    def finish_redraw() -> None:
        if len(state["redraw_points"]) >= 3:
            state["polygon"] = list(state["redraw_points"])
        state["redraw_mode"] = False
        state["redraw_points"] = []
        refresh()

    def on_key(event: Any) -> None:
        key = event.key
        if key is None:
            return
        if state["redraw_mode"]:
            if key in ("d", "enter"):
                finish_redraw()
            elif key == "escape":
                state["redraw_mode"] = False
                state["redraw_points"] = []
                refresh()
            return

        step_map: dict[str, tuple[float, float]] = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "up": (0.0, -1.0),
            "down": (0.0, 1.0),
            "shift+left": (-10.0, 0.0),
            "shift+right": (10.0, 0.0),
            "shift+up": (0.0, -10.0),
            "shift+down": (0.0, 10.0),
        }
        if key in step_map:
            dx, dy = step_map[key]
            shift_polygon(dx, dy)
        elif key == "r":
            state["polygon"] = list(base_polygon)
            refresh()
        elif key == "d":
            state["redraw_mode"] = True
            state["redraw_points"] = []
            refresh()
        elif key in ("y", "enter"):
            state["decision"] = LabelingDecision(
                action="accept",
                gt_polygon=list(state["polygon"]),
            )
            plt.close(fig)
        elif key == "u":
            state["decision"] = LabelingDecision(action="unmatchable", gt_polygon=None)
            plt.close(fig)
        elif key == "s":
            state["decision"] = LabelingDecision(action="skip")
            plt.close(fig)
        elif key == "q":
            state["decision"] = LabelingDecision(action="quit")
            plt.close(fig)

    def on_press(event: Any) -> None:
        if event.inaxes != ax_src or event.xdata is None or event.ydata is None:
            return
        if state["redraw_mode"]:
            if event.button == 1:
                state["redraw_points"].append((float(event.xdata), float(event.ydata)))
                refresh()
            elif event.button == 3:
                finish_redraw()
            return
        if event.button == 1:
            state["dragging"] = True
            state["drag_start"] = (float(event.xdata), float(event.ydata))
            state["drag_origin_polygon"] = list(state["polygon"])

    def on_motion(event: Any) -> None:
        if (
            not state["dragging"]
            or event.inaxes != ax_src
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y = state["drag_start"]
        dx = float(event.xdata) - start_x
        dy = float(event.ydata) - start_y
        state["polygon"] = [(x + dx, y + dy) for x, y in state["drag_origin_polygon"]]
        refresh()

    def on_release(_event: Any) -> None:
        state["dragging"] = False
        state["drag_start"] = None
        state["drag_origin_polygon"] = None

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.tight_layout()
    plt.show()
    return state["decision"]
