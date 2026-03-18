from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True, slots=True)
class VizStyle:
    # Use 896 so MP4 export doesn't need macroblock resizing (divisible by 16).
    width: int = 896
    height: int = 896
    bg: tuple[float, float, float] = (0.05, 0.06, 0.08)
    ring_radius: float = 0.36  # relative to min(width,height)
    node_radius: float = 0.055
    edge_alpha: float = 0.85
    base_node_alpha: float = 0.18
    max_node_alpha: float = 0.98


def _node_positions(style: VizStyle):
    size = min(style.width, style.height)
    cx, cy = style.width / 2.0, style.height / 2.0
    r = style.ring_radius * size
    pos = []
    for i in range(12):
        # Put C at the top, clockwise.
        ang = -math.pi / 2 + i * (2 * math.pi / 12)
        pos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pos


def _prob_to_alpha(p: float, style: VizStyle):
    p = max(0.0, float(p))
    # Use a gentle curve so small probabilities still show.
    x = p ** 0.6
    return style.base_node_alpha + (style.max_node_alpha - style.base_node_alpha) * x


def render_frame(
    note_probs: Sequence[float],
    melody_bases: Sequence[int],
    *,
    step: int,
    style: VizStyle = VizStyle(),
):
    """
    Render a single RGB frame as a numpy array.

    - `note_probs`: length-12 probabilities for node brightness.
    - `melody_bases`: pitch classes (0..11) for the melody (ignores rhythm/octave).
    - `step`: how many melody notes to show as “played so far”.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    note_probs = np.asarray(note_probs, dtype=float)
    if note_probs.shape != (12,):
        raise ValueError("note_probs must be length 12.")

    fig = plt.figure(figsize=(style.width / 100, style.height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, style.width)
    ax.set_ylim(style.height, 0)  # invert Y so it feels screen-like
    ax.axis("off")
    fig.patch.set_facecolor(style.bg)
    ax.set_facecolor(style.bg)

    pos = _node_positions(style)

    # Draw edges of the directed “melody so far”.
    played = list(melody_bases[: max(0, min(step, len(melody_bases)))])
    for a, b in zip(played, played[1:]):
        x1, y1 = pos[int(a) % 12]
        x2, y2 = pos[int(b) % 12]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.6,
            color=(0.92, 0.34, 0.23, style.edge_alpha),
            shrinkA=22,
            shrinkB=22,
        )
        ax.add_patch(arrow)

    # Melody text at the bottom (grows as notes are played).
    # We wrap across multiple lines so longer melodies stay readable in the video.
    if played:
        names = [NOTE_NAMES[int(n) % 12] for n in played]
        max_per_line = 18
        lines = []
        for i in range(0, len(names), max_per_line):
            lines.append(", ".join(names[i:i + max_per_line]))

        ax.text(
            style.width / 2,
            style.height - 64,
            "Melody:",
            ha="center",
            va="center",
            fontsize=16,
            color=(0.92, 0.94, 0.98, 0.9),
            fontweight="bold",
        )
        # Draw lines from bottom upward.
        base_y = style.height - 34
        for li, line in enumerate(reversed(lines[-3:])):  # show last ~3 lines if very long
            ax.text(
                style.width / 2,
                base_y - li * 22,
                line,
                ha="center",
                va="center",
                fontsize=14,
                color=(0.92, 0.94, 0.98, 0.82),
            )

    # Draw nodes (brightness encodes probability).
    for i, (x, y) in enumerate(pos):
        alpha = _prob_to_alpha(note_probs[i], style)
        node = Circle(
            (x, y),
            radius=style.node_radius * min(style.width, style.height),
            facecolor=(0.78, 0.86, 0.96, alpha),
            edgecolor=(0.85, 0.9, 0.98, min(1.0, alpha + 0.12)),
            linewidth=2.0,
        )
        ax.add_patch(node)
        ax.text(
            x,
            y,
            NOTE_NAMES[i],
            ha="center",
            va="center",
            fontsize=18,
            color=(0.03, 0.04, 0.06, 0.95),
            fontweight="bold",
        )

    # Headline and current note indicator.
    ax.text(
        style.width / 2,
        70,
        "Melody Generator Probability Graph",
        ha="center",
        va="center",
        fontsize=20,
        color=(0.92, 0.94, 0.98, 0.95),
        fontweight="bold",
    )
    if played:
        cur = int(played[-1]) % 12
        ax.text(
            style.width / 2,
            110,
            f"Step {len(played)}/{len(melody_bases)}  •  Current: {NOTE_NAMES[cur]}",
            ha="center",
            va="center",
            fontsize=16,
            color=(0.92, 0.94, 0.98, 0.85),
        )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return frame


def save_melody_video(
    *,
    note_prob_history: Sequence[Sequence[float]],
    melody_bases: Sequence[int],
    out_path: str,
    fps: int = 30,
    hold_frames: int = 12,
    style: VizStyle = VizStyle(),
):
    """
    Export an animation.

    - If out_path ends with ".gif": writes a GIF (no ffmpeg required).
    - If out_path ends with ".mp4": writes an MP4 (requires ffmpeg available to imageio).
    """
    import imageio.v3 as iio

    if len(note_prob_history) < 2:
        raise ValueError("note_prob_history should include an initial snapshot + updates.")

    frames = []
    # We expect history length to be len(melody)+1, but we tolerate mismatch.
    steps = min(len(melody_bases), len(note_prob_history) - 1)

    for i in range(steps + 1):
        probs = note_prob_history[min(i, len(note_prob_history) - 1)]
        frame = render_frame(probs, melody_bases, step=i, style=style)
        frames.append(frame)

    # Hold the final frame a bit so the viewer can read the result.
    frames.extend([frames[-1]] * int(hold_frames))

    ext = out_path.lower().split(".")[-1]
    if ext == "gif":
        iio.imwrite(out_path, frames, duration=1.0 / fps)
        return out_path
    if ext == "mp4":
        iio.imwrite(out_path, frames, fps=fps)
        return out_path
    raise ValueError("out_path must end with .gif or .mp4")


def melody_bases_from_events(events: Iterable[dict]) -> list[int]:
    """
    Utility: extract pitch classes from MelodyGenerator events.
    """
    bases = []
    for ev in events:
        if "note_base" in ev:
            bases.append(int(ev["note_base"]) % 12)
        else:
            bases.append(int(ev["midi"]) % 12)
    return bases

