"""Render a waveform with cue markers to a PNG for visual review."""
from __future__ import annotations

import numpy as np

from autohotcue import analysis

CUE_COLORS = {
    "A": "#34c759", "B": "#30b0c7", "C": "#ffcc00", "D": "#ff3b30",
    "E": "#007aff", "F": "#af52de", "G": "#5ac8fa", "H": "#ff9500",
}
CUE_LABELS = {
    "A": "First Beat", "B": "Loop In", "C": "Vocal/Buildup", "D": "Drop",
    "E": "Breakdown", "F": "Special", "G": "Outro", "H": "Loop Out",
}


def render(path: str, grid, prop, out_png: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = analysis.decode(path)
    sr = analysis.SR
    # Downsample envelope for plotting
    win = sr // 100  # 10ms
    n = len(y) // win
    env = np.abs(y[: n * win]).reshape(n, win).max(axis=1)
    t = np.arange(n) * win / sr

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.fill_between(t, env, color="#cccccc", linewidth=0)
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylim(0, env.max() * 1.05)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title, fontsize=11)

    for pad in "ABCDEFGH":
        ts = prop.positions.get(pad)
        if ts is None:
            continue
        color = CUE_COLORS[pad]
        ax.axvline(ts, color=color, linewidth=2)
        ax.text(
            ts, env.max() * (1.0 if pad in "ACEG" else 0.88), f" {pad} {CUE_LABELS[pad]}",
            color=color, fontsize=8, rotation=90, va="top", ha="left", fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
