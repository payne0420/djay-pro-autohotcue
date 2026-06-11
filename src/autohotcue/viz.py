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
SEG_COLORS = {
    "intro": "#a8dadc55",
    "verse": "#457b9d44",
    "chorus": "#e6394644",
    "break": "#1d355744",
    "bridge": "#7209b744",
    "solo": "#f4a26144",
    "outro": "#2a9d8f44",
}


def render(path: str, track: analysis.TrackAnalysis, prop, out_png: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = analysis.decode(path)
    sr = analysis.SR
    win = sr // 100  # 10ms
    n = len(y) // win
    env = np.abs(y[: n * win]).reshape(n, win).max(axis=1)
    t = np.arange(n) * win / sr
    duration = len(y) / sr

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.fill_between(t, env, color="#cccccc", linewidth=0)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, env.max() * 1.05)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title, fontsize=11)

    ymax = env.max() * 1.05
    if track.segments:
        for seg in track.segments:
            color = SEG_COLORS.get(seg.label, "#88888833")
            ax.axvspan(seg.start, seg.end, color=color, linewidth=0)
            mid = (seg.start + seg.end) / 2.0
            ax.text(
                mid, ymax * 0.06, seg.label,
                ha="center", va="bottom", fontsize=7, color="#333333", alpha=0.85,
            )

    if track.downbeats is not None and len(track.downbeats):
        for db in track.downbeats:
            ax.axvline(db, color="#00000022", linewidth=0.6, zorder=1)

    for pad in "ABCDEFGH":
        ts = prop.positions.get(pad)
        if ts is None:
            continue
        color = CUE_COLORS[pad]
        ax.axvline(ts, color=color, linewidth=2, zorder=3)
        ax.text(
            ts, env.max() * (1.0 if pad in "ACEG" else 0.88), f" {pad} {CUE_LABELS[pad]}",
            color=color, fontsize=8, rotation=90, va="top", ha="left", fontweight="bold",
            zorder=4,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
