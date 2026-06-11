"""Learned beat tracking and librosa structure segmentation backends."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.ndimage
import scipy.sparse.csgraph
import sklearn.cluster

from autohotcue.analysis import HOP, SR, band_energy

_beat_model = None
_beat_device: str | None = None


@dataclass(frozen=True)
class BeatAnalysis:
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    duration_s: float
    source: str

    def bar_s(self) -> float:
        if len(self.downbeats) >= 2:
            return float(np.median(np.diff(self.downbeats)))
        if len(self.beats) >= 2:
            return float(np.median(np.diff(self.beats))) * 4.0
        return 4.0 * 60.0 / self.bpm

    def nearest_downbeat(self, t: float) -> float:
        if len(self.downbeats) == 0:
            return t
        idx = int(np.argmin(np.abs(self.downbeats - t)))
        return float(self.downbeats[idx])


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: str
    energy_rank: float = 0.0


@dataclass(frozen=True)
class StructureAnalysis:
    segments: list[Segment]
    source: str


def init_worker(jobs: int) -> None:
    """Cap torch thread count per process pool worker."""
    import torch

    n = max(1, jobs)
    threads = max(1, torch.get_num_threads() // n)
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))


def _get_beat_model(device: str | None = None):
    global _beat_model, _beat_device
    dev = device or "cpu"
    if _beat_model is None or _beat_device != dev:
        from beat_this.inference import Audio2Beats

        _beat_model = Audio2Beats(device=dev, dbn=False)
        _beat_device = dev
    return _beat_model


def track_beats(
    y: np.ndarray,
    sr: int = SR,
    device: str | None = None,
) -> BeatAnalysis:
    """Track beats and downbeats with beat_this on an ffmpeg-decoded array."""
    model = _get_beat_model(device)
    beats, downbeats = model(y, sr)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    duration_s = len(y) / sr
    if len(beats) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    else:
        bpm = 120.0
    return BeatAnalysis(
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        duration_s=duration_s,
        source="beat_this",
    )


def bpm_octave_ratio(a: float, b: float) -> float:
    """Relative BPM deviation after aligning *b* to the octave nearest *a*."""
    if a <= 0 or b <= 0:
        return float("inf")
    aligned = b
    while aligned < a * 0.75:
        aligned *= 2.0
    while aligned > a * 1.5:
        aligned /= 2.0
    best = min(
        abs(a - aligned / 2.0),
        abs(a - aligned),
        abs(a - aligned * 2.0),
    )
    return best / a


def _eigengap_k(evals: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
    n = min(k_max + 1, len(evals) - 1)
    if n <= k_min:
        return k_min
    gaps = np.diff(evals[: n + 1])
    lo = max(0, k_min - 1)
    hi = min(len(gaps), k_max)
    if lo >= hi:
        return k_min
    return int(lo + np.argmax(gaps[lo:hi]) + 1)


def _snap_time_to_downbeat(t: float, downbeats: np.ndarray) -> float:
    if len(downbeats) == 0:
        return t
    idx = int(np.argmin(np.abs(downbeats - t)))
    return float(downbeats[idx])


def _label_segments(
    boundaries: list[float],
    energy_ranks: list[float],
) -> list[Segment]:
    n = len(boundaries) - 1
    if n <= 0:
        return []
    median_rank = float(np.median(energy_ranks))
    segments: list[Segment] = []
    for i in range(n):
        rank = energy_ranks[i]
        if i == 0 and rank < median_rank:
            label = "intro"
        elif i == n - 1 and rank < median_rank:
            label = "outro"
        elif rank >= 0.75:
            label = "chorus"
        elif rank <= 0.25:
            label = "break"
        else:
            label = "verse"
        segments.append(
            Segment(
                start=boundaries[i],
                end=boundaries[i + 1],
                label=label,
                energy_rank=rank,
            )
        )
    return segments


def segment_structure(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    structure_backend: str = "librosa",
) -> StructureAnalysis:
    """Beat-synchronous structure: librosa Laplacian (default) or all-in-one-mlx."""
    if structure_backend == "allin1":
        from autohotcue._allin1 import segment_structure_allin1

        return segment_structure_allin1(path, y, sr, beat)
    if structure_backend not in ("librosa", "ml"):
        raise ValueError(f"unknown structure backend: {structure_backend}")
    return segment_structure_librosa(path, y, sr, beat)


def segment_structure_librosa(
    path: str,
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
) -> StructureAnalysis:
    """Laplacian spectral clustering on beat-synchronous features (McFee–Ellis)."""
    del path  # ffmpeg-decoded array is the input; path kept for API stability
    import librosa

    if len(beat.beats) < 4:
        return StructureAnalysis(segments=[], source="librosa")

    beat_frames = librosa.time_to_frames(beat.beats, sr=sr, hop_length=HOP)

    bins_per_octave = 12 * 3
    n_octaves = 7
    cqt = librosa.amplitude_to_db(
        np.abs(
            librosa.cqt(
                y=y,
                sr=sr,
                bins_per_octave=bins_per_octave,
                n_bins=n_octaves * bins_per_octave,
            )
        ),
        ref=np.max,
    )
    csync = librosa.util.sync(cqt, beat_frames, aggregate=np.median)

    rec = librosa.segment.recurrence_matrix(csync, width=3, mode="affinity", sym=True)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    rf = df(rec, size=(1, 7))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP)
    msync = librosa.util.sync(mfcc, beat_frames)
    path_distance = np.sum(np.diff(msync, axis=1) ** 2, axis=0)
    sigma = float(np.median(path_distance)) or 1.0
    path_sim = np.exp(-path_distance / sigma)
    n_beats = csync.shape[1]
    r_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    if n_beats > 2:
        r_path = r_path[:n_beats, :n_beats]

    deg_path = np.sum(r_path, axis=1)
    deg_rec = np.sum(rf, axis=1)
    denom = float(np.sum((deg_path + deg_rec) ** 2)) or 1.0
    mu = float(deg_path.dot(deg_path + deg_rec) / denom)
    a_mat = mu * rf + (1.0 - mu) * r_path

    lap = scipy.sparse.csgraph.laplacian(a_mat, normed=True)
    evals, evecs = scipy.linalg.eigh(lap.toarray() if scipy.sparse.issparse(lap) else lap)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    cnorm = np.cumsum(evecs**2, axis=1) ** 0.5

    k = _eigengap_k(evals)
    k = min(k, max(2, n_beats - 1))
    x_feat = evecs[:, :k] / np.maximum(cnorm[:, k - 1 : k], 1e-8)

    km = sklearn.cluster.KMeans(n_clusters=k, n_init="auto", random_state=0)
    seg_ids = km.fit_predict(x_feat)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
    bound_times = [_snap_time_to_downbeat(float(beat.beats[i]), beat.downbeats)
                   for i in bound_beats if i < len(beat.beats)]

    low, _, _, _ = band_energy(y, sr, HOP)

    boundaries = [0.0]
    for t in bound_times:
        if t > boundaries[-1]:
            boundaries.append(t)
    if beat.duration_s > boundaries[-1]:
        boundaries.append(beat.duration_s)

    segment_energies: list[float] = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i + 1]
        i0 = int(t0 * sr / HOP)
        i1 = max(i0 + 1, int(t1 * sr / HOP))
        segment_energies.append(float(low[i0:i1].mean()) if i1 <= len(low) else 0.0)

    if not segment_energies:
        return StructureAnalysis(segments=[], source="librosa")

    from scipy.stats import rankdata

    ranks = (rankdata(segment_energies, method="average") - 1) / max(
        1, len(segment_energies) - 1
    )

    segments = _label_segments(boundaries, [float(r) for r in ranks])
    return StructureAnalysis(segments=segments, source="librosa")
