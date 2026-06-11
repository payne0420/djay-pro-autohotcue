"""Audio structure analysis for automatic cue placement.

Decodes a track (any ffmpeg-readable format, including .opus), locks a
straight beatgrid to a known BPM, and derives section boundaries (drop,
breakdown, buildup, second drop, outro) from band-split energy contours.

The cue system mirrors djcues (https://github.com/.../djcues):
    A First Beat | B Loop In | C Vocal/Buildup | D Drop
    E Breakdown  | F Special | G Outro         | H Loop Out
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np

SR = 44100
HOP = 512


@dataclass
class GridAnalysis:
    bpm: float
    first_beat_s: float  # grid anchor: first audible downbeat
    duration_s: float

    def beat_s(self) -> float:
        return 60.0 / self.bpm

    def bar_s(self) -> float:
        return 4 * 60.0 / self.bpm

    def snap_to_bar(self, t: float) -> float:
        """Snap a time to the nearest bar boundary on the grid."""
        bars = round((t - self.first_beat_s) / self.bar_s())
        return self.first_beat_s + max(0, bars) * self.bar_s()


@dataclass
class CueProposal:
    positions: dict[str, float] = field(default_factory=dict)  # pad letter -> seconds
    notes: list[str] = field(default_factory=list)


@dataclass
class TrackAnalysis:
    """Unified analysis result consumed by cli + viz for both engines."""

    bpm: float
    first_beat_s: float
    duration_s: float
    engine: str
    beats: np.ndarray | None = None
    downbeats: np.ndarray | None = None
    segments: list | None = None  # list[Segment] when ml engine
    djay_bpm: float | None = None


def decode(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio file to mono float32 via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-ac", "1", "-ar", str(sr), tmp.name],
            check=True,
        )
        import soundfile as sf

        y, got_sr = sf.read(tmp.name, dtype="float32")
        assert got_sr == sr
        return y


def band_energy(y: np.ndarray, sr: int = SR, hop: int = HOP):
    """Per-frame RMS energy in low (<150Hz), mid (150-4k), high (>4k) bands."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    mid = S[(freqs >= 150) & (freqs < 4000)].sum(axis=0)
    high = S[freqs >= 4000].sum(axis=0)
    total = S.sum(axis=0)
    return low, mid, high, total


def lock_grid(y: np.ndarray, known_bpm: float, sr: int = SR, hop: int = HOP) -> GridAnalysis:
    """Lock a straight grid at known_bpm; find phase + first downbeat.

    Phase: chosen to maximize onset-envelope energy at beat positions.
    First beat: first grid beat where low-band energy reaches 25% of the
    track's 95th-percentile kick level.
    """
    import librosa

    duration = len(y) / sr
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.times_like(onset, sr=sr, hop_length=hop)

    beat = 60.0 / known_bpm
    # Score each candidate phase by summed onset strength on beat positions.
    best_phase, best_score = 0.0, -1.0
    for phase in np.arange(0, beat, 0.005):
        grid = np.arange(phase, duration, beat)
        idx = np.searchsorted(times, grid)
        idx = idx[idx < len(onset)]
        score = onset[idx].sum()
        if score > best_score:
            best_score, best_phase = score, phase

    # Refine phase to sub-5ms by local search.
    for phase in np.arange(max(0, best_phase - 0.005), best_phase + 0.005, 0.001):
        grid = np.arange(phase, duration, beat)
        idx = np.searchsorted(times, grid)
        idx = idx[idx < len(onset)]
        score = onset[idx].sum()
        if score > best_score:
            best_score, best_phase = score, phase

    low, _, _, _ = band_energy(y, sr, hop)
    kick_level = np.percentile(low, 95)
    first_beat = best_phase
    grid = np.arange(best_phase, duration, beat)
    for t in grid:
        i = int(t * sr / hop)
        if i < len(low) and low[i:i + 4].max() >= 0.25 * kick_level:
            first_beat = t
            break

    return GridAnalysis(bpm=known_bpm, first_beat_s=float(first_beat), duration_s=duration)


def bar_profile(y: np.ndarray, grid: GridAnalysis, sr: int = SR, hop: int = HOP):
    """Aggregate band energies per bar from the grid anchor onward.

    Returns (bar_starts_s, low, mid, high, total) arrays normalized to the
    track's 95th-percentile per band.
    """
    low, mid, high, total = band_energy(y, sr, hop)
    bar = grid.bar_s()
    starts = np.arange(grid.first_beat_s, grid.duration_s - bar * 0.5, bar)
    out = {"low": [], "mid": [], "high": [], "total": []}
    for t in starts:
        i0 = int(t * sr / hop)
        i1 = int((t + bar) * sr / hop)
        for name, band in (("low", low), ("mid", mid), ("high", high), ("total", total)):
            seg = band[i0:i1]
            out[name].append(float(seg.mean()) if len(seg) else 0.0)
    norm = {k: np.array(v) for k, v in out.items()}
    for k in norm:
        ref = np.percentile(norm[k], 95)
        if ref > 0:
            norm[k] = norm[k] / ref
    return starts, norm["low"], norm["mid"], norm["high"], norm["total"]


def propose_cues_legacy(
    path: str, known_bpm: float, phrase_bars: int = 4
) -> tuple[GridAnalysis, CueProposal]:
    """Legacy band-energy analysis — identical to the pre-overhaul propose_cues."""
    y = decode(path)
    grid = lock_grid(y, known_bpm)
    starts, low, mid, high, total = bar_profile(y, grid)
    n = len(starts)
    p = CueProposal()

    # --- A: First Beat / B: Loop In ---
    p.positions["A"] = grid.first_beat_s
    p.positions["B"] = grid.first_beat_s
    p.notes.append(f"A/B (First Beat/Loop In): grid anchor {grid.first_beat_s:.2f}s")

    # Too short to detect structure (e.g. a clip / loop): place only first beat.
    if n < 2 * phrase_bars + 8:
        p.notes.append(f"track too short ({n} bars) for structure analysis; first beat only")
        return grid, p

    # Sustained bass presence per bar (smoothed over 2 bars)
    bass_on = low >= 0.45

    # --- D: Drop ---
    # First bar after 20% of track where bass switches from a sustained
    # low/absent stretch (>= 4 bars) to a sustained high stretch (>= 8 bars).
    min_bar = int(0.20 * n)
    drop_idx = None
    for i in range(min_bar, n - 8):
        if (not bass_on[max(0, i - 4):i].any()) and bass_on[i:i + 8].sum() >= 7:
            drop_idx = i
            break
    if drop_idx is None:
        # Fallback: largest single-bar low-band jump after 20%
        jumps = np.diff(low)
        cand = np.argsort(jumps[min_bar:])[::-1] + min_bar + 1
        drop_idx = int(cand[0]) if len(cand) else min_bar
        p.notes.append("D (Drop): fallback to largest bass jump")
    drop_idx = (drop_idx // phrase_bars) * phrase_bars
    p.positions["D"] = float(starts[drop_idx])
    p.notes.append(f"D (Drop): bar {drop_idx} at {starts[drop_idx]:.2f}s")

    # --- C: Vocal/Buildup ---
    # Start of the pre-drop section where bass cuts out: last bar before the
    # drop with bass, plus 1 -> start of bass-less buildup, phrase-snapped.
    c_idx = None
    for i in range(drop_idx - 1, 0, -1):
        if bass_on[i]:
            c_idx = i + 1
            break
    if c_idx is not None and c_idx < drop_idx:
        c_idx = (c_idx // phrase_bars) * phrase_bars
        if c_idx == drop_idx:
            c_idx = max(0, drop_idx - phrase_bars)
        p.positions["C"] = float(starts[c_idx])
        p.notes.append(f"C (Buildup): bass drops out at bar {c_idx}, {starts[c_idx]:.2f}s")
    else:
        c_idx = max(0, drop_idx - 4 * phrase_bars)
        p.positions["C"] = float(starts[c_idx])
        p.notes.append(f"C (Buildup): fallback {4 * phrase_bars} bars before drop")

    # --- E: Breakdown ---
    # Prefer the full breakdown: bass nearly absent (< 0.25) for >= 3 bars.
    # Fallback: first 2-bar dropout below the bass_on threshold.
    e_idx = None
    for i in range(drop_idx + 8, n - 3):
        if (low[i:i + 3] < 0.25).all():
            e_idx = i
            break
    if e_idx is None:
        for i in range(drop_idx + 8, n - 1):
            if not bass_on[i] and not bass_on[min(n - 1, i + 1)]:
                e_idx = i
                break
    if e_idx is not None:
        e_snap = round(e_idx / phrase_bars) * phrase_bars
        if e_snap <= drop_idx or e_snap >= n:
            e_snap = e_idx
        p.positions["E"] = float(starts[e_snap])
        p.notes.append(f"E (Breakdown): bass out at bar {e_idx}, snapped {starts[e_snap]:.2f}s")

    # --- F: Special / second drop ---
    # Next sustained bass recovery (>= 4 bars) after the breakdown.
    if e_idx is not None:
        f_idx = None
        for i in range(e_idx + 1, n - 4):
            if bass_on[i:i + 4].all():
                f_idx = i
                break
        if f_idx is not None:
            f_snap = round(f_idx / phrase_bars) * phrase_bars
            if f_snap <= e_idx or f_snap >= n:
                f_snap = f_idx
            p.positions["F"] = float(starts[f_snap])
            p.notes.append(f"F (Special): bass back at bar {f_idx}, snapped {starts[f_snap]:.2f}s")

    # --- G: Outro / H: Loop Out ---
    # Last bar where total energy holds >= 60% of peak for 4 bars; the outro
    # starts at the following phrase boundary where energy begins decaying.
    sustained = total >= 0.6
    g_idx = None
    for i in range(n - 4, 0, -1):
        if sustained[max(0, i - 4):i].all():
            g_idx = i
            break
    if g_idx is not None:
        g_snap = ((g_idx + phrase_bars - 1) // phrase_bars) * phrase_bars
        g_snap = min(g_snap, n - 1)
        p.positions["G"] = float(starts[g_snap])
        p.positions["H"] = float(starts[g_snap])
        p.notes.append(f"G/H (Outro/Loop Out): energy tail from bar {g_snap}, {starts[g_snap]:.2f}s")
    else:
        last = float(starts[max(0, n - 8)])
        p.positions["G"] = last
        p.positions["H"] = last
        p.notes.append("G/H (Outro): fallback near end")

    return grid, p


VALID_ENGINES = frozenset({"ml", "ml-librosa", "ml-allin1", "legacy"})


def normalize_engine(engine: str) -> tuple[str, str | None]:
    """Map CLI engine name to (track.engine label, structure_backend or None)."""
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"unknown engine {engine!r} (choose from {', '.join(sorted(VALID_ENGINES))})"
        )
    if engine == "legacy":
        return "legacy", None
    if engine == "ml-allin1":
        return "ml-allin1", "allin1"
    return engine, "librosa"


def _resolve_device(device: str | None, jobs: int = 1) -> str:
    """Workers always cpu; mps only when effective jobs == 1."""
    if jobs > 1:
        return "cpu"
    if device:
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def analyze(
    path: str,
    known_bpm: float | None = None,
    engine: str = "ml",
    device: str | None = None,
    jobs: int = 1,
) -> tuple[TrackAnalysis, CueProposal]:
    """Analyze a track and propose cues via the ml or legacy engine."""
    track_engine, structure_backend = normalize_engine(engine)
    if track_engine == "legacy":
        bpm = known_bpm or 120.0
        grid, prop = propose_cues_legacy(path, bpm)
        track = TrackAnalysis(
            bpm=grid.bpm,
            first_beat_s=grid.first_beat_s,
            duration_s=grid.duration_s,
            engine="legacy",
            djay_bpm=known_bpm,
        )
        return track, prop

    from autohotcue.backends import segment_structure, track_beats
    from autohotcue.cuepolicy import propose_cues as policy_propose

    dev = _resolve_device(device, jobs)
    y = decode(path)
    beat = track_beats(y, device=dev)
    structure = segment_structure(
        path, y, SR, beat, structure_backend=structure_backend or "librosa"
    )
    prop = policy_propose(beat, structure, djay_bpm=known_bpm)

    first_beat = float(beat.downbeats[0]) if len(beat.downbeats) else (
        float(beat.beats[0]) if len(beat.beats) else 0.0
    )
    track = TrackAnalysis(
        bpm=beat.bpm,
        first_beat_s=first_beat,
        duration_s=beat.duration_s,
        engine=track_engine,
        beats=beat.beats,
        downbeats=beat.downbeats,
        segments=list(structure.segments),
        djay_bpm=known_bpm,
    )
    return track, prop
