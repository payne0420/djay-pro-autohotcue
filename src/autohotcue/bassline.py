"""Kick-band bass events + phrase snapping for ml-bass cue placement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autohotcue.analysis import CueProposal
from autohotcue.backends import BeatAnalysis, bpm_octave_ratio
from autohotcue.cuepolicy import _check_monotonicity
from autohotcue.gridlock import GridFit, kick_band

BASS_ON_FRAC = 0.75
BASS_OFF_FRAC = 0.55
DROP_MIN_ON_BARS = 8
PRE_DROP_MIN_OFF_BARS = 4
BREAK_MIN_OFF_BARS = 4
RETURN_MIN_ON_BARS = 8
PHRASE_BARS = 16
SNAP_MAX_BARS = 2
A_MIN_FRAC = 0.25
LOUD_REF_PCTL = 90.0
OUTRO_MIN_BARS = 2
MIN_BARS = 24


@dataclass(frozen=True)
class BassAnalysis:
    bar_starts: np.ndarray
    bar_period: float
    kick_rms: np.ndarray
    bass_on: np.ndarray
    loud_ref: float
    phrase_origin: int | None
    snapped: bool


def _build_bar_lattice(
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    fit: GridFit | None,
) -> tuple[np.ndarray, float, bool]:
    duration = len(y) / sr
    if fit is not None and fit.ok:
        bar_period = 4.0 * 60.0 / fit.render_bpm
        anchor = fit.anchor_s
        k = 0
        starts: list[float] = []
        while True:
            t0 = anchor + k * bar_period
            if t0 + bar_period > duration + 1e-6:
                break
            starts.append(t0)
            k += 1
        return np.asarray(starts, dtype=np.float64), bar_period, True

    downbeats = np.asarray(beat.downbeats, dtype=np.float64)
    bar_period = beat.bar_s()
    if len(downbeats) == 0:
        return np.array([], dtype=np.float64), bar_period, False
    return downbeats, bar_period, False


def _per_bar_kick_rms(
    y_kick: np.ndarray,
    sr: int,
    bar_starts: np.ndarray,
    bar_period: float,
    *,
    lattice_locked: bool,
) -> np.ndarray:
    n = len(bar_starts)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t0 = float(bar_starts[i])
        if lattice_locked:
            t1 = t0 + bar_period
        elif i + 1 < n:
            t1 = float(bar_starts[i + 1])
        else:
            t1 = t0 + bar_period
        out[i] = _rms_segment(y_kick, sr, t0, t1)
    return out


def _rms_segment(y: np.ndarray, sr: int, t0: float, t1: float) -> float:
    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    seg = y[i0:i1]
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg * seg)))


def _loud_ref(kick_rms: np.ndarray) -> float:
    if len(kick_rms) == 0:
        return 0.0
    return float(np.percentile(kick_rms, LOUD_REF_PCTL))


def _hysteresis_bass_on(kick_rms: np.ndarray, loud_ref: float) -> np.ndarray:
    on_thr = BASS_ON_FRAC * loud_ref
    off_thr = BASS_OFF_FRAC * loud_ref
    out = np.zeros(len(kick_rms), dtype=bool)
    state = False
    for i, rms in enumerate(kick_rms):
        if not state and rms >= on_thr:
            state = True
        elif state and rms < off_thr:
            state = False
        out[i] = state
    return out


def _runs(bass_on: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return (start_index, length, is_on) for each contiguous run."""
    if len(bass_on) == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    cur = bool(bass_on[0])
    for i in range(1, len(bass_on)):
        if bool(bass_on[i]) != cur:
            runs.append((start, i - start, cur))
            start = i
            cur = bool(bass_on[i])
    runs.append((start, len(bass_on) - start, cur))
    return runs


def _nearest_phrase_boundary(bar: int, origin: int, phrase: int) -> int:
    k = round((bar - origin) / phrase)
    return origin + k * phrase


def _phrase_offset(bar: int, origin: int, phrase: int) -> int:
    nearest = _nearest_phrase_boundary(bar, origin, phrase)
    return bar - nearest


def _snap_bar(bar: int, origin: int) -> tuple[int, int]:
    nearest = _nearest_phrase_boundary(bar, origin, PHRASE_BARS)
    d = bar - nearest
    if abs(d) <= SNAP_MAX_BARS:
        return nearest, d
    return bar, d


def _event_note(slot: str, raw_bar: int, origin: int, final_bar: int) -> str:
    off16 = _phrase_offset(raw_bar, origin, PHRASE_BARS)
    off32 = _phrase_offset(raw_bar, origin, 32)
    prefix = f"{slot}: raw bar {raw_bar}, off16={off16:+d}, off32={off32:+d}, "
    if final_bar != raw_bar:
        return prefix + f"snapped to bar {final_bar}"
    if abs(off16) > SNAP_MAX_BARS:
        return prefix + "off-phrase (unsnapped)"
    return prefix + f"at bar {final_bar}"


def _last_phrase_between(origin: int, after_bar: int, before_bar: int) -> int | None:
    k_lo = int(np.floor((after_bar - origin) / PHRASE_BARS)) + 1
    k_hi = int(np.ceil((before_bar - origin) / PHRASE_BARS)) - 1
    if k_lo > k_hi:
        return None
    return origin + k_hi * PHRASE_BARS


def _last_phrase_before_end(origin: int, n_bars: int) -> int | None:
    last_start = n_bars - OUTRO_MIN_BARS
    if last_start <= origin:
        return None
    k = int(np.floor((last_start - origin) / PHRASE_BARS))
    return origin + k * PHRASE_BARS


def analyze_bass(
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    fit: GridFit | None,
) -> BassAnalysis:
    bar_starts, bar_period, lattice_locked = _build_bar_lattice(y, sr, beat, fit)
    y_kick = kick_band(y, sr)
    kick_rms = _per_bar_kick_rms(
        y_kick, sr, bar_starts, bar_period, lattice_locked=lattice_locked,
    )
    loud = _loud_ref(kick_rms)
    bass_on = _hysteresis_bass_on(kick_rms, loud) if loud > 0 else np.zeros(
        len(bar_starts), dtype=bool,
    )
    return BassAnalysis(
        bar_starts=bar_starts,
        bar_period=bar_period,
        kick_rms=kick_rms,
        bass_on=bass_on,
        loud_ref=loud,
        phrase_origin=None,
        snapped=lattice_locked,
    )


def propose_cues_bass(
    y: np.ndarray,
    sr: int,
    beat: BeatAnalysis,
    fit: GridFit | None,
    djay_bpm: float | None = None,
) -> tuple[CueProposal, BassAnalysis]:
    p = CueProposal()
    bass = analyze_bass(y, sr, beat, fit)
    bar_starts = bass.bar_starts
    n_bars = len(bar_starts)

    lattice_locked = fit is not None and fit.ok

    if not lattice_locked:
        reason = fit.reason if fit is not None else "no grid fit"
        p.notes.append(f"grid not locked ({reason}); phrase snapping disabled")

    if djay_bpm is not None:
        ratio = bpm_octave_ratio(beat.bpm, djay_bpm)
        if ratio > 0.02:
            p.notes.append(f"djay says {djay_bpm:.1f}, tracked {beat.bpm:.1f}")

    if bass.loud_ref <= 0:
        if len(beat.downbeats) > 0:
            a_t = float(beat.downbeats[0])
        elif len(beat.beats) > 0:
            a_t = float(beat.beats[0])
        else:
            a_t = 0.0
        p.positions["A"] = a_t
        p.positions["B"] = a_t
        p.notes.append("no kick-band energy detected; first beat only")
        return p, bass

    if n_bars < MIN_BARS:
        a_idx = _first_a_bar(bass.kick_rms, bass.loud_ref)
        a_t = float(bar_starts[a_idx]) if n_bars else 0.0
        p.positions["A"] = a_t
        p.positions["B"] = a_t
        p.notes.append("track too short for bass structure analysis; first beat only")
        return p, bass

    a_idx = _first_a_bar(bass.kick_rms, bass.loud_ref)
    phrase_origin = a_idx
    bass = BassAnalysis(
        bar_starts=bass.bar_starts,
        bar_period=bass.bar_period,
        kick_rms=bass.kick_rms,
        bass_on=bass.bass_on,
        loud_ref=bass.loud_ref,
        phrase_origin=phrase_origin if lattice_locked else None,
        snapped=lattice_locked,
    )

    p.positions["A"] = float(bar_starts[a_idx])
    p.positions["B"] = float(bar_starts[a_idx])

    runs = _runs(bass.bass_on)
    d_raw = _detect_drop(runs)
    e_raw = _detect_break(runs, d_raw)
    f_raw = _detect_return(runs, e_raw)

    d_bar = e_bar = f_bar = g_bar = None

    if d_raw is not None:
        d_bar, d_note = _finalize_event_bar("D", d_raw, phrase_origin, lattice_locked)
        p.positions["D"] = float(bar_starts[d_bar])
        if lattice_locked:
            p.notes.append(d_note)
    else:
        p.notes.append("D (Drop): omitted (no bass-in transition)")
        p.notes.append("C (Buildup): omitted (no D)")
        p.notes.append("E (Breakdown): omitted (no D)")
        p.notes.append("F (2nd Drop): omitted (no E)")

    if d_bar is not None and lattice_locked:
        c_bar = _last_phrase_between(phrase_origin, a_idx, d_bar)
        if c_bar is not None and c_bar > a_idx:
            p.positions["C"] = float(bar_starts[c_bar])
        else:
            p.notes.append("C (Buildup): omitted (nothing between B and D)")
    elif d_bar is not None:
        p.notes.append("C (Buildup): omitted (phrase snapping disabled)")

    if e_raw is not None:
        e_bar, e_note = _finalize_event_bar("E", e_raw, phrase_origin, lattice_locked)
        p.positions["E"] = float(bar_starts[e_bar])
        if lattice_locked:
            p.notes.append(e_note)
    elif d_raw is not None:
        p.notes.append("E (Breakdown): omitted (no candidate after D)")

    if f_raw is not None:
        f_bar, f_note = _finalize_event_bar("F", f_raw, phrase_origin, lattice_locked)
        p.positions["F"] = float(bar_starts[f_bar])
        if lattice_locked:
            p.notes.append(f_note)
    elif e_raw is not None:
        p.notes.append("F (2nd Drop): omitted (no candidate)")
    elif d_bar is not None:
        p.notes.append("F (2nd Drop): omitted (no E)")

    g_bar = _detect_outro_bar(bass.bass_on, n_bars, phrase_origin, lattice_locked)
    if g_bar is not None:
        g_t = float(bar_starts[g_bar])
        latest = max(p.positions.values()) if p.positions else 0.0
        if g_t > latest:
            p.positions["G"] = g_t
            p.positions["H"] = g_t
        else:
            p.notes.append("G/H (Outro): omitted (not after earlier cues)")
    else:
        p.notes.append("G/H (Outro): omitted (no candidate)")

    _check_monotonicity(p)
    return p, bass


def _first_a_bar(kick_rms: np.ndarray, loud_ref: float) -> int:
    thr = A_MIN_FRAC * loud_ref
    for i, rms in enumerate(kick_rms):
        if rms >= thr:
            return i
    return 0


def _detect_drop(runs: list[tuple[int, int, bool]]) -> int | None:
    prev_off_len = 0
    for start, length, is_on in runs:
        if is_on:
            if length >= DROP_MIN_ON_BARS and prev_off_len >= PRE_DROP_MIN_OFF_BARS:
                return start
        else:
            prev_off_len = length
    return None


def _detect_break(runs: list[tuple[int, int, bool]], d_bar: int | None) -> int | None:
    if d_bar is None:
        return None
    for start, length, is_on in runs:
        if not is_on and start > d_bar and length >= BREAK_MIN_OFF_BARS:
            return start
    return None


def _detect_return(runs: list[tuple[int, int, bool]], e_bar: int | None) -> int | None:
    if e_bar is None:
        return None
    for start, length, is_on in runs:
        if is_on and start > e_bar and length >= RETURN_MIN_ON_BARS:
            return start
    return None


def _finalize_event_bar(
    slot: str,
    raw_bar: int,
    origin: int,
    lattice_locked: bool,
) -> tuple[int, str]:
    if not lattice_locked:
        return raw_bar, ""
    final, _ = _snap_bar(raw_bar, origin)
    return final, _event_note(slot, raw_bar, origin, final)


def _detect_outro_bar(
    bass_on: np.ndarray,
    n_bars: int,
    origin: int,
    lattice_locked: bool,
) -> int | None:
    if len(bass_on) == 0:
        return None

    runs = _runs(bass_on)
    trailing_off: int | None = None
    if runs and not runs[-1][2]:
        start, length, _ = runs[-1]
        if length >= OUTRO_MIN_BARS:
            trailing_off = start

    max_outro_bar = n_bars - OUTRO_MIN_BARS

    if trailing_off is not None:
        if lattice_locked:
            final, _ = _snap_bar(trailing_off, origin)
            if final > max_outro_bar:
                return trailing_off
            return final
        return trailing_off

    if lattice_locked:
        candidate = _last_phrase_before_end(origin, n_bars)
        if candidate is not None and candidate > max_outro_bar:
            return max_outro_bar if max_outro_bar > 0 else None
        return candidate

    candidate = n_bars - OUTRO_MIN_BARS
    if candidate > 0:
        return candidate
    return None
