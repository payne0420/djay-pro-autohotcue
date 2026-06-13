"""Kick-band bass events + phrase snapping for ml-bass cue placement."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from autohotcue.analysis import CueProposal
from autohotcue.backends import BeatAnalysis, bpm_octave_ratio
from autohotcue.cuepolicy import _check_monotonicity
from autohotcue.gridlock import GridFit, kick_band

BASS_ON_FRAC = 0.75
BASS_OFF_FRAC = 0.55
DROP_MIN_ON_BARS = 8
PRE_DROP_MIN_OFF_BARS = 4
BREAK_MIN_OFF_BARS = 8
RETURN_MIN_ON_BARS = 8
PHRASE_BARS = 8
SNAP_MAX_BARS = 2
A_MIN_FRAC = 0.25
A_STEADY_FRAC = 0.75
MIN_DROP_FRAC = 0.20
LOUD_REF_PCTL = 90.0
OUTRO_MIN_BARS = 2
AUDIBLE_FRAC = 0.10
OUTRO_TAIL_BARS = 16
LOOP_OUT_BARS = 8
MIN_BARS = 24


@dataclass(frozen=True)
class BassAnalysis:
    bar_starts: np.ndarray
    bar_period: float
    kick_rms: np.ndarray
    full_rms: np.ndarray
    bass_on: np.ndarray
    audible: np.ndarray
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


def _per_bar_rms(
    y: np.ndarray,
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
        out[i] = _rms_segment(y, sr, t0, t1)
    return out


def _per_bar_kick_rms(
    y_kick: np.ndarray,
    sr: int,
    bar_starts: np.ndarray,
    bar_period: float,
    *,
    lattice_locked: bool,
) -> np.ndarray:
    return _per_bar_rms(
        y_kick, sr, bar_starts, bar_period, lattice_locked=lattice_locked,
    )


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
    off8 = _phrase_offset(raw_bar, origin, PHRASE_BARS)
    off16 = _phrase_offset(raw_bar, origin, 16)
    off32 = _phrase_offset(raw_bar, origin, 32)
    prefix = (
        f"{slot}: raw bar {raw_bar}, off8={off8:+d}, off16={off16:+d}, off32={off32:+d}, "
    )
    if final_bar != raw_bar:
        return prefix + f"snapped to bar {final_bar}"
    if abs(off8) > SNAP_MAX_BARS:
        return prefix + "off-phrase (unsnapped)"
    return prefix + f"at bar {final_bar}"


def _last_phrase_between(origin: int, after_bar: int, before_bar: int) -> int | None:
    k_lo = int(np.floor((after_bar - origin) / PHRASE_BARS)) + 1
    k_hi = int(np.ceil((before_bar - origin) / PHRASE_BARS)) - 1
    if k_lo > k_hi:
        return None
    return origin + k_hi * PHRASE_BARS


def _apply_d_a_backstop(
    d_raw: int | None,
    e_raw: int | None,
    f_raw: int | None,
    a_idx: int,
) -> tuple[int | None, int | None, int | None, bool]:
    """When D coincides with A, clear D and dependent E/F raw slots."""
    if d_raw is not None and d_raw == a_idx:
        return None, None, None, True
    return d_raw, e_raw, f_raw, False


def _last_phrase_before_end(origin: int, n_bars: int) -> int | None:
    last_start = n_bars - OUTRO_MIN_BARS
    if last_start <= origin:
        return None
    k = int(np.floor((last_start - origin) / PHRASE_BARS))
    return origin + k * PHRASE_BARS


def _last_phrase_boundary_lte(origin: int, max_bar: int) -> int | None:
    if max_bar < origin:
        return None
    k = int(np.floor((max_bar - origin) / PHRASE_BARS))
    return origin + k * PHRASE_BARS


def _last_phrase_boundary_after(
    origin: int,
    after_bar: int,
    max_bar: int,
) -> int | None:
    if max_bar <= after_bar:
        return None
    k_hi = int(np.floor((max_bar - origin) / PHRASE_BARS))
    for k in range(k_hi, -1, -1):
        bar = origin + k * PHRASE_BARS
        if bar > after_bar:
            return bar
    return None


def _audible_mask(full_rms: np.ndarray) -> np.ndarray:
    n_bars = len(full_rms)
    if n_bars == 0:
        return np.array([], dtype=bool)
    ref = float(np.percentile(full_rms, 95.0))
    if ref <= 0:
        return np.ones(n_bars, dtype=bool)
    thr = AUDIBLE_FRAC * ref
    return full_rms >= thr


def _last_audible_bar(audible: np.ndarray) -> int:
    if len(audible) == 0:
        return 0
    idx = np.flatnonzero(audible)
    if len(idx) == 0:
        return len(audible) - 1
    return int(idx[-1])


def _max_earlier_cue_bar(
    a_idx: int,
    c_bar: int | None,
    d_bar: int | None,
    e_bar: int | None,
    f_bar: int | None,
) -> int:
    bars = [a_idx]
    for b in (c_bar, d_bar, e_bar, f_bar):
        if b is not None:
            bars.append(b)
    return max(bars)


def _trailing_off_start(bass_on: np.ndarray) -> int | None:
    """Start of the final bass-out (the run that lasts to the end of the track)."""
    runs = _runs(bass_on)
    if runs and not runs[-1][2] and runs[-1][1] >= OUTRO_MIN_BARS:
        return runs[-1][0]
    return None


def _g_bar_valid(bar: int, earlier_max: int) -> bool:
    return bar > earlier_max


def _finalize_g_bar(
    raw_bar: int,
    origin: int,
    lattice_locked: bool,
    last_audible: int,
    earlier_max: int,
    *,
    preferred_path: bool,
) -> tuple[int | None, str]:
    if not lattice_locked:
        if _g_bar_valid(raw_bar, earlier_max):
            return raw_bar, ""
        return None, ""

    snapped, _ = _snap_bar(raw_bar, origin)
    candidates: list[int] = []
    if not preferred_path or last_audible - snapped >= OUTRO_TAIL_BARS:
        candidates.append(snapped)
    if raw_bar not in candidates:
        candidates.append(raw_bar)

    for bar in candidates:
        if not _g_bar_valid(bar, earlier_max):
            continue
        if bar > last_audible:
            # Snapping must not push G past the audible tail (or off bar_starts:
            # a boundary can sit at n_bars, one past the last lattice index).
            continue
        if preferred_path and last_audible - bar < OUTRO_TAIL_BARS:
            continue
        return bar, _event_note("G", raw_bar, origin, bar)
    return None, ""


def _loop_window_fully_audible(audible: np.ndarray, start: int) -> bool:
    end = start + LOOP_OUT_BARS
    if end > len(audible):
        return False
    return bool(np.all(audible[start:end]))


def _detect_loop_out_bar(
    audible: np.ndarray,
    last_audible: int,
    origin: int,
    lattice_locked: bool,
) -> int | None:
    max_start = last_audible + 1 - LOOP_OUT_BARS
    if max_start < 0:
        return None
    if lattice_locked:
        k_hi = int(np.floor((max_start - origin) / PHRASE_BARS))
        for k in range(k_hi, -1, -1):
            bar = origin + k * PHRASE_BARS
            if bar <= max_start and _loop_window_fully_audible(audible, bar):
                return bar
        return None
    for bar in range(max_start, -1, -1):
        if _loop_window_fully_audible(audible, bar):
            return bar
    return None


def _legacy_outro_candidates(
    bass_on: np.ndarray,
    n_bars: int,
    origin: int,
    lattice_locked: bool,
) -> list[int]:
    """Short-tail candidates in preference order: final bass-out, then a late boundary."""
    candidates: list[int] = []
    trailing_off = _trailing_off_start(bass_on)
    if trailing_off is not None:
        candidates.append(trailing_off)

    max_outro_bar = n_bars - OUTRO_MIN_BARS
    if lattice_locked:
        boundary = _last_phrase_before_end(origin, n_bars)
        if boundary is not None and boundary > max_outro_bar:
            boundary = max_outro_bar if max_outro_bar > 0 else None
    else:
        boundary = max_outro_bar if max_outro_bar > 0 else None
    if boundary is not None and boundary not in candidates:
        candidates.append(boundary)
    return candidates


def _detect_outro_g_bar(
    bass_on: np.ndarray,
    n_bars: int,
    last_audible: int,
    origin: int,
    lattice_locked: bool,
    earlier_max: int,
) -> tuple[int | None, str | None]:
    max_g_start = last_audible - OUTRO_TAIL_BARS
    if max_g_start >= 0:
        # The outro is the FINAL bass-out — a mid-track bass-out with a section
        # after it is a breakdown, never G.
        trailing_off = _trailing_off_start(bass_on)
        if (
            trailing_off is not None
            and trailing_off <= max_g_start
            and _g_bar_valid(trailing_off, earlier_max)
        ):
            return trailing_off, None

        # Bass runs too late for a 16-bar tail: mix-out point on the phrase
        # grid while the music still plays.
        if lattice_locked:
            phrase_g = _last_phrase_boundary_after(origin, earlier_max, max_g_start)
        else:
            phrase_g = max_g_start if _g_bar_valid(max_g_start, earlier_max) else None
        if phrase_g is not None:
            return phrase_g, None

    for raw in _legacy_outro_candidates(bass_on, n_bars, origin, lattice_locked):
        if _g_bar_valid(raw, earlier_max) and raw <= last_audible:
            return raw, "G/H (Outro): short tail; using legacy placement"
    return None, None


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
    full_rms = _per_bar_rms(
        y, sr, bar_starts, bar_period, lattice_locked=lattice_locked,
    )
    loud = _loud_ref(kick_rms)
    bass_on = _hysteresis_bass_on(kick_rms, loud) if loud > 0 else np.zeros(
        len(bar_starts), dtype=bool,
    )
    audible = _audible_mask(full_rms)
    return BassAnalysis(
        bar_starts=bar_starts,
        bar_period=bar_period,
        kick_rms=kick_rms,
        full_rms=full_rms,
        bass_on=bass_on,
        audible=audible,
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
        return p, replace(bass, snapped=False)

    if n_bars < MIN_BARS:
        a_idx = _first_a_bar(bass.kick_rms, bass.loud_ref)
        a_t = float(bar_starts[a_idx]) if n_bars else 0.0
        p.positions["A"] = a_t
        p.positions["B"] = a_t
        p.notes.append("track too short for bass structure analysis; first beat only")
        return p, replace(bass, snapped=False)

    a_idx = _first_a_bar(bass.kick_rms, bass.loud_ref)
    phrase_origin = a_idx
    bass = BassAnalysis(
        bar_starts=bass.bar_starts,
        bar_period=bass.bar_period,
        kick_rms=bass.kick_rms,
        full_rms=bass.full_rms,
        bass_on=bass.bass_on,
        audible=bass.audible,
        loud_ref=bass.loud_ref,
        phrase_origin=phrase_origin if lattice_locked else None,
        snapped=lattice_locked,
    )

    p.positions["A"] = float(bar_starts[a_idx])
    p.positions["B"] = float(bar_starts[a_idx])

    runs = _runs(bass.bass_on)
    d_raw = _detect_drop(runs, n_bars)
    e_raw = _detect_break(runs, d_raw)
    f_raw = _detect_return(runs, e_raw)
    d_raw, e_raw, f_raw, d_coincides_a = _apply_d_a_backstop(
        d_raw, e_raw, f_raw, a_idx,
    )

    d_bar = e_bar = f_bar = g_bar = h_bar = None
    c_bar = None
    d_note = ""

    if d_raw is not None:
        d_bar, d_note = _finalize_event_bar("D", d_raw, phrase_origin, lattice_locked)
        if d_bar <= a_idx:
            # A raw drop within SNAP_MAX_BARS of A snaps back onto A's bar; a drop
            # on top of the loop-in is not a drop. Omit D and its dependents.
            d_raw = e_raw = f_raw = None
            d_bar = None
            d_coincides_a = True

    if d_coincides_a:
        p.notes.append("D (Drop): omitted (coincides with A)")

    if d_bar is not None:
        p.positions["D"] = float(bar_starts[d_bar])
        if lattice_locked:
            p.notes.append(d_note)
    else:
        if not d_coincides_a:
            p.notes.append("D (Drop): omitted (no qualifying drop; groove only)")
        p.notes.append("C (Buildup): omitted (no D)")
        p.notes.append("E (Breakdown): omitted (no D)")
        p.notes.append("F (2nd Drop): omitted (no E)")

    if d_bar is not None:
        c_bar, c_note = _place_buildup(
            runs,
            d_raw,
            d_bar,
            a_idx,
            phrase_origin,
            lattice_locked,
        )
        if c_bar is not None:
            p.positions["C"] = float(bar_starts[c_bar])
            if c_note:
                p.notes.append(c_note)
        elif c_note:
            p.notes.append(c_note)

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

    last_audible = _last_audible_bar(bass.audible)
    earlier_max = _max_earlier_cue_bar(a_idx, c_bar, d_bar, e_bar, f_bar)

    g_raw, g_path_note = _detect_outro_g_bar(
        bass.bass_on,
        n_bars,
        last_audible,
        phrase_origin,
        lattice_locked,
        earlier_max,
    )
    if g_path_note:
        p.notes.append(g_path_note)

    g_bar: int | None = None
    if g_raw is not None:
        g_bar, g_note = _finalize_g_bar(
            g_raw,
            phrase_origin,
            lattice_locked,
            last_audible,
            earlier_max,
            preferred_path=g_path_note is None,
        )
        if g_bar is None:
            p.notes.append("G/H (Outro): omitted (not after earlier cues)")
        else:
            p.positions["G"] = float(bar_starts[g_bar])
            if lattice_locked and g_note:
                p.notes.append(g_note)

            h_raw = _detect_loop_out_bar(
                bass.audible, last_audible, phrase_origin, lattice_locked,
            )
            if h_raw is not None:
                if lattice_locked:
                    h_bar, _ = _snap_bar(h_raw, phrase_origin)
                    h_note = _event_note("H", h_raw, phrase_origin, h_bar)
                else:
                    h_bar = h_raw
                    h_note = ""
                if h_bar < g_bar:
                    if _loop_window_fully_audible(bass.audible, g_bar):
                        h_bar = g_bar
                        p.notes.append("H (Loop Out): clamped to G")
                    else:
                        h_bar = None
                        p.notes.append(
                            "H (Loop Out): omitted (no audible loop after G)",
                        )
                elif lattice_locked and h_note:
                    p.notes.append(h_note)
                if h_bar is not None:
                    p.positions["H"] = float(bar_starts[h_bar])
    else:
        p.notes.append("G/H (Outro): omitted (no candidate)")

    if "H" in p.positions and "G" not in p.positions:
        del p.positions["H"]
        p.notes.append("H: omitted (G omitted)")

    _check_monotonicity(p)
    return p, bass


def _first_a_bar(kick_rms: np.ndarray, loud_ref: float) -> int:
    thr = A_MIN_FRAC * loud_ref
    for i, rms in enumerate(kick_rms):
        if rms < thr:
            continue
        if i + 1 < len(kick_rms) and rms < A_STEADY_FRAC * kick_rms[i + 1]:
            continue
        return i
    return 0


def _detect_drop(runs: list[tuple[int, int, bool]], n_bars: int) -> int | None:
    prev_off_len = 0
    on_run_count = 0
    min_drop_bar = MIN_DROP_FRAC * n_bars
    for start, length, is_on in runs:
        if is_on:
            if length >= DROP_MIN_ON_BARS and prev_off_len >= PRE_DROP_MIN_OFF_BARS:
                is_return = on_run_count >= 1
                is_late = start >= min_drop_bar
                if is_return or is_late:
                    return start
            on_run_count += 1
        else:
            prev_off_len = length
    return None


def _preceding_off_start(runs: list[tuple[int, int, bool]], d_bar: int) -> int | None:
    prev_off_len = 0
    for start, length, is_on in runs:
        if is_on and start == d_bar:
            if prev_off_len > 0:
                return start - prev_off_len
            return None
        if not is_on:
            prev_off_len = length
    return None


def _place_buildup(
    runs: list[tuple[int, int, bool]],
    d_raw: int,
    d_bar: int,
    a_idx: int,
    origin: int,
    lattice_locked: bool,
) -> tuple[int | None, str | None]:
    off_start = _preceding_off_start(runs, d_raw)
    if off_start is not None and off_start > a_idx:
        return _finalize_c_event(off_start, d_bar, a_idx, origin, lattice_locked)

    if not lattice_locked:
        return None, "C (Buildup): omitted (phrase snapping disabled)"

    c_bar = _last_phrase_between(origin, a_idx, d_bar)
    if c_bar is not None and c_bar > a_idx:
        return c_bar, None
    return None, "C (Buildup): omitted (nothing between B and D)"


def _finalize_c_event(
    raw_bar: int,
    d_bar: int,
    a_idx: int,
    origin: int,
    lattice_locked: bool,
) -> tuple[int | None, str | None]:
    if not lattice_locked:
        if a_idx < raw_bar < d_bar:
            return raw_bar, None
        return None, "C (Buildup): omitted (nothing between B and D)"

    snapped, _ = _snap_bar(raw_bar, origin)
    for candidate in (snapped, raw_bar):
        if a_idx < candidate < d_bar:
            note = _event_note("C", raw_bar, origin, candidate)
            return candidate, note
    return None, "C (Buildup): omitted (nothing between B and D)"


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
