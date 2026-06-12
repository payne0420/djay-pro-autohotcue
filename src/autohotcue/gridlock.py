"""Pure beat-grid fitting and cue snapping for djay grid-lock."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

BEAT_FIT_MAX = 0.10
SPLICE_STEP_MAX = 0.15
BAR_RESID_STD_MAX = 0.15
SPLICE_WINDOW_S = 60.0
SPLICE_HOP_S = 30.0
SPLICE_MIN_BEATS = 10
BPM_MIN = 60.0
BPM_MAX = 200.0
DJAY_BPM_TOLERANCE = 0.1
TEMPO_REFINE_STEP = 0.01
TEMPO_SEARCH_RANGE = 2.0
TEMPO_SEARCH_STEP = 0.25
ROTATION_NOVELTY_TOP = 12
KEEP_FLOOR_FRAC = 0.25
PARITY_MARGIN = 0.25
PARITY_MIN_CLUSTER = 4
MIN_KEPT_BEATS = 32
MIN_KEPT_DOWNBEATS = 8
MIN_DURATION_S = 30.0
KICK_BAND_LO_HZ = 30.0
KICK_BAND_HI_HZ = 150.0

_kick_sos: dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class GridFit:
    bpm: float
    render_bpm: float
    anchor_s: float
    beat_fit: float
    bar_resid_std: float
    splice_jump: float
    ok: bool
    reason: str


def _kick_sos_for_sr(sr: int) -> np.ndarray:
    if sr not in _kick_sos:
        _kick_sos[sr] = butter(
            4,
            [KICK_BAND_LO_HZ, KICK_BAND_HI_HZ],
            btype="bandpass",
            fs=sr,
            output="sos",
        )
    return _kick_sos[sr]


def _kick_band(y: np.ndarray, sr: int) -> np.ndarray:
    return np.asarray(sosfilt(_kick_sos_for_sr(sr), y), dtype=np.float64)


def kick_band(y: np.ndarray, sr: int) -> np.ndarray:
    """Public 30-150 Hz bandpass used by both grid fitting and bassline events."""
    return _kick_band(y, sr)


def _rms(y: np.ndarray, i0: int, i1: int) -> float:
    seg = y[i0:i1]
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg * seg)))


def _rms_window(y: np.ndarray, sr: int, t0: float, t1: float) -> float:
    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    return _rms(y, i0, i1)


def _circular_variance(times: np.ndarray, period: float, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0 or period <= 0:
        return 1.0
    z = np.sum(w * np.exp(2j * np.pi * times / period))
    return float(1.0 - abs(z) / w.sum())


def _circular_mean_phase(times: np.ndarray, period: float, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0 or period <= 0:
        return 0.0
    z = np.sum(w * np.exp(2j * np.pi * times / period))
    phase = float(np.angle(z) / (2 * np.pi) * period)
    if phase < 0:
        phase += period
    return phase


def _rough_bpm(beats: np.ndarray) -> float:
    if len(beats) < 2:
        return 120.0
    intervals = np.diff(beats)
    med = float(np.median(intervals))
    return 60.0 / med if med > 0 else 120.0


def _energy_weights(
    y: np.ndarray,
    sr: int,
    times: np.ndarray,
    bpm_est: float,
    *,
    bar_window: bool,
) -> np.ndarray:
    win = (4.0 * 60.0 / bpm_est) if bar_window else (60.0 / bpm_est)
    return np.array([_rms_window(y, sr, t, t + win) for t in times], dtype=np.float64)


def _kept(times: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(times) == 0:
        return times, weights
    med = float(np.median(weights))
    wmax = float(np.max(weights))
    threshold = max(med, KEEP_FLOOR_FRAC * wmax)
    mask = weights >= threshold
    if not mask.any():
        mask = np.ones(len(times), dtype=bool)
    return times[mask], weights[mask]


def _circular_phase_dist(a: float, b: float, period: float) -> float:
    d = abs(a - b)
    return min(d, period - d)


def _score_bpm(bpm: float, beats: np.ndarray, weights: np.ndarray) -> float:
    period = 60.0 / bpm
    return _circular_variance(beats, period, weights)


def _refine_bpm(
    center: float,
    beats: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    lo = max(BPM_MIN, center - TEMPO_SEARCH_RANGE)
    hi = min(BPM_MAX, center + TEMPO_SEARCH_RANGE)
    grid = np.arange(lo, hi + TEMPO_REFINE_STEP * 0.5, TEMPO_REFINE_STEP)
    best_bpm, best_score = center, _score_bpm(center, beats, weights)
    for bpm in grid:
        score = _score_bpm(bpm, beats, weights)
        if score < best_score:
            best_bpm, best_score = float(bpm), score
    return best_bpm, best_score


def _regression_bpm(beats: np.ndarray, kept_beats: np.ndarray) -> float | None:
    """Linear fit of time vs beat index; slope is seconds per beat."""
    if len(kept_beats) < 2:
        return None
    t0 = float(kept_beats[0])
    beat_period = 60.0 / _rough_bpm(kept_beats)
    grid = beats[beats >= t0 - 0.5 * beat_period]
    if len(grid) < 2:
        return None
    slope = float(np.polyfit(np.arange(len(grid)), grid, 1)[0])
    if slope <= 0:
        return None
    reg = 60.0 / slope
    if BPM_MIN <= reg <= BPM_MAX:
        return reg
    return None


def _tempo_candidates(djay_bpm: float | None, kept_beats: np.ndarray, beats: np.ndarray) -> list[float]:
    bases: list[float] = []
    if djay_bpm is not None:
        for k in (0.5, 1.0, 2.0):
            b = djay_bpm * k
            if BPM_MIN <= b <= BPM_MAX:
                bases.append(b)
    reg = _regression_bpm(beats, kept_beats)
    if reg is not None:
        bases.append(reg)
    if not bases:
        bases.append(_rough_bpm(kept_beats))
    cands: set[float] = set()
    for base in bases:
        for bpm in np.arange(
            max(BPM_MIN, base - TEMPO_SEARCH_RANGE),
            min(BPM_MAX, base + TEMPO_SEARCH_RANGE) + TEMPO_SEARCH_STEP * 0.5,
            TEMPO_SEARCH_STEP,
        ):
            cands.add(round(float(bpm), 10))
    return sorted(cands)


def _djay_multiples(djay_bpm: float) -> list[float]:
    return [djay_bpm * k for k in (0.5, 1.0, 2.0)]


def _is_half_double(fitted_bpm: float, render_bpm: float) -> bool:
    ratio = fitted_bpm / render_bpm
    return abs(ratio - 2.0) < 0.2 or abs(ratio - 0.5) < 0.2


def _needs_parity_gate(fitted_bpm: float, render_bpm: float) -> bool:
    """Render bar spans multiple fitted bars (djay half-time, fitted 2× render)."""
    return _is_half_double(fitted_bpm, render_bpm) and fitted_bpm > render_bpm


def _circular_phase_dist_frac(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def _splice_window_starts(t_end: float) -> list[float]:
    """60s windows on 30s hops plus one final window flush with the track end."""
    starts: list[float] = []
    t0 = 0.0
    while t0 < t_end:
        if t0 + SPLICE_WINDOW_S <= t_end + 1e-9:
            starts.append(t0)
        t0 += SPLICE_HOP_S
    flush = max(0.0, t_end - SPLICE_WINDOW_S)
    if not any(abs(s - flush) < 1e-9 for s in starts):
        starts.append(flush)
    return sorted(set(starts))


def _splice_jump(
    beats: np.ndarray,
    period: float,
    weights: np.ndarray,
    duration: float,
) -> float:
    if len(beats) < SPLICE_MIN_BEATS:
        return 0.0
    t_end = max(float(beats[-1]), duration)
    mean_phases: list[float] = []
    beat_phases: list[np.ndarray] = []
    for t0 in _splice_window_starts(t_end):
        t1 = min(t0 + SPLICE_WINDOW_S, t_end)
        mask = (beats >= t0) & (beats < t1)
        if int(mask.sum()) >= SPLICE_MIN_BEATS:
            wb = beats[mask]
            ww = weights[mask]
            mean_phases.append(_circular_mean_phase(wb, period, ww) / period)
            beat_phases.append((wb % period) / period)
    if len(mean_phases) < 2:
        return 0.0
    max_jump = 0.0
    for i in range(len(mean_phases)):
        for j in range(i + 1, len(mean_phases)):
            max_jump = max(
                max_jump,
                _circular_phase_dist_frac(mean_phases[i], mean_phases[j]),
            )
            for a in beat_phases[i]:
                for b in beat_phases[j]:
                    max_jump = max(max_jump, _circular_phase_dist_frac(float(a), float(b)))
    return max_jump


def _parity_clusters(
    kept_downbeats: np.ndarray,
    kept_weights: np.ndarray,
    bar_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    half = bar_period * 0.5
    c0 = _circular_mean_phase(kept_downbeats, bar_period, kept_weights)
    c1 = (c0 + half) % bar_period

    for _ in range(2):
        m0 = np.array([
            _circular_phase_dist(t % bar_period, c0, bar_period)
            <= _circular_phase_dist(t % bar_period, c1, bar_period)
            for t in kept_downbeats
        ])
        m1 = ~m0
        if m0.any():
            c0 = _circular_mean_phase(kept_downbeats[m0], bar_period, kept_weights[m0])
        if m1.any():
            c1 = _circular_mean_phase(kept_downbeats[m1], bar_period, kept_weights[m1])

    return m0, m1


def _half_double_bar_phase(
    kept_downbeats: np.ndarray,
    kept_weights: np.ndarray,
    bar_period: float,
) -> tuple[float | None, str]:
    """Gate half/double render-bar parity; never guess."""
    m0, m1 = _parity_clusters(kept_downbeats, kept_weights, bar_period)
    w0 = float(kept_weights[m0].sum())
    w1 = float(kept_weights[m1].sum())
    total = w0 + w1

    if w0 >= w1:
        win, w_win, w_lose, n_win = m0, w0, w1, int(m0.sum())
    else:
        win, w_win, w_lose, n_win = m1, w1, w0, int(m1.sum())

    if (
        total <= 0
        or w0 <= 0
        or w1 <= 0
        or n_win < PARITY_MIN_CLUSTER
        or (w_win - w_lose) / total < PARITY_MARGIN
    ):
        return None, "ambiguous half-time bar phase"

    sub_db = kept_downbeats[win]
    sub_w = kept_weights[win]
    anchor = _circular_mean_phase(sub_db, bar_period, sub_w) % bar_period
    return anchor, ""


def _bar_phase(
    kept_downbeats: np.ndarray,
    kept_weights: np.ndarray,
    fitted_bpm: float,
    render_bpm: float,
    bar_period: float,
) -> tuple[float | None, str]:
    """Weighted circular mean on the render bar period; gate half/double parity."""
    if len(kept_downbeats) == 0:
        return None, "ambiguous half-time bar phase"

    if _needs_parity_gate(fitted_bpm, render_bpm):
        return _half_double_bar_phase(kept_downbeats, kept_weights, bar_period)

    anchor = _circular_mean_phase(kept_downbeats, bar_period, kept_weights) % bar_period
    return anchor, ""


def _bar_residual_std(downbeats: np.ndarray, period: float, anchor: float) -> float:
    if len(downbeats) == 0:
        return 0.0
    resid = (downbeats - anchor) / period
    resid = resid - np.round(resid)
    return float(np.std(resid))


def _rotation_anchor(
    y: np.ndarray,
    sr: int,
    bpm: float,
    render_bpm: float,
    beat_phase: float,
) -> float:
    beat_period = 60.0 / bpm
    bar_period = 4.0 * 60.0 / render_bpm
    duration = len(y) / sr
    n_beats = max(0, int((duration - beat_phase) / beat_period) + 2)
    novelty = np.zeros(n_beats, dtype=np.float64)
    for i in range(n_beats):
        t = beat_phase + i * beat_period
        if t < 0 or t >= duration:
            continue
        before = _rms_window(y, sr, t - bar_period, t)
        after = _rms_window(y, sr, t, t + bar_period)
        novelty[i] = after - before
    top = np.argsort(novelty)[-ROTATION_NOVELTY_TOP:]
    scores = np.zeros(4, dtype=np.float64)
    for k in top:
        scores[k % 4] += novelty[k]
    best_r = int(np.argmax(scores))
    anchor = (beat_phase + best_r * beat_period) % bar_period
    if abs(anchor - bar_period) < 1e-6:
        anchor = 0.0
    return float(anchor)


def _fail(
    *,
    bpm: float,
    render_bpm: float,
    beat_fit: float,
    splice: float,
    reason: str,
) -> GridFit:
    return GridFit(
        bpm=bpm,
        render_bpm=render_bpm,
        anchor_s=0.0,
        beat_fit=beat_fit,
        bar_resid_std=1.0,
        splice_jump=splice,
        ok=False,
        reason=reason,
    )


def fit_grid(
    y: np.ndarray,
    sr: int,
    beats: np.ndarray,
    downbeats: np.ndarray,
    djay_bpm: float | None,
) -> GridFit:
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    duration = len(y) / sr

    if len(beats) < 2:
        return _fail(
            bpm=120.0,
            render_bpm=djay_bpm or 120.0,
            beat_fit=1.0,
            splice=0.0,
            reason="beats do not fit a straight grid",
        )

    y_kick = _kick_band(y, sr)
    bpm_est = _rough_bpm(beats)
    beat_w = _energy_weights(y_kick, sr, beats, bpm_est, bar_window=False)
    db_w = _energy_weights(y_kick, sr, downbeats, bpm_est, bar_window=True) if len(downbeats) else np.array([])
    kept_beats, kept_beat_w = _kept(beats, beat_w)
    if len(downbeats):
        kept_db, kept_db_w = _kept(downbeats, db_w)
    else:
        kept_db, kept_db_w = downbeats, db_w

    if djay_bpm is not None:
        render_bpm = djay_bpm
    else:
        render_bpm = 120.0

    if (
        len(kept_beats) < MIN_KEPT_BEATS
        or len(kept_db) < MIN_KEPT_DOWNBEATS
        or duration < MIN_DURATION_S
    ):
        return _fail(
            bpm=bpm_est,
            render_bpm=render_bpm,
            beat_fit=1.0,
            splice=0.0,
            reason="insufficient evidence",
        )

    best_bpm, best_score = 120.0, 1.0
    for cand in _tempo_candidates(djay_bpm, kept_beats, beats):
        bpm, score = _refine_bpm(cand, kept_beats, kept_beat_w)
        if score < best_score:
            best_bpm, best_score = bpm, score

    if djay_bpm is not None:
        render_bpm = djay_bpm
    else:
        render_bpm = best_bpm

    winning_period = 60.0 / best_bpm
    splice = _splice_jump(kept_beats, winning_period, kept_beat_w, duration)

    if splice > SPLICE_STEP_MAX:
        return _fail(
            bpm=best_bpm,
            render_bpm=render_bpm,
            beat_fit=best_score,
            splice=splice,
            reason="phase jumps between sections (spliced edit?)",
        )

    if best_score > BEAT_FIT_MAX:
        return _fail(
            bpm=best_bpm,
            render_bpm=render_bpm,
            beat_fit=best_score,
            splice=splice,
            reason="beats do not fit a straight grid",
        )

    if djay_bpm is not None:
        multiples = _djay_multiples(djay_bpm)
        if not any(abs(best_bpm - m) <= DJAY_BPM_TOLERANCE + 1e-6 for m in multiples):
            return _fail(
                bpm=best_bpm,
                render_bpm=render_bpm,
                beat_fit=best_score,
                splice=splice,
                reason="fitted tempo disagrees with djay's BPM",
            )

    bar_period = 4.0 * 60.0 / render_bpm
    anchor, bar_reason = _bar_phase(
        kept_db, kept_db_w, best_bpm, render_bpm, bar_period,
    )

    if anchor is None:
        return _fail(
            bpm=best_bpm,
            render_bpm=render_bpm,
            beat_fit=best_score,
            splice=splice,
            reason=bar_reason,
        )

    bar_std = _bar_residual_std(kept_db, bar_period, anchor)

    if _needs_parity_gate(best_bpm, render_bpm):
        if bar_std > BAR_RESID_STD_MAX:
            return _fail(
                bpm=best_bpm,
                render_bpm=render_bpm,
                beat_fit=best_score,
                splice=splice,
                reason="ambiguous half-time bar phase",
            )
    elif bar_std > BAR_RESID_STD_MAX:
        beat_phase = (
            _circular_mean_phase(kept_beats, winning_period, kept_beat_w)
            if len(kept_beats) >= 1
            else 0.0
        )
        anchor = _rotation_anchor(y_kick, sr, best_bpm, render_bpm, beat_phase)
        bar_std = _bar_residual_std(kept_db, bar_period, anchor)
        anchor = float(anchor % bar_period)
    else:
        anchor = float(anchor)

    return GridFit(
        bpm=best_bpm,
        render_bpm=render_bpm,
        anchor_s=anchor,
        beat_fit=best_score,
        bar_resid_std=bar_std,
        splice_jump=splice,
        ok=True,
        reason="",
    )


def _nearest_nonnegative_lattice(t: float, anchor: float, beat_period: float) -> float:
    n_min = int(np.ceil(-anchor / beat_period - 1e-12)) if anchor > 0 else 0
    rel = (t - anchor) / beat_period
    candidates = {
        max(n_min, int(np.floor(rel))),
        max(n_min, int(np.ceil(rel - 1e-12))),
    }
    best_n = min(candidates, key=lambda k: abs(t - (anchor + k * beat_period)))
    return anchor + best_n * beat_period


def snap_cues(positions: dict[str, float], fit: GridFit) -> dict[str, float]:
    if not fit.ok:
        return dict(positions)
    beat_period = 60.0 / fit.bpm
    anchor = fit.anchor_s
    out: dict[str, float] = {}
    for pad, t in positions.items():
        out[pad] = _nearest_nonnegative_lattice(t, anchor, beat_period)
    return out
