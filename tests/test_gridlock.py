"""Unit tests for grid-lock fitting, snapping, and beatGridEdits encoding."""
from __future__ import annotations

import numpy as np
import pytest

from autohotcue import djaydb, tsaf
from autohotcue.gridlock import (
    BEAT_FIT_MAX,
    DJAY_BPM_TOLERANCE,
    GridFit,
    SPLICE_STEP_MAX,
    fit_grid,
    snap_cues,
)


SR = 44100
BPM = 122.0
BEAT_S = 60.0 / BPM
BAR_S = 4 * BEAT_S


def _synth_lattice(
    bpm: float,
    bars: int,
    *,
    true_anchor: float = 1.547,
    jitter_ms: float = 10.0,
    intro_beats: np.ndarray | None = None,
    intro_downbeats: np.ndarray | None = None,
    phase_step_at: float | None = None,
    phase_step_beats: float = 0.0,
    energy_jump_at: float | None = None,
    duration_s: float | None = None,
    trailing_silence_s: float = 2.0,
    kick: float = 0.6,
    pad_intro_s: float | None = None,
    pad_rms: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build mono audio + beat/downbeat arrays on a straight lattice."""
    beat_s = 60.0 / bpm
    bar_s = 4 * beat_s
    if duration_s is None:
        duration_s = bars * bar_s + true_anchor + trailing_silence_s
    n = int(duration_s * SR)
    y = np.zeros(n, dtype=np.float32)

    if pad_intro_s is not None:
        pad_n = int(pad_intro_s * SR)
        rng = np.random.default_rng(99)
        y[:pad_n] = rng.standard_normal(pad_n).astype(np.float32) * pad_rms

    beats = np.arange(true_anchor, duration_s, beat_s)
    if jitter_ms > 0:
        rng = np.random.default_rng(42)
        beats = beats + rng.uniform(-jitter_ms, jitter_ms, len(beats)) / 1000.0

    if phase_step_at is not None:
        idx = int(np.searchsorted(beats, phase_step_at))
        if 0 < idx < len(beats):
            beats[idx:] += phase_step_beats * beat_s

    if intro_beats is not None:
        beats = np.sort(np.concatenate([intro_beats, beats]))

    downbeats = np.arange(true_anchor, duration_s, bar_s)
    if intro_downbeats is not None:
        downbeats = np.sort(np.concatenate([intro_downbeats, downbeats]))

    intro_set: set[float] = set()
    if intro_beats is not None:
        intro_set = {float(t) for t in intro_beats}
    for t in beats:
        if t in intro_set or (intro_beats is not None and t < true_anchor - 1e-6):
            continue
        i0 = int(t * SR)
        i1 = min(n, i0 + int(0.04 * SR))
        y[i0:i1] += kick

    if energy_jump_at is not None:
        i0 = int(energy_jump_at * SR)
        i1 = min(n, i0 + int(bar_s * SR))
        y[i0:i1] += 0.9

    return y, beats, downbeats


def _boost_render_downbeats(
    y: np.ndarray,
    downbeats: np.ndarray,
    true_anchor: float,
    render_bpm: float,
    *,
    boost: float = 0.08,
) -> None:
    """Extra kick energy on render-bar downbeats so parity gate can pass."""
    render_bar = 4 * 60 / render_bpm
    ref = true_anchor % render_bar
    n = len(y)
    for t in downbeats:
        phase = (float(t) - ref) % render_bar
        if phase < 0.05 or abs(phase - render_bar) < 0.05:
            i0 = int(t * SR)
            i1 = min(n, i0 + int(0.04 * SR))
            y[i0:i1] += boost


def _half_time_lattice(
    bars: int,
    true_anchor: float,
    *,
    jitter_ms: float = 0.0,
    intro_downbeats: np.ndarray | None = None,
    trailing_silence_s: float = 2.0,
    boost_render: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, beats, downbeats = _synth_lattice(
        160.0,
        bars,
        true_anchor=true_anchor,
        jitter_ms=jitter_ms,
        intro_downbeats=intro_downbeats,
        trailing_silence_s=trailing_silence_s,
    )
    if boost_render:
        _boost_render_downbeats(y, downbeats, true_anchor, 80.0)
    return y, beats, downbeats


def test_lattice_with_quiet_intro_beats():
    """122 BPM lattice; low-energy intro beats must not move tempo or anchor."""
    true_anchor = 1.547
    intro = np.arange(0.0, 0.45, 60.0 / 100.0)
    y, beats, downbeats = _synth_lattice(
        BPM, 64, true_anchor=true_anchor, intro_beats=intro
    )

    fit_djay = fit_grid(y, SR, beats, downbeats, djay_bpm=122.0)
    fit_none = fit_grid(y, SR, beats, downbeats, djay_bpm=None)

    for fit in (fit_djay, fit_none):
        assert fit.ok, fit.reason
        assert fit.bpm == pytest.approx(122.0, abs=0.01)
        assert fit.anchor_s == pytest.approx(true_anchor, abs=0.015)


def test_half_time_djay_bpm():
    """djay stores 80 BPM; true lattice is 160 on the render bar grid."""
    true_anchor = 1.75
    y, beats, downbeats = _half_time_lattice(40, true_anchor)

    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    assert fit.ok, fit.reason
    assert fit.bpm == pytest.approx(160.0, abs=0.01)
    assert fit.render_bpm == pytest.approx(80.0)
    render_bar = 4 * 60 / 80.0
    assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)


def test_regression_bpm_without_djay():
    """Regression tempo candidate alone must recover BPM when djay_bpm is unknown."""
    y, beats, downbeats = _synth_lattice(BPM, 48, true_anchor=0.5, jitter_ms=5.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=None)
    assert fit.ok, fit.reason
    assert fit.bpm == pytest.approx(BPM, abs=0.01)


def test_half_time_render_bar_parity():
    """160 BPM lattice rendered at djay 80: pick the correct render-bar parity."""
    true_anchor = 1.75
    render_bar = 4 * 60 / 80.0
    y, beats, downbeats = _half_time_lattice(40, true_anchor)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    assert fit.ok, fit.reason
    assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)
    wrong = (true_anchor + 4 * 60 / 160.0) % render_bar
    assert fit.anchor_s != pytest.approx(wrong, abs=0.05)


@pytest.mark.parametrize("jitter_ms", [0.0, 10.0])
def test_half_time_parity_ignores_silent_intro_downbeat(jitter_ms: float):
    """Quiet hallucinated intro downbeat must not steer half/double parity."""
    true_anchor = 1.75
    render_bar = 4 * 60 / 80.0
    y, beats, downbeats = _half_time_lattice(
        40,
        true_anchor,
        jitter_ms=jitter_ms,
        intro_downbeats=np.array([0.26]),
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    if fit.ok:
        assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)
        assert fit.anchor_s != pytest.approx(0.26, abs=0.05)
    else:
        assert fit.reason == "ambiguous half-time bar phase"


@pytest.mark.parametrize("bars", [8, 12, 20])
def test_half_time_parity_survives_quiet_intro_flood(bars: int):
    """Many quiet intro downbeats must not flip half/double render-bar parity."""
    true_anchor = 40.25
    render_bar = 4 * 60 / 80.0
    bar_s_160 = 4 * 60 / 160.0
    duration_s = bars * bar_s_160 + true_anchor + 2.0
    intro = np.arange(0.26, duration_s, bar_s_160)
    y, beats, downbeats = _synth_lattice(
        160.0,
        bars,
        true_anchor=true_anchor,
        intro_downbeats=intro,
        jitter_ms=0.0,
        duration_s=duration_s,
    )
    _boost_render_downbeats(y, downbeats, true_anchor, 80.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    if fit.ok:
        assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)
    else:
        assert fit.reason in ("ambiguous half-time bar phase", "insufficient evidence")


def test_half_time_parity_survives_missing_downbeat():
    """A single missed downbeat must not flip parity via index alignment."""
    true_anchor = 1.75
    y, beats, downbeats = _half_time_lattice(40, true_anchor, jitter_ms=0.0)
    baseline = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    assert baseline.ok, baseline.reason
    missing = np.delete(downbeats, 0)
    fit = fit_grid(y, SR, beats, missing, djay_bpm=80.0)
    assert fit.ok, fit.reason
    assert fit.anchor_s == pytest.approx(baseline.anchor_s, abs=0.015)


def test_rotation_preserves_fitted_beat_phase():
    """Rotation fallback must stay on the fitted beat lattice, not absolute zero."""
    true_anchor = 1.547
    y, beats, downbeats = _synth_lattice(BPM, 64, true_anchor=true_anchor, jitter_ms=0.0)
    jump_t = true_anchor + 16 * BAR_S
    i0 = int(jump_t * SR)
    i1 = min(len(y), i0 + int(BAR_S * SR))
    y[i0:i1] += 0.9

    rng = np.random.default_rng(11)
    db_times = np.arange(true_anchor, len(y) / SR, BAR_S)
    downbeats = db_times + rng.uniform(-0.55, 0.55, len(db_times))

    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert fit.ok, fit.reason
    render_bar = 4 * 60 / BPM
    assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.05)


def test_spliced_phase_step_gate():
    """Mid-track phase jump trips the splice gate (overlapping windows)."""
    duration = 180.0
    y, beats, downbeats = _synth_lattice(
        BPM,
        64,
        duration_s=duration,
        phase_step_at=75.0,
        phase_step_beats=0.5,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert not fit.ok
    assert fit.reason == "phase jumps between sections (spliced edit?)"
    assert fit.splice_jump > SPLICE_STEP_MAX + 0.02


def test_splice_midwindow_small_step_gates():
    """0.2-beat mid-window splice at 122 BPM must not pass with ok=True."""
    duration = 180.0
    y, beats, downbeats = _synth_lattice(
        BPM,
        64,
        duration_s=duration,
        phase_step_at=90.0,
        phase_step_beats=0.2,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert not fit.ok


def test_beat_fit_gate_white_noise():
    """Random beat times on white noise fail the beat_fit gate."""
    rng = np.random.default_rng(7)
    duration = 120.0
    n = int(duration * SR)
    y = rng.standard_normal(n).astype(np.float32) * 0.5
    beats = np.sort(rng.uniform(0, duration, 200))
    downbeats = beats[::4]

    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=120.0)
    assert not fit.ok
    assert fit.beat_fit > BEAT_FIT_MAX


def test_snap_cues_on_lattice():
    bpm = 126.0
    anchor = 1.547
    beat_s = 60.0 / bpm
    fit = GridFit(
        bpm=bpm,
        render_bpm=bpm,
        anchor_s=anchor,
        beat_fit=0.01,
        bar_resid_std=0.01,
        splice_jump=0.0,
        ok=True,
        reason="",
    )
    positions = {"A": anchor + 3.7 * beat_s, "D": anchor + 16.2 * beat_s, "G": 200.0}
    snapped = snap_cues(positions, fit)
    assert set(snapped) == {"A", "D", "G"}
    for t in snapped.values():
        k = round((t - anchor) / beat_s)
        assert t == pytest.approx(anchor + k * beat_s, abs=1e-9)


def test_snap_cues_near_zero_nonnegative():
    """Cues near t=0 must snap to a non-negative lattice point."""
    bpm = 122.0
    anchor = 1.547
    beat_s = 60.0 / bpm
    fit = GridFit(
        bpm=bpm,
        render_bpm=bpm,
        anchor_s=anchor,
        beat_fit=0.01,
        bar_resid_std=0.01,
        splice_jump=0.0,
        ok=True,
        reason="",
    )
    snapped = snap_cues({"A": 0.05, "B": 0.0, "C": 0.3}, fit)
    for t in snapped.values():
        assert t >= 0.0


def test_build_beat_grid_edits_roundtrip():
    root = tsaf.Obj("ADCMediaItemUserData")
    root.fields = [
        ("uuid", "abc"),
        ("beatGridEdits", djaydb.build_beat_grid_edits(1.547)),
    ]
    doc = tsaf.Document((3, 3), root)
    blob = tsaf.serialize(doc)
    assert tsaf.serialize(tsaf.parse(blob)) == blob

    obj = tsaf.parse(blob).root.get("beatGridEdits")
    assert obj.classname == "ADCBeatGridEdits"
    names = [name for name, _ in obj.fields]
    assert names == [
        "firstDownbeatPosition",
        "nrOfBeatShift",
        "downbeatMarkers",
        "firstGridSegmentTempoExponent",
        "lastGridSegmentTempoExponent",
        "fractionalBeatShift",
    ]
    assert obj.get("firstDownbeatPosition").value == pytest.approx(1.547)
    assert isinstance(obj.get("nrOfBeatShift"), tsaf.Marker)
    assert obj.get("nrOfBeatShift").tag == tsaf.TAG_M2E
    assert obj.get("downbeatMarkers").tag == tsaf.TAG_ARRAY_A
    assert obj.get("downbeatMarkers").items == []
    assert obj.get("fractionalBeatShift").value == pytest.approx(0.0)


def test_rotation_fallback_energy_jump():
    """Ambiguous downbeats but a strong energy jump on a known downbeat."""
    true_anchor = 0.0
    y, beats, downbeats = _synth_lattice(BPM, 64, true_anchor=true_anchor, jitter_ms=0.0)
    jump_t = 32 * BAR_S
    i0 = int(jump_t * SR)
    i1 = min(len(y), i0 + int(BAR_S * SR))
    y[i0:i1] += 0.95

    rng = np.random.default_rng(99)
    db_times = np.arange(true_anchor, len(y) / SR, BAR_S)
    downbeats = db_times + rng.uniform(-0.6, 0.6, len(db_times))

    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=122.0)
    assert fit.ok, fit.reason
    assert fit.anchor_s == pytest.approx(true_anchor % (4 * 60 / BPM), abs=0.05)


def test_half_time_wrong_parity_louder_bar_gates():
    """One wrong-parity bar 13% louder than 39 equal bars must not write wrong+ok."""
    true_anchor = 1.75
    render_bar = 4 * 60 / 80.0
    bar_s_160 = 4 * 60 / 160.0
    y, beats, downbeats = _half_time_lattice(40, true_anchor, boost_render=False)
    wrong_times = downbeats[1::2]
    for t in wrong_times:
        i0 = int(t * SR)
        i1 = min(len(y), i0 + int(bar_s_160 * SR))
        y[i0:i1] += 0.6 * 1.13
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    assert not fit.ok
    assert fit.reason == "ambiguous half-time bar phase"


def test_half_time_missed_first_downbeat_gates_or_correct():
    """Missing the first downbeat must gate out or keep the correct anchor."""
    true_anchor = 1.75
    render_bar = 4 * 60 / 80.0
    y, beats, downbeats = _half_time_lattice(40, true_anchor, boost_render=True)
    missing = np.delete(downbeats, 0)
    fit = fit_grid(y, SR, beats, missing, djay_bpm=80.0)
    if fit.ok:
        assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.02)
    else:
        assert fit.reason == "ambiguous half-time bar phase"


@pytest.mark.parametrize("trailing_s", [0.3, 2.0])
def test_half_time_trailing_silence_invariant(trailing_s: float):
    """Trailing silence length alone must not flip parity."""
    true_anchor = 1.75
    render_bar = 4 * 60 / 80.0
    y, beats, downbeats = _half_time_lattice(
        40, true_anchor, trailing_silence_s=trailing_s, boost_render=True
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=80.0)
    if fit.ok:
        assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.02)
    else:
        assert fit.reason == "ambiguous half-time bar phase"


def test_pad_intro_does_not_move_anchor():
    """Full-band pad intro without kick-band energy must not move anchor."""
    true_anchor = 1.547
    y, beats, downbeats = _synth_lattice(
        BPM,
        64,
        true_anchor=true_anchor,
        pad_intro_s=8.0,
        pad_rms=0.55,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert fit.ok, fit.reason
    assert fit.anchor_s == pytest.approx(true_anchor, abs=0.02)


def test_insufficient_evidence_short_clip():
    """Very short clips must fail the minimum-evidence gate."""
    y, beats, downbeats = _synth_lattice(BPM, 3, true_anchor=0.5, duration_s=6.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert not fit.ok
    assert fit.reason == "insufficient evidence"


def test_djay_bpm_tolerance_boundary():
    """Fitted BPM just outside djay tolerance must gate."""
    y, beats, downbeats = _synth_lattice(BPM, 64, true_anchor=1.547, jitter_ms=0.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM + DJAY_BPM_TOLERANCE + 0.02)
    assert not fit.ok
    assert fit.reason == "fitted tempo disagrees with djay's BPM"


def test_double_time_djay_no_parity_gate():
    """djay double-time (render > fitted): plain render-bar mean, no parity gate."""
    true_anchor = 1.547
    render_bar = 4 * 60 / 180.0
    y, beats, downbeats = _synth_lattice(90.0, 64, true_anchor=true_anchor, jitter_ms=0.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=180.0)
    assert fit.ok, fit.reason
    assert fit.bpm == pytest.approx(90.0, abs=0.01)
    assert fit.render_bpm == pytest.approx(180.0)
    assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)


def test_double_time_mirrored_djay_160_fitted_80():
    """Mirrored double-time case: djay 160 BPM, true lattice 80."""
    true_anchor = 1.547
    render_bar = 4 * 60 / 160.0
    y, beats, downbeats = _synth_lattice(80.0, 64, true_anchor=true_anchor, jitter_ms=0.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=160.0)
    assert fit.ok, fit.reason
    assert fit.bpm == pytest.approx(80.0, abs=0.01)
    assert fit.render_bpm == pytest.approx(160.0)
    assert fit.anchor_s == pytest.approx(true_anchor % render_bar, abs=0.015)


def test_djay_bpm_tolerance_inclusive_boundary():
    """122.10 fitted vs djay 122.0 must pass the BPM tolerance gate."""
    y, beats, downbeats = _synth_lattice(122.10, 64, true_anchor=1.547, jitter_ms=0.0)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=122.0)
    assert fit.ok, fit.reason


def test_splice_tail_window_catches_late_shift():
    """61s track with a 0.2-beat shift at 55s must fail the splice gate."""
    y, beats, downbeats = _synth_lattice(
        BPM,
        32,
        duration_s=61.0,
        phase_step_at=55.0,
        phase_step_beats=0.2,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert not fit.ok
    assert fit.reason == "phase jumps between sections (spliced edit?)"


def test_splice_midoverlap_pairwise_range():
    """0.2-beat step at 90s on a 180s track must fail via pairwise splice range."""
    y, beats, downbeats = _synth_lattice(
        BPM,
        64,
        duration_s=180.0,
        phase_step_at=90.0,
        phase_step_beats=0.2,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert not fit.ok
    assert fit.reason == "phase jumps between sections (spliced edit?)"
    assert fit.splice_jump > SPLICE_STEP_MAX


@pytest.mark.parametrize("duration_s", [61.0, 75.0])
def test_splice_tail_window_no_false_positive(duration_s: float):
    """Clean constant-tempo tracks must stay ok with tail-flush windows."""
    y, beats, downbeats = _synth_lattice(
        BPM,
        32,
        duration_s=duration_s,
        true_anchor=1.547,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert fit.ok, fit.reason


def test_splice_fitted_period_no_djay_rounding_false_positive():
    """Splice scan on fitted period must not trip when djay rounds BPM down."""
    y, beats, downbeats = _synth_lattice(
        122.08,
        32,
        duration_s=120.0,
        jitter_ms=0.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=122.0)
    assert fit.ok, fit.reason


@pytest.mark.parametrize("duration_s", [61.0, 75.0, 120.0, 180.0])
def test_splice_no_false_positive_with_beat_this_jitter(duration_s: float):
    """Realistic beat_this jitter must not trip the splice gate on clean tracks."""
    bars = 32 if duration_s < 100.0 else 64
    y, beats, downbeats = _synth_lattice(
        BPM,
        bars,
        duration_s=duration_s,
        true_anchor=1.547,
        jitter_ms=20.0,
    )
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=BPM)
    assert fit.ok, fit.reason
    assert fit.splice_jump <= SPLICE_STEP_MAX


def test_snap_cues_clamps_before_nearest():
    """Negative round candidate must lose to the clamped non-negative lattice point."""
    fit = GridFit(
        bpm=120.0,
        render_bpm=120.0,
        anchor_s=0.9,
        beat_fit=0.01,
        bar_resid_std=0.01,
        splice_jump=0.0,
        ok=True,
        reason="",
    )
    assert snap_cues({"A": 0.1}, fit)["A"] == pytest.approx(0.4, abs=1e-9)
