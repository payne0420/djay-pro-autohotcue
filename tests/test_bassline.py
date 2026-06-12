"""Unit tests for kick-band bass events and ml-bass cue placement."""
from __future__ import annotations

import numpy as np
import pytest

from autohotcue import analysis, cli
from autohotcue.backends import BeatAnalysis
from autohotcue.bassline import propose_cues_bass
from autohotcue.gridlock import GridFit, fit_grid

SR = 44100


def _synth_bass_track(
    bpm: float,
    bars: int,
    *,
    true_anchor: float = 1.5,
    bass_bars: tuple[int, ...] | range = (),
    kick: float = 0.9,
    bass_amp: float = 0.4,
    pad_bars: tuple[int, ...] | range = (),
    kick_from_bar: int = 0,
    jitter_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build mono audio + beat/downbeat arrays on a straight lattice."""
    beat_s = 60.0 / bpm
    bar_s = 4 * beat_s
    duration_s = bars * bar_s + true_anchor + 2.0
    n = int(duration_s * SR)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float64) / SR

    beats = np.arange(true_anchor, duration_s, beat_s)
    if jitter_ms > 0:
        rng = np.random.default_rng(42)
        beats = beats + rng.uniform(-jitter_ms, jitter_ms, len(beats)) / 1000.0
    downbeats = np.arange(true_anchor, duration_s, bar_s)

    kick_bar_set = set(range(kick_from_bar, bars))
    kick_len_s = 0.1
    for bi in kick_bar_set:
        for beat_i in range(4):
            bt = true_anchor + bi * bar_s + beat_i * beat_s
            i0 = int(bt * SR)
            i1 = min(n, i0 + int(kick_len_s * SR))
            tau = (np.arange(i0, i1, dtype=np.float64) - i0) / SR
            burst = (
                kick
                * np.sin(2 * np.pi * 60.0 * tau)
                * np.exp(-tau / 0.05)
            ).astype(np.float32)
            y[i0:i1] += burst

    bass_set = set(bass_bars)
    for bi in bass_set:
        b0 = true_anchor + bi * bar_s
        b1 = b0 + bar_s
        mask = (t >= b0) & (t < b1)
        y[mask] += (bass_amp * np.sin(2 * np.pi * 55.0 * t[mask])).astype(np.float32)

    pad_set = set(pad_bars)
    for bi in pad_set:
        b0 = true_anchor + bi * bar_s
        b1 = b0 + bar_s
        mask = (t >= b0) & (t < b1)
        pad = 0.3 * (np.sin(2 * np.pi * 800.0 * t[mask]) + np.sin(2 * np.pi * 1200.0 * t[mask]))
        y[mask] += pad.astype(np.float32)

    return y, beats, downbeats


def _beat_analysis(
    beats: np.ndarray,
    downbeats: np.ndarray,
    bpm: float,
    duration_s: float,
) -> BeatAnalysis:
    return BeatAnalysis(
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        duration_s=duration_s,
        source="test",
    )


def _bar_time(bpm: float, bar: int, anchor: float = 1.5) -> float:
    bar_s = 4 * 60.0 / bpm
    return anchor + bar * bar_s


def _assert_near_bar(t: float, bpm: float, bar: int, anchor: float = 1.5) -> None:
    bar_s = 4 * 60.0 / bpm
    assert t == pytest.approx(_bar_time(bpm, bar, anchor), abs=bar_s * 0.5)


def _assert_on_lattice(t: float, anchor: float, bar_period: float) -> None:
    k = (t - anchor) / bar_period
    assert abs(k - round(k)) < 1e-6


def test_full_layout():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=tuple(list(range(32, 64)) + list(range(80, 92))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    assert fit.ok, fit.reason

    prop, bass = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)
    bar_period = 4 * 60 / fit.render_bpm

    _assert_near_bar(prop.positions["A"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["C"], bpm, 24, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 32, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 64, anchor)
    _assert_near_bar(prop.positions["F"], bpm, 80, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 92, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 92, anchor)

    assert bass.snapped is True
    for t in prop.positions.values():
        _assert_on_lattice(t, fit.anchor_s, bar_period)

    assert any("off8=+0" in n for n in prop.notes)


def test_snap_governed_by_off8():
    """off8 (+1) governs snap to bar 40; off16 (-7) does not pull toward bar 48."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        kick_from_bar=0,
        bass_bars=range(41, 80),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 40, anchor)
    assert any(
        "D: raw bar 41, off8=+1, off16=-7," in n and "snapped to bar 40" in n
        for n in prop.notes
    )


def test_snap_plus_one():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=range(33, 64),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 32, anchor)
    assert (
        "D: raw bar 33, off8=+1, off16=+1, off32=+1, snapped to bar 32" in prop.notes
    )


def test_off_phrase_flag():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=range(37, 64),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 37, anchor)
    assert any("off-phrase" in n for n in prop.notes)


def test_no_transition():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=range(0, 96),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    assert "D" not in prop.positions
    assert "E" not in prop.positions
    assert "F" not in prop.positions
    assert "A" in prop.positions and "B" in prop.positions
    assert any("no bass-in transition" in n for n in prop.notes)
    assert "G" in prop.positions
    _assert_ordering(prop.positions)


def test_hysteresis():
    bpm = 124.0
    anchor = 1.5
    bass_bars = list(range(40, 80))
    bass_bars.remove(55)  # single-bar dip
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=tuple(bass_bars),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    assert "E" in prop.positions
    _assert_near_bar(prop.positions["E"], bpm, 80, anchor)


def test_gate_refused_fallback():
    bpm = 124.0
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        bass_bars=range(32, 64),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = GridFit(
        bpm=bpm,
        render_bpm=bpm,
        anchor_s=0.0,
        beat_fit=0.01,
        bar_resid_std=0.01,
        splice_jump=0.2,
        ok=False,
        reason="phase jumps between sections (spliced edit?)",
    )
    prop, bass = propose_cues_bass(y, SR, beat, fit)
    assert bass.snapped is False
    assert any("phrase snapping disabled" in n for n in prop.notes)
    assert not any("off8=" in n for n in prop.notes)
    assert len(prop.positions) >= 2
    _assert_near_bar(prop.positions["D"], bpm, 32)


def test_pad_intro():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        pad_bars=range(0, 8),
        kick_from_bar=8,
        bass_bars=range(40, 64),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["A"], bpm, 8, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 40, anchor)
    assert any("off8=+0" in n for n in prop.notes)


def test_breakdown_skips_short_dip():
    """4-bar dropout at bar 64 is skipped; 8-bar dropout at 76 becomes E."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=tuple(
            list(range(32, 64)) + list(range(68, 76)) + list(range(84, 94))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 32, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 76, anchor)
    _assert_near_bar(prop.positions["F"], bpm, 84, anchor)
    assert "E" not in prop.positions or prop.positions["E"] != _bar_time(bpm, 64, anchor)


def test_short_track():
    bpm = 124.0
    y, beats, downbeats = _synth_bass_track(bpm, 12, bass_bars=range(0, 12))
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit)

    assert set(prop.positions.keys()) == {"A", "B"}
    assert any("track too short for bass structure analysis" in n for n in prop.notes)


def test_wiring():
    assert analysis.normalize_engine("ml-bass") == ("ml-bass", None)
    assert "ml-bass" in analysis.VALID_ENGINES
    assert analysis.effective_parallel_jobs("ml-bass", 4) == 4
    assert "ml-bass" in cli._ENGINE_CHOICES
    assert cli._is_ml_engine("ml-bass")


def test_outro_snap_fallback():
    """Trailing off-run at bar 94 must not snap to bar 96 (out of range)."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=range(8, 94),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 8, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 94, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 94, anchor)


def test_bass_majority_track():
    """Bass from bar 32 with kick from bar 0: A/B at 0, D at 32."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=range(32, 94),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["A"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 32, anchor)


def test_ordering():
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=tuple(list(range(32, 64)) + list(range(80, 92))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)
    _assert_ordering(prop.positions)


def _assert_ordering(pos: dict[str, float]) -> None:
    order = ["A", "B", "C", "D", "E", "F", "G", "H"]
    present = [k for k in order if k in pos]
    prev = None
    for letter in present:
        cur = pos[letter]
        if prev is not None:
            assert cur >= prev, f"{letter}={cur} before previous {prev}"
        if letter == "C" and "D" in pos:
            assert cur < pos["D"]
        if letter == "D" and "E" in pos:
            assert cur < pos["E"]
        if letter == "E" and "F" in pos:
            assert cur < pos["F"]
        prev = cur
    if "G" in pos and "H" in pos:
        assert pos["G"] == pos["H"]
