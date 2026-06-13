"""Unit tests for kick-band bass events and ml-bass cue placement."""
from __future__ import annotations

import numpy as np
import pytest

from autohotcue import analysis, cli
from autohotcue.backends import BeatAnalysis
from autohotcue.bassline import _apply_d_a_backstop, propose_cues_bass
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
    fade_bars: tuple[int, ...] | range = (),
    fade_amp: float | None = None,
    kick_from_bar: int = 0,
    kick_to_bar: int | None = None,
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

    kick_bar_set = set(range(kick_from_bar, kick_to_bar if kick_to_bar is not None else bars))
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

    fade_set = set(fade_bars)
    fade_peak = bass_amp if fade_amp is None else fade_amp
    for bi in fade_set:
        b0 = true_anchor + bi * bar_s
        b1 = b0 + bar_s
        mask = (t >= b0) & (t < b1)
        ramp = (t[mask] - b0) / bar_s
        y[mask] += (ramp * fade_peak * np.sin(2 * np.pi * 55.0 * t[mask])).astype(np.float32)

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

    # Only 4 bars remain after G: no fully audible 8-bar loop fits, so H is omitted.
    assert "H" not in prop.positions
    assert any("short tail; using legacy placement" in n for n in prop.notes)
    assert any("H (Loop Out): omitted (no audible loop after G)" in n for n in prop.notes)

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
    assert any("no qualifying drop" in n for n in prop.notes)
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


def test_d_a_backstop_clears_e_f():
    d, e, f, fired = _apply_d_a_backstop(32, 64, 80, 32)
    assert fired
    assert d is None and e is None and f is None

    d, e, f, fired = _apply_d_a_backstop(32, 64, 80, 0)
    assert not fired
    assert (d, e, f) == (32, 64, 80)


def test_d_a_backstop_integration():
    """Late first bass-in at A's bar triggers D==A backstop; E/F must not emit."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        24,
        true_anchor=anchor,
        kick_from_bar=5,
        bass_bars=range(5, 24),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    assert "D" not in prop.positions
    assert "E" not in prop.positions
    assert "F" not in prop.positions
    assert any("coincides with A" in n for n in prop.notes)
    assert any("C (Buildup): omitted (no D)" in n for n in prop.notes)
    assert any("E (Breakdown): omitted (no D)" in n for n in prop.notes)
    assert any("F (2nd Drop): omitted (no E)" in n for n in prop.notes)
    assert not any("no qualifying drop" in n for n in prop.notes)


def test_gate_refused_groove_start_c():
    """Gate-refused: pre-drop OFF-run C at raw bar, same as lattice-locked path."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        bass_bars=tuple(
            list(range(16, 29)) + list(range(64, 120)) + list(range(136, 152))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = GridFit(
        bpm=bpm,
        render_bpm=bpm,
        anchor_s=anchor,
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
    _assert_near_bar(prop.positions["C"], bpm, 29, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 64, anchor)


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


def test_ml_bass_is_default_engine():
    import argparse
    import inspect

    assert inspect.signature(analysis.analyze).parameters["engine"].default == "ml-bass"

    p = argparse.ArgumentParser()
    cli._add_engine_arg(p)
    assert p.parse_args([]).engine == "ml-bass"


def test_outro_snap_fallback():
    """Audible tail pulls G/H back from the trailing bass-off run near file end."""
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

    assert "D" not in prop.positions
    assert any("no qualifying drop" in n for n in prop.notes)
    _assert_near_bar(prop.positions["G"], bpm, 72, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 88, anchor)
    _assert_ordering(prop.positions)


def test_fade_in_does_not_shift_a():
    """Fade-in bar crossing A_MIN_FRAC must not pull A one bar early."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        kick_from_bar=8,
        fade_bars=(7,),
        fade_amp=0.22,
        bass_bars=range(40, 64),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["A"], bpm, 8, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 8, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 40, anchor)
    assert any("off8=+0" in n for n in prop.notes)


def test_flat_intro_a_unchanged():
    """Flat kick from bar 0 still places A at bar 0 (steady ratio passes trivially)."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        kick_from_bar=0,
        bass_bars=range(32, 94),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["A"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 0, anchor)


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


def test_groove_start_not_drop():
    """Groove-start bass-in at bar 16 is not D; real drop is the return at bar 64."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        bass_bars=tuple(
            list(range(16, 29)) + list(range(64, 120)) + list(range(136, 152))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["A"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["C"], bpm, 29, anchor)
    _assert_near_bar(prop.positions["D"], bpm, 64, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 120, anchor)
    _assert_near_bar(prop.positions["F"], bpm, 136, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 152, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 152, anchor)
    assert any("short tail; using legacy placement" in n for n in prop.notes)
    assert any("C: raw bar 29" in n and "off-phrase (unsnapped)" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def test_no_qualifying_drop_omits():
    """Single early bass-in with no return is groove-only: D/C/E/F omitted."""
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

    _assert_near_bar(prop.positions["A"], bpm, 0, anchor)
    _assert_near_bar(prop.positions["B"], bpm, 0, anchor)
    assert "D" not in prop.positions
    assert "C" not in prop.positions
    assert "E" not in prop.positions
    assert "F" not in prop.positions
    assert any("no qualifying drop" in n for n in prop.notes)
    _assert_near_bar(prop.positions["G"], bpm, 72, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 88, anchor)


def test_outro_pulls_back_from_silent_tail():
    """Silent bars after audible content pull G/H back from the file end."""
    from autohotcue.bassline import LOOP_OUT_BARS, OUTRO_TAIL_BARS

    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        kick_to_bar=150,
        bass_bars=tuple(
            list(range(16, 29))
            + list(range(64, 96))
            + list(range(104, 121))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, bass = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    last_audible = 149
    bar_s = 4 * 60 / bpm
    g_bar = round((prop.positions["G"] - fit.anchor_s) / bar_s)
    h_bar = round((prop.positions["H"] - fit.anchor_s) / bar_s)

    _assert_near_bar(prop.positions["D"], bpm, 64, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 96, anchor)
    _assert_near_bar(prop.positions["F"], bpm, 104, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 120, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 136, anchor)

    assert last_audible - g_bar >= OUTRO_TAIL_BARS
    assert g_bar > round((prop.positions["F"] - fit.anchor_s) / bar_s)
    assert g_bar < h_bar <= last_audible
    assert h_bar + LOOP_OUT_BARS <= last_audible + 1
    assert len(bass.full_rms) > last_audible


def test_loop_out_distinct_when_audible_to_end():
    """Ample audible tail: G takes the preferred 16-bar-tail path, H a later loop point."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        128,
        true_anchor=anchor,
        bass_bars=tuple(list(range(24, 48)) + list(range(56, 72))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["G"], bpm, 104, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 120, anchor)
    assert prop.positions["G"] < prop.positions["H"]
    assert not any("short tail" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def test_nudge_beats():
    import argparse
    from dataclasses import replace

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
    assert fit.ok

    prop0, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)
    beat_period = 60.0 / fit.bpm
    nudged = replace(
        fit,
        anchor_s=(fit.anchor_s + beat_period) % (4 * 60.0 / fit.render_bpm),
    )
    prop1, _ = propose_cues_bass(y, SR, beat, nudged, djay_bpm=bpm)

    for letter in prop0.positions:
        assert prop1.positions[letter] == pytest.approx(
            prop0.positions[letter] + beat_period,
            abs=1e-6,
        )

    for cmd in ("propose", "viz", "apply"):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        sp = sub.add_parser(cmd)
        if cmd in ("propose", "apply"):
            sp.add_argument("-j", "--jobs", type=int, default=1)
        cli._add_engine_arg(sp)
        cli._add_nudge_arg(sp)
        args = p.parse_args([cmd, "--nudge-beats", "1.0"])
        assert args.nudge_beats == 1.0


def test_outro_snap_collision_falls_back():
    """Snapped G colliding with an earlier cue falls back to the raw bar.

    Under current constants this is unreachable end-to-end (every G candidate
    sits >= RETURN_MIN_ON_BARS past F while snapping moves <= SNAP_MAX_BARS),
    so the defensive guard is pinned at the helper level: raw 82 snaps to 80,
    which collides with earlier_max=80, and the raw bar must win.
    """
    from autohotcue.bassline import _finalize_g_bar

    g_bar, note = _finalize_g_bar(
        82, 0, True, last_audible=95, earlier_max=80, preferred_path=False,
    )
    assert g_bar == 82
    assert "raw bar 82" in note
    assert "snapped to bar 80" not in note

    # Companion end-to-end check: when the snap target is valid, it wins.
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        96,
        true_anchor=anchor,
        bass_bars=tuple(list(range(32, 64)) + list(range(72, 82))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["F"], bpm, 72, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 80, anchor)
    assert any("short tail; using legacy placement" in n for n in prop.notes)
    assert any("G: raw bar 82" in n and "snapped to bar 80" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def _clamp_track(kick_to_bar: int):
    """128 bars, D@32 E@96 F@104, trailing bass-out at 115, kick to kick_to_bar."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        128,
        true_anchor=anchor,
        kick_to_bar=kick_to_bar,
        bass_bars=tuple(list(range(32, 96)) + list(range(104, 115))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    return bpm, anchor, propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)


def test_h_clamp_to_g_when_loop_audible():
    """Natural H (112) < G (115) and [115, 123) fully audible — H clamps to G."""
    bpm, anchor, (prop, _) = _clamp_track(kick_to_bar=126)

    _assert_near_bar(prop.positions["F"], bpm, 104, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 115, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 115, anchor)
    assert any("H (Loop Out): clamped to G" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def test_outro_skips_mid_track_bass_out():
    """A bass-out with a full section after it is a breakdown, never G."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        192,
        true_anchor=anchor,
        bass_bars=tuple(list(range(40, 80)) + list(range(88, 120)) + list(range(128, 184))),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 40, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 80, anchor)
    _assert_near_bar(prop.positions["F"], bpm, 88, anchor)
    # The mid-track bass-out at bar 120 must not win; the final bass-out (184)
    # leaves too little audible tail, so G is the last grid mix-out boundary.
    _assert_near_bar(prop.positions["G"], bpm, 168, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 184, anchor)
    assert prop.positions["G"] < prop.positions["H"]
    _assert_ordering(prop.positions)


def test_outro_when_breakdown_runs_to_end():
    """E's bass-out lasting to the end must still yield a later grid G (not omit G/H)."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        bass_bars=range(40, 144),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["D"], bpm, 40, anchor)
    _assert_near_bar(prop.positions["E"], bpm, 144, anchor)
    assert "F" not in prop.positions
    _assert_near_bar(prop.positions["G"], bpm, 152, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 152, anchor)
    assert any("short tail; using legacy placement" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def test_h_clamp_omitted_when_loop_after_g_inaudible():
    """Natural H < G and [G, G+8) straddles the silent tail — omit H."""
    bpm, anchor, (prop, _) = _clamp_track(kick_to_bar=120)

    _assert_near_bar(prop.positions["F"], bpm, 104, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 115, anchor)
    assert "H" not in prop.positions
    assert any(
        "H (Loop Out): omitted (no audible loop after G)" in n for n in prop.notes
    )


def test_monotonicity_allows_g_without_h():
    """_check_monotonicity must not resurrect H when G survives without it."""
    from autohotcue.analysis import CueProposal
    from autohotcue.cuepolicy import _check_monotonicity

    p = CueProposal()
    p.positions["A"] = 0.0
    p.positions["B"] = 0.0
    p.positions["G"] = 100.0
    p.notes.append("H (Loop Out): omitted (no audible loop after G)")
    _check_monotonicity(p)
    assert "G" in p.positions
    assert "H" not in p.positions
    assert not any("H:" in n and "monotonicity" in n for n in p.notes)


def test_loop_out_skips_silent_hole():
    """Silent bars inside the last phrase-aligned loop window push H earlier."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        kick_to_bar=140,
        bass_bars=tuple(
            list(range(16, 29))
            + list(range(64, 120))
            + list(range(136, 140))
            + list(range(144, 151))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    _assert_near_bar(prop.positions["G"], bpm, 128, anchor)
    _assert_near_bar(prop.positions["H"], bpm, 128, anchor)
    bar_s = 4 * 60 / bpm
    h_bar = round((prop.positions["H"] - fit.anchor_s) / bar_s)
    for bi in range(h_bar, h_bar + 8):
        b0 = anchor + bi * bar_s
        seg = y[int(b0 * SR): int((b0 + bar_s) * SR)]
        assert float(np.sqrt(np.mean(seg * seg))) > 0.01


def test_outro_after_f_strict():
    """Ample audible tail after F: G uses preferred path and sits strictly after F."""
    bpm = 124.0
    anchor = 1.5
    y, beats, downbeats = _synth_bass_track(
        bpm,
        160,
        true_anchor=anchor,
        bass_bars=tuple(
            list(range(16, 29))
            + list(range(64, 96))
            + list(range(104, 121))
        ),
    )
    beat = _beat_analysis(beats, downbeats, bpm, len(y) / SR)
    fit = fit_grid(y, SR, beats, downbeats, djay_bpm=bpm)
    prop, _ = propose_cues_bass(y, SR, beat, fit, djay_bpm=bpm)

    bar_s = 4 * 60 / bpm
    f_bar = round((prop.positions["F"] - fit.anchor_s) / bar_s)
    g_bar = round((prop.positions["G"] - fit.anchor_s) / bar_s)

    _assert_near_bar(prop.positions["F"], bpm, 104, anchor)
    _assert_near_bar(prop.positions["G"], bpm, 120, anchor)
    assert g_bar > f_bar
    assert prop.positions["G"] > prop.positions["F"]
    assert not any("short tail; using legacy placement" in n for n in prop.notes)
    _assert_ordering(prop.positions)


def test_analyze_nudge_anchor(monkeypatch, tmp_path):
    """analysis.analyze nudge shifts grid_fit.anchor_s; cli write path uses it."""
    import argparse
    from unittest.mock import MagicMock

    from autohotcue import djaydb

    bpm = 124.0
    anchor = 1.5
    duration_s = 120.0
    y = np.zeros(int(duration_s * SR), dtype=np.float32)
    beats = np.arange(anchor, duration_s, 60.0 / bpm)
    downbeats = np.arange(anchor, duration_s, 4 * 60.0 / bpm)
    beat = BeatAnalysis(
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        duration_s=duration_s,
        source="test",
    )
    fit = GridFit(
        bpm=bpm,
        render_bpm=bpm,
        anchor_s=anchor,
        beat_fit=0.01,
        bar_resid_std=0.01,
        splice_jump=0.1,
        ok=True,
        reason="ok",
    )

    monkeypatch.setattr(analysis, "decode", lambda path, sr=SR: y)
    monkeypatch.setattr("autohotcue.backends.track_beats", lambda y, device=None: beat)
    monkeypatch.setattr("autohotcue.gridlock.fit_grid", lambda *a, **k: fit)

    path = tmp_path / "track.wav"
    path.touch()

    bar_period = 4 * 60.0 / fit.render_bpm
    beat_period = 60.0 / fit.bpm

    track0, _ = analysis.analyze(str(path), engine="ml-bass", nudge_beats=0)
    assert track0.grid_fit is not None
    assert track0.grid_fit.anchor_s == pytest.approx(anchor)

    track_neg, _ = analysis.analyze(str(path), engine="ml-bass", nudge_beats=-1)
    assert track_neg.grid_fit.anchor_s == pytest.approx(
        (anchor - beat_period) % bar_period,
    )

    track_half, _ = analysis.analyze(str(path), engine="ml-bass", nudge_beats=0.5)
    assert track_half.grid_fit.anchor_s == pytest.approx(
        (anchor + 0.5 * beat_period) % bar_period,
    )

    assert djaydb.build_beat_grid_edits(track0.grid_fit.anchor_s) != (
        djaydb.build_beat_grid_edits(track_neg.grid_fit.anchor_s)
    )

    existing = MagicMock()
    existing.root.get.return_value = None
    ensure_backup = MagicMock()
    monkeypatch.setattr(cli, "_djay_running", lambda: False)
    track_half, prop = analysis.analyze(str(path), engine="ml-bass", nudge_beats=0.5)
    prop.positions.setdefault("A", anchor)
    cli._write_one(
        MagicMock(),
        "key",
        existing,
        track_half,
        prop,
        ensure_backup,
        grid_lock=True,
        force=True,
    )
    grid_call = next(
        c for c in existing.root.set.call_args_list if c[0][0] == "beatGridEdits"
    )
    written = grid_call[0][1]
    expected = djaydb.build_beat_grid_edits(track_half.grid_fit.anchor_s)
    assert written.fields[0][1].value == pytest.approx(expected.fields[0][1].value)

    for cmd in ("propose", "viz", "apply"):
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
        sp = sub.add_parser(cmd)
        if cmd in ("propose", "apply"):
            sp.add_argument("-j", "--jobs", type=int, default=1)
        cli._add_engine_arg(sp)
        cli._add_nudge_arg(sp)
        args = p.parse_args([cmd, "--nudge-beats", "1.0"])
        assert args.nudge_beats == 1.0


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
        if letter == "F" and "G" in pos:
            assert cur < pos["G"]
        prev = cur
    if "G" in pos and "H" in pos:
        assert pos["G"] <= pos["H"]
