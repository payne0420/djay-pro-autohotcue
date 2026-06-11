"""Unit tests for pure cue-placement policy (no audio)."""
from __future__ import annotations

import numpy as np
import pytest

from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis
from autohotcue.cuepolicy import OUTRO_BEATS_MIN, propose_cues


def mk_beat(bpm: float, bars: int, meter: int = 4) -> BeatAnalysis:
    """Regular synthetic beat grid."""
    beat_s = 60.0 / bpm
    bar_s = beat_s * meter
    duration = bars * bar_s
    beats = np.arange(0.0, duration, beat_s)
    downbeats = np.arange(0.0, duration, bar_s)
    return BeatAnalysis(
        bpm=bpm,
        beats=beats,
        downbeats=downbeats,
        duration_s=duration,
        source="test",
    )


def mk_segs(spec: list[tuple[str, float, float, float]]) -> StructureAnalysis:
    """Build segments from (label, start_bar, end_bar, energy_rank) tuples."""
    segments = [
        Segment(
            start=start,
            end=end,
            label=label,
            energy_rank=rank,
        )
        for label, start, end, rank in spec
    ]
    return StructureAnalysis(segments=segments, source="test")


def _bar_t(beat: BeatAnalysis, bar: float) -> float:
    return bar * beat.bar_s()


def test_canonical_edm_all_eight_on_downbeats():
    beat = mk_beat(120.0, 128, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 16 * bar, 0.1),
        ("verse", 16 * bar, 32 * bar, 0.4),
        ("chorus", 32 * bar, 64 * bar, 0.95),
        ("break", 64 * bar, 80 * bar, 0.15),
        ("chorus", 80 * bar, 112 * bar, 0.9),
        ("outro", 112 * bar, 128 * bar, 0.2),
    ])
    p = propose_cues(beat, structure)

    for letter in "ABDEFGH":
        assert letter in p.positions, f"missing cue {letter}: {p.notes}"
    assert "C" not in p.positions
    assert any("C (Buildup)" in n and "omitted" in n for n in p.notes)

    downbeat_set = set(beat.downbeats.tolist())
    for letter, t in p.positions.items():
        assert t in downbeat_set, f"{letter}={t} not on a downbeat"

    pos = p.positions
    assert pos["A"] <= pos["B"] < pos["D"] < pos["E"] < pos["F"] <= pos["G"] == pos["H"]
    assert pos["A"] == pytest.approx(0.0)
    assert pos["B"] == pytest.approx(16 * bar)
    assert pos["D"] == pytest.approx(32 * bar)
    assert pos["E"] == pytest.approx(64 * bar)
    assert pos["F"] == pytest.approx(80 * bar)
    assert pos["G"] == pytest.approx(112 * bar)
    assert pos["H"] == pos["G"]


def test_f_fallback_without_e():
    """When E is omitted, F is the first HIGH after D (second HIGH overall)."""
    beat = mk_beat(120.0, 128, meter=4)
    structure = mk_segs([
        ("intro", 0, 32, 0.1),
        ("verse", 32, 64, 0.4),
        ("chorus", 64, 200, 0.95),
        ("solo", 224, 256, 0.9),
        ("outro", 224, 256, 0.2),
    ])
    p = propose_cues(beat, structure)
    assert "E" not in p.positions
    assert any("Breakdown" in n and "omitted" in n for n in p.notes)
    assert "F" in p.positions
    assert p.positions["F"] == pytest.approx(224)


def test_outro_guard_regression():
    """Last segment runs to EOF; G must leave >= OUTRO_BEATS_MIN beats after it."""
    beat = mk_beat(120.0, 64, meter=4)
    bar = beat.bar_s()
    # Outro starts very late — naive policy would land ~1s before end.
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.1),
        ("chorus", 8 * bar, 48 * bar, 0.9),
        ("break", 48 * bar, 56 * bar, 0.2),
        ("outro", 56 * bar, 64 * bar, 0.15),
    ])
    p = propose_cues(beat, structure)
    assert "G" in p.positions
    beats_after = int(np.sum(beat.beats > p.positions["G"]))
    assert beats_after >= OUTRO_BEATS_MIN


def test_no_high_label_resolved_by_rank():
    beat = mk_beat(128.0, 64, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.1),
        ("verse", 8 * bar, 24 * bar, 0.3),
        ("bridge", 24 * bar, 40 * bar, 0.9),
        ("verse", 40 * bar, 56 * bar, 0.2),
        ("outro", 56 * bar, 64 * bar, 0.1),
    ])
    p = propose_cues(beat, structure)
    assert "D" in p.positions
    assert p.positions["D"] == pytest.approx(24 * bar)


def test_uniform_ranks_drop_breakdown_second_drop_omitted():
    beat = mk_beat(120.0, 64, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.5),
        ("verse", 8 * bar, 24 * bar, 0.5),
        ("bridge", 24 * bar, 40 * bar, 0.5),
        ("verse", 40 * bar, 56 * bar, 0.5),
        ("outro", 56 * bar, 64 * bar, 0.5),
    ])
    p = propose_cues(beat, structure)
    assert "D" not in p.positions
    assert any("no drop" in n.lower() for n in p.notes)
    assert "E" not in p.positions
    assert "F" not in p.positions


def test_sub_bar_segment_merged():
    beat = mk_beat(120.0, 64, meter=4)
    bar = beat.bar_s()
    # Tiny 0.5-bar blip should merge into the following chorus.
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.1),
        ("verse", 8 * bar, 31.5 * bar, 0.4),
        ("verse", 31.5 * bar, 32 * bar, 0.45),
        ("chorus", 32 * bar, 56 * bar, 0.95),
        ("outro", 56 * bar, 64 * bar, 0.2),
    ])
    p = propose_cues(beat, structure)
    assert "D" in p.positions
    assert p.positions["D"] == pytest.approx(32 * bar)


def test_intro_less_b_equals_a():
    beat = mk_beat(120.0, 64, meter=4)
    bar = beat.bar_s()
    # First segment at t=0 (no intro label).
    structure = mk_segs([
        ("verse", 0, 16 * bar, 0.4),
        ("chorus", 16 * bar, 48 * bar, 0.9),
        ("break", 48 * bar, 56 * bar, 0.2),
        ("outro", 56 * bar, 64 * bar, 0.15),
    ])
    p = propose_cues(beat, structure)
    assert p.positions["B"] == p.positions["A"]

    # Gap before first segment: B must still fall back to A, not the first boundary.
    structure_delayed = mk_segs([
        ("verse", 8 * bar, 16 * bar, 0.4),
        ("chorus", 16 * bar, 48 * bar, 0.9),
        ("outro", 48 * bar, 64 * bar, 0.15),
    ])
    p2 = propose_cues(beat, structure_delayed)
    assert p2.positions["B"] == p2.positions["A"]


def test_three_four_meter():
    beat = mk_beat(120.0, 96, meter=3)
    bar = beat.bar_s()
    assert bar == pytest.approx(3 * 60.0 / 120.0)
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.1),
        ("verse", 8 * bar, 24 * bar, 0.4),
        ("chorus", 24 * bar, 48 * bar, 0.9),
        ("break", 48 * bar, 64 * bar, 0.2),
        ("chorus", 64 * bar, 80 * bar, 0.85),
        ("outro", 80 * bar, 96 * bar, 0.15),
    ])
    p = propose_cues(beat, structure)
    for letter in "ABDEFGH":
        assert letter in p.positions
    assert "C" not in p.positions
    downbeat_set = set(beat.downbeats.tolist())
    for t in p.positions.values():
        assert t in downbeat_set


def test_violation_omits_not_clamps():
    from autohotcue.analysis import CueProposal
    from autohotcue.cuepolicy import _check_monotonicity

    violating_c = 20.0
    d_at = 20.0
    before = {
        "A": 0.0,
        "B": 10.0,
        "C": violating_c,
        "D": d_at,
        "E": 30.0,
        "F": 40.0,
        "G": 50.0,
        "H": 50.0,
    }
    p = CueProposal(positions=dict(before))
    _check_monotonicity(p)
    assert "C" not in p.positions
    for letter, t in before.items():
        if letter == "C":
            continue
        assert p.positions[letter] == t
    assert any("C" in n and "monotonicity" in n for n in p.notes)


def test_short_track_ab_only():
    beat = mk_beat(120.0, 4, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 2 * bar, 0.1),
        ("verse", 2 * bar, 4 * bar, 0.5),
    ])
    p = propose_cues(beat, structure)
    assert set(p.positions.keys()) == {"A", "B"}
    assert any("too short" in n for n in p.notes)


def test_buildup_omitted_when_snapped_start_equals_b():
    """C omitted when segment before D snaps to B (e.g. Move Extended Mix ml-allin1)."""
    beat = mk_beat(120.0, 128, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 16 * bar, 0.1),
        ("verse", 16 * bar, 32 * bar, 0.4),
        ("chorus", 32 * bar, 64 * bar, 0.95),
        ("break", 64 * bar, 80 * bar, 0.15),
        ("chorus", 80 * bar, 112 * bar, 0.9),
        ("outro", 112 * bar, 128 * bar, 0.2),
    ])
    p = propose_cues(beat, structure)
    assert "C" not in p.positions
    assert any("C (Buildup)" in n and "omitted" in n for n in p.notes)
    assert p.positions["B"] == pytest.approx(16 * bar)
    assert p.positions["D"] == pytest.approx(32 * bar)


def test_djay_bpm_crosscheck_note():
    beat = mk_beat(120.0, 64, meter=4)
    bar = beat.bar_s()
    structure = mk_segs([
        ("intro", 0, 8 * bar, 0.1),
        ("chorus", 8 * bar, 56 * bar, 0.9),
        ("outro", 56 * bar, 64 * bar, 0.2),
    ])
    p = propose_cues(beat, structure, djay_bpm=140.0)
    assert any("djay says" in n for n in p.notes)
