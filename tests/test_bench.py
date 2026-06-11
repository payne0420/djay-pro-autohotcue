"""Tests for bench harness truth parsing and metric math."""
from __future__ import annotations

import json

import pytest

from autohotcue import analysis
from autohotcue.bench import (
    EngineResult,
    GroundTruthTrack,
    SlotMetrics,
    evaluate_proposal,
    load_ground_truth,
    yardstick_bpm,
)


def test_load_ground_truth(tmp_path):
    data = {
        "tracks": [
            {"path": "/a.opus", "cues": {"A": 0.5, "D": 92.31}},
            {"path": "/b.opus", "cues": {"G": 180.0}},
        ]
    }
    p = tmp_path / "truth.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    tracks = load_ground_truth(str(p))
    assert len(tracks) == 2
    assert tracks[0].path == "/a.opus"
    assert tracks[0].cues == {"A": 0.5, "D": 92.31}


def test_slot_metrics_beat_and_bar_hits():
    m = SlotMetrics()
    bpm = 120.0
    beat_s = 60.0 / bpm
    bar_s = 4.0 * beat_s
    m.record(10.0, 10.0, beat_s, bar_s)
    m.record(10.0, 10.5, beat_s, bar_s)  # 0.5s off — within 1 bar (2.0s), not 1 beat
    m.record(20.0, 22.5, beat_s, bar_s)  # 2.5s off — outside 1 bar
    assert m.n == 3
    assert m.hits_beat == 2
    assert m.hits_bar == 2
    assert m.mae == pytest.approx((0.0 + 0.5 + 2.5) / 3)


def test_evaluate_proposal_skips_missing_slots():
    truth = GroundTruthTrack(path="/x.opus", cues={"A": 1.0, "D": 50.0})
    prop = analysis.CueProposal(positions={"A": 1.0})
    result = EngineResult(engine="ml")
    evaluate_proposal(prop, truth, bpm=120.0, result=result)
    assert result.slots["A"].n == 1
    assert "D" not in result.slots


def test_yardstick_bpm_prefers_djay():
    track = analysis.TrackAnalysis(
        bpm=126.0,
        first_beat_s=0.0,
        duration_s=200.0,
        engine="ml",
        djay_bpm=124.0,
    )
    assert yardstick_bpm(track, fallback=130.0) == 124.0


def test_yardstick_bpm_falls_back_to_tracked():
    track = analysis.TrackAnalysis(
        bpm=126.0,
        first_beat_s=0.0,
        duration_s=200.0,
        engine="ml",
        djay_bpm=None,
    )
    assert yardstick_bpm(track, fallback=None) == 126.0
