"""Unit tests for ml-allin1 label mapping and engine validation (no model)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autohotcue import analysis
from autohotcue._allin1 import (
    ALLIN1_LABELS,
    _AllInOneForwardWrapper,
    _energy_ranks_for_segments,
    _map_allin1_segments,
    resolve_weights_dir,
)
from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis


def test_normalize_engine_aliases():
    assert analysis.normalize_engine("ml") == ("ml", "librosa")
    assert analysis.normalize_engine("ml-librosa") == ("ml-librosa", "librosa")
    assert analysis.normalize_engine("ml-allin1") == ("ml-allin1", "allin1")
    assert analysis.normalize_engine("legacy") == ("legacy", None)


def test_normalize_engine_rejects_unknown():
    with pytest.raises(ValueError, match="unknown engine"):
        analysis.normalize_engine("ml-torch")


def test_allin1_label_mapping_verbatim():
    result = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=16.0, label="intro"),
            SimpleNamespace(start=16.0, end=48.0, label="verse"),
            SimpleNamespace(start=48.0, end=80.0, label="chorus"),
            SimpleNamespace(start=80.0, end=96.0, label="bridge"),
            SimpleNamespace(start=96.0, end=112.0, label="break"),
            SimpleNamespace(start=112.0, end=128.0, label="inst"),
            SimpleNamespace(start=128.0, end=144.0, label="solo"),
            SimpleNamespace(start=144.0, end=160.0, label="outro"),
            SimpleNamespace(start=0.0, end=1.0, label="start"),
            SimpleNamespace(start=159.0, end=160.0, label="end"),
            SimpleNamespace(start=32.0, end=40.0, label="unknown_tag"),
        ]
    )
    segs = _map_allin1_segments(result, duration_s=160.0)
    labels = [s.label for s in segs]
    assert labels == [
        "intro", "verse", "chorus", "bridge", "break", "inst", "solo", "outro", "verse",
    ]
    assert set(labels).issubset(ALLIN1_LABELS)


def test_energy_ranks_per_track():
    y = np.random.default_rng(0).standard_normal(44100 * 4).astype(np.float32)
    segs = [
        Segment(0.0, 1.0, "intro"),
        Segment(1.0, 2.0, "chorus"),
        Segment(2.0, 3.0, "break"),
    ]
    ranked = _energy_ranks_for_segments(y, 44100, segs)
    assert len(ranked) == 3
    assert all(0.0 <= s.energy_rank <= 1.0 for s in ranked)
    assert ranked[0].label == "intro"


def test_allin1_forward_wrapper():
    class _M:
        cfg = object()

        def __call__(self, x):
            return x

    w = _AllInOneForwardWrapper(_M())
    assert w(np.array([1.0]), return_embeddings=True) == 1.0


def test_structure_analysis_contract_json_roundtrip(tmp_path):
    payload = {
        "bpm": 120,
        "beats": [0.0, 0.5, 1.0],
        "downbeats": [0.0, 1.0],
        "segments": [
            {"start": 0.0, "end": 32.0, "label": "intro"},
            {"start": 32.0, "end": 64.0, "label": "chorus"},
        ],
    }
    p = tmp_path / "out.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    data = json.loads(p.read_text(encoding="utf-8"))
    beat = BeatAnalysis(
        bpm=float(data["bpm"]),
        beats=np.asarray(data["beats"], dtype=float),
        downbeats=np.asarray(data["downbeats"], dtype=float),
        duration_s=64.0,
        source="test",
    )
    segments = [
        Segment(float(s["start"]), float(s["end"]), s["label"]) for s in data["segments"]
    ]
    structure = StructureAnalysis(segments=segments, source="allin1")
    assert structure.source == "allin1"
    assert beat.bpm == 120.0
    assert len(structure.segments) == 2


@pytest.mark.skipif(
    not (Path("all-in-one-mlx/mlx-weights/harmonix-fold0_mlx.npz").is_file()),
    reason="mlx weights not present (clone all-in-one-mlx with mlx-weights/)",
)
def test_resolve_weights_dir_finds_clone():
    d = resolve_weights_dir()
    assert (d / "harmonix-fold0_mlx.npz").is_file()


def test_bench_parse_engines_includes_allin1():
    from autohotcue.bench import parse_engines

    assert parse_engines("ml,ml-allin1,legacy") == ["ml", "ml-allin1", "legacy"]


def test_cli_engine_choices():
    from autohotcue.cli import _ENGINE_CHOICES

    assert "ml-allin1" in _ENGINE_CHOICES
    assert "ml" in _ENGINE_CHOICES
