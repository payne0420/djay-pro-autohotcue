"""Unit tests for ml-allin1 label mapping and engine validation (no model)."""
from __future__ import annotations

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
from autohotcue.backends import Segment


def test_normalize_engine_aliases():
    assert analysis.normalize_engine("ml") == ("ml", "librosa")
    assert analysis.normalize_engine("ml-librosa") == ("ml-librosa", "librosa")
    assert analysis.normalize_engine("ml-allin1") == ("ml-allin1", "allin1")
    assert analysis.normalize_engine("legacy") == ("legacy", None)


def test_normalize_engine_rejects_unknown():
    with pytest.raises(ValueError, match="unknown engine"):
        analysis.normalize_engine("ml-torch")


def test_effective_parallel_jobs_forces_single_worker_for_allin1():
    assert analysis.effective_parallel_jobs("ml-allin1", 8) == 1
    assert analysis.effective_parallel_jobs("ml", 8) == 8
    assert analysis.effective_parallel_jobs("legacy", 4) == 4


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
        ]
    )
    segs = _map_allin1_segments(result, duration_s=160.0)
    labels = [s.label for s in segs]
    assert labels == [
        "intro", "verse", "chorus", "bridge", "break", "inst", "solo", "outro",
    ]
    assert set(labels).issubset(ALLIN1_LABELS)


def test_allin1_unknown_label_raises():
    result = SimpleNamespace(
        segments=[SimpleNamespace(start=0.0, end=8.0, label="unknown_tag")]
    )
    with pytest.raises(ValueError, match="unknown all-in-one segment label"):
        _map_allin1_segments(result, duration_s=160.0)


def test_allin1_non_finite_boundary_raises():
    result = SimpleNamespace(
        segments=[SimpleNamespace(start=float("nan"), end=8.0, label="verse")]
    )
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_allin1_segments(result, duration_s=160.0)


def test_allin1_skips_degenerate_segments():
    result = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=0.0, label="intro"),
            SimpleNamespace(start=8.0, end=4.0, label="verse"),
            SimpleNamespace(start=16.0, end=32.0, label="chorus"),
        ]
    )
    segs = _map_allin1_segments(result, duration_s=160.0)
    assert [s.label for s in segs] == ["chorus"]


def test_allin1_clamps_to_track_duration():
    result = SimpleNamespace(
        segments=[SimpleNamespace(start=150.0, end=200.0, label="outro")]
    )
    segs = _map_allin1_segments(result, duration_s=160.0)
    assert len(segs) == 1
    assert segs[0].start == 150.0
    assert segs[0].end == 160.0


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
