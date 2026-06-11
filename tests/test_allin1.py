"""Unit tests for ml-allin1 label mapping and engine validation (no model)."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autohotcue import analysis
from autohotcue._allin1 import (
    ALLIN1_LABELS,
    _AllInOneForwardWrapper,
    _WEIGHTS_MARKER,
    _energy_ranks_for_segments,
    _map_allin1_segments,
    resolve_weights_dir,
    segment_structure_allin1,
)
from autohotcue.backends import BeatAnalysis, Segment


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
        "intro",
        "verse",
        "chorus",
        "bridge",
        "break",
        "inst",
        "solo",
        "outro",
        "intro",
        "outro",
    ]
    assert set(labels).issubset(ALLIN1_LABELS)


def test_allin1_start_end_map_to_edge_labels():
    result = SimpleNamespace(
        segments=[
            SimpleNamespace(start=0.0, end=24.0, label="start"),
            SimpleNamespace(start=24.0, end=96.0, label="verse"),
            SimpleNamespace(start=96.0, end=120.0, label="end"),
        ]
    )
    segs = _map_allin1_segments(result, duration_s=120.0)
    assert [s.label for s in segs] == ["intro", "verse", "outro"]
    assert segs[0].start == 0.0
    assert segs[-1].end == 120.0


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


def test_energy_ranks_partial_tail_slice(monkeypatch):
    """Segment end past low-band length uses clamped partial slice, not 0.0."""
    from autohotcue.analysis import HOP

    sr = 44100
    n_frames = 10
    low = np.linspace(0.1, 1.0, n_frames, dtype=np.float64)
    monkeypatch.setattr(
        "autohotcue._allin1.band_energy",
        lambda _y, _sr, _hop: (low, None, None, None),
    )
    segs = [
        Segment(0.0, 0.05, "intro"),
        Segment(0.1, 5.0, "outro"),
    ]
    ranked = _energy_ranks_for_segments(np.zeros(sr), sr, segs)
    assert len(ranked) == 2
    i0 = int(0.1 * sr / HOP)
    i1 = max(i0 + 1, int(5.0 * sr / HOP))
    tail_mean = float(low[i0 : min(i1, len(low))].mean())
    assert tail_mean == pytest.approx(0.95)
    assert ranked[1].energy_rank > ranked[0].energy_rank
    assert ranked[1].energy_rank == pytest.approx(1.0)


def test_allin1_forward_wrapper():
    class _M:
        cfg = object()

        def __call__(self, x):
            return x

    w = _AllInOneForwardWrapper(_M())
    assert w(np.array([1.0]), return_embeddings=True) == 1.0


def test_resolve_weights_dir_env_overrides_stale_cache(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    cache = home / ".cache/autohotcue/mlx-weights"
    cache.mkdir(parents=True)
    (cache / _WEIGHTS_MARKER).write_bytes(b"cached")

    experimental = tmp_path / "experimental"
    experimental.mkdir()
    (experimental / _WEIGHTS_MARKER).write_bytes(b"new")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ALLIN1_MLX_WEIGHTS_DIR", str(experimental))

    assert resolve_weights_dir() == experimental.resolve()


@pytest.mark.skipif(
    not (Path("all-in-one-mlx/mlx-weights/harmonix-fold0_mlx.npz").is_file()),
    reason="mlx weights not present (clone all-in-one-mlx with mlx-weights/)",
)
def test_resolve_weights_dir_finds_clone():
    d = resolve_weights_dir()
    assert (d / _WEIGHTS_MARKER).is_file()


def test_segment_structure_allin1_plumbing(monkeypatch, tmp_path):
    """Mock demucs/inference path: return_mx, sample_rate, edge labels, errors."""
    import autohotcue._allin1 as mod

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / _WEIGHTS_MARKER).write_bytes(b"x")
    monkeypatch.setattr(mod, "resolve_weights_dir", lambda: weights)
    monkeypatch.setattr(mod, "_check_platform", lambda: None)
    monkeypatch.setattr(mod, "_ensure_mlx_env", lambda: None)

    sep_calls: dict[str, object] = {}

    class FakeSeparator:
        samplerate = 44100

        def separate_audio_file(self, path, return_mx=False):
            sep_calls["path"] = path
            sep_calls["return_mx"] = return_mx
            return None, "stems_mx"

    model = SimpleNamespace(cfg=SimpleNamespace(sample_rate=44100))
    monkeypatch.setattr(mod, "_get_separator", lambda: FakeSeparator())
    monkeypatch.setattr(mod, "_get_model", lambda _wd: model)

    spec_calls: dict[str, object] = {}

    def fake_spectrogram_from_stems(stems, sample_rate, backend, return_mx):
        spec_calls["stems"] = stems
        spec_calls["sample_rate"] = sample_rate
        spec_calls["backend"] = backend
        spec_calls["return_mx"] = return_mx
        return "spec"

    inf_calls: dict[str, object] = {}

    def fake_run_inference_mlx_spec(**kwargs):
        inf_calls.update(kwargs)
        return SimpleNamespace(
            segments=[
                SimpleNamespace(start=0.0, end=16.0, label="start"),
                SimpleNamespace(start=16.0, end=48.0, label="verse"),
                SimpleNamespace(start=48.0, end=64.0, label="end"),
            ]
        )

    helpers = types.ModuleType("allin1_mlx.helpers")
    helpers.run_inference_mlx_spec = fake_run_inference_mlx_spec
    spec_mod = types.ModuleType("allin1_mlx.spectrogram")
    spec_mod.spectrogram_from_stems = fake_spectrogram_from_stems
    monkeypatch.setitem(sys.modules, "allin1_mlx.helpers", helpers)
    monkeypatch.setitem(sys.modules, "allin1_mlx.spectrogram", spec_mod)

    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"wav")
    y = np.zeros(44100 * 2, dtype=np.float32)
    beat = BeatAnalysis(
        bpm=120.0,
        beats=np.linspace(0.0, 2.0, 5),
        downbeats=np.array([0.0, 1.0, 2.0]),
        duration_s=64.0,
        source="test",
    )

    out = segment_structure_allin1(str(audio_path), y, 44100, beat)

    assert sep_calls["return_mx"] is True
    assert sep_calls["path"] == str(audio_path)
    assert spec_calls["return_mx"] is True
    assert spec_calls["sample_rate"] == 44100
    assert spec_calls["backend"] == "mlx_fast"
    assert inf_calls["compile_forward"] is False
    assert inf_calls["include_embeddings"] is False
    assert [s.label for s in out.segments] == ["intro", "verse", "outro"]
    assert out.source == "allin1"


def test_segment_structure_allin1_sample_rate_mismatch_raises(monkeypatch, tmp_path):
    import autohotcue._allin1 as mod

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / _WEIGHTS_MARKER).write_bytes(b"x")
    monkeypatch.setattr(mod, "resolve_weights_dir", lambda: weights)
    monkeypatch.setattr(mod, "_check_platform", lambda: None)
    monkeypatch.setattr(mod, "_ensure_mlx_env", lambda: None)

    class FakeSeparator:
        samplerate = 44100

        def separate_audio_file(self, path, return_mx=False):
            del path, return_mx
            return None, "stems_mx"

    monkeypatch.setattr(mod, "_get_separator", lambda: FakeSeparator())
    monkeypatch.setattr(
        mod,
        "_get_model",
        lambda _wd: SimpleNamespace(cfg=SimpleNamespace(sample_rate=48000)),
    )

    helpers = types.ModuleType("allin1_mlx.helpers")
    helpers.run_inference_mlx_spec = lambda **_: SimpleNamespace(segments=[])
    spec_mod = types.ModuleType("allin1_mlx.spectrogram")
    spec_mod.spectrogram_from_stems = lambda **_: "spec"
    monkeypatch.setitem(sys.modules, "allin1_mlx.helpers", helpers)
    monkeypatch.setitem(sys.modules, "allin1_mlx.spectrogram", spec_mod)

    beat = BeatAnalysis(
        bpm=120.0,
        beats=np.array([0.0, 0.5]),
        downbeats=np.array([0.0]),
        duration_s=1.0,
        source="test",
    )
    with pytest.raises(RuntimeError, match="samplerate"):
        segment_structure_allin1("track.wav", np.zeros(100), 44100, beat)


def test_bench_parse_engines_includes_allin1():
    from autohotcue.bench import parse_engines

    assert parse_engines("ml,ml-allin1,legacy") == ["ml", "ml-allin1", "legacy"]


def test_cli_engine_choices():
    from autohotcue.cli import _ENGINE_CHOICES

    assert "ml-allin1" in _ENGINE_CHOICES
    assert "ml" in _ENGINE_CHOICES
