"""Unit tests for ml-songformer label mapping and engine validation (no model)."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from autohotcue import analysis
from autohotcue._songformer import (
    SONGFORMER_LABELS,
    _HF_REVISION,
    _MODEL_SR,
    _apply_memory_ceiling,
    _get_model,
    _hf_revision,
    _map_songformer_segments,
    _resolve_device,
    _stub_msaf,
    segment_structure_songformer,
)
from autohotcue.backends import BeatAnalysis


def test_normalize_engine_ml_songformer():
    assert analysis.normalize_engine("ml-songformer") == ("ml-songformer", "songformer")


def test_ml_songformer_in_valid_engines():
    assert "ml-songformer" in analysis.VALID_ENGINES
    from autohotcue.bench import VALID_ENGINES

    assert "ml-songformer" in VALID_ENGINES


def test_effective_parallel_jobs_forces_single_worker_for_songformer():
    assert analysis.effective_parallel_jobs("ml-songformer", 8) == 1
    assert analysis.effective_parallel_jobs("ml", 8) == 8


def test_bench_parse_engines_includes_songformer():
    from autohotcue.bench import parse_engines

    assert parse_engines("ml,ml-songformer") == ["ml", "ml-songformer"]


def test_cli_engine_choices_includes_songformer():
    from autohotcue.cli import _ENGINE_CHOICES

    assert "ml-songformer" in _ENGINE_CHOICES


def test_songformer_label_mapping_happy_path():
    raw = [
        {"label": "intro", "start": 0.0, "end": 16.0},
        {"label": "verse", "start": 16.0, "end": 48.0},
        {"label": "chorus", "start": 48.0, "end": 80.0},
        {"label": "bridge", "start": 80.0, "end": 96.0},
        {"label": "inst", "start": 96.0, "end": 112.0},
        {"label": "outro", "start": 112.0, "end": 128.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=160.0)
    assert [(s.label, s.start, s.end) for s in segs] == [
        ("intro", 0.0, 16.0),
        ("verse", 16.0, 48.0),
        ("chorus", 48.0, 80.0),
        ("bridge", 80.0, 96.0),
        ("inst", 96.0, 112.0),
        ("outro", 112.0, 128.0),
    ]
    assert set(s.label for s in segs).issubset(SONGFORMER_LABELS)


def test_songformer_label_case_and_prechorus_mapping():
    raw = [
        {"label": "INTRO", "start": 0.0, "end": 8.0},
        {"label": "pre-chorus", "start": 8.0, "end": 16.0},
        {"label": "prechorus", "start": 16.0, "end": 24.0},
        {"label": "CHORUS", "start": 24.0, "end": 32.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=64.0)
    assert [s.label for s in segs] == ["intro", "verse", "verse", "chorus"]


def test_songformer_drops_silence_segments():
    raw = [
        {"label": "intro", "start": 0.0, "end": 1.0},
        {"label": "silence", "start": 1.0, "end": 1.2},
        {"label": "chorus", "start": 1.2, "end": 5.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=10.0)
    assert [s.label for s in segs] == ["intro", "chorus"]


def test_songformer_silence_with_non_finite_boundary_raises():
    raw = [{"label": "silence", "start": float("nan"), "end": 1.2}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw, duration_s=10.0)


def test_songformer_unknown_label_raises():
    raw = [{"label": "unknown_tag", "start": 0.0, "end": 8.0}]
    with pytest.raises(ValueError, match="unknown songformer segment label"):
        _map_songformer_segments(raw, duration_s=160.0)


def test_songformer_non_finite_boundary_raises():
    raw = [{"label": "verse", "start": float("nan"), "end": 8.0}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw, duration_s=160.0)

    raw_inf = [{"label": "verse", "start": 0.0, "end": float("inf")}]
    with pytest.raises(ValueError, match="non-finite segment boundary"):
        _map_songformer_segments(raw_inf, duration_s=160.0)


def test_songformer_clamps_and_skips_degenerate():
    raw = [
        {"label": "intro", "start": -4.0, "end": 8.0},
        {"label": "verse", "start": 8.0, "end": 4.0},
        {"label": "chorus", "start": 0.0, "end": 0.0},
        {"label": "outro", "start": 150.0, "end": 200.0},
    ]
    segs = _map_songformer_segments(raw, duration_s=160.0)
    assert len(segs) == 2
    assert segs[0].label == "intro"
    assert segs[0].start == 0.0
    assert segs[0].end == 8.0
    assert segs[1].label == "outro"
    assert segs[1].start == 150.0
    assert segs[1].end == 160.0


def test_songformer_empty_input_returns_empty():
    assert _map_songformer_segments([], duration_s=120.0) == []


def test_resolve_device_default_and_override(monkeypatch):
    monkeypatch.delenv("AUTOHOTCUE_SONGFORMER_DEVICE", raising=False)
    assert _resolve_device() == "cpu"

    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_DEVICE", "mps")
    assert _resolve_device() == "mps"


def test_hf_revision_default_and_override(monkeypatch):
    monkeypatch.delenv("AUTOHOTCUE_SONGFORMER_REVISION", raising=False)
    assert _hf_revision() == _HF_REVISION

    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_REVISION", "deadbeef")
    assert _hf_revision() == "deadbeef"


def test_get_model_raises_when_device_or_revision_changes(monkeypatch):
    import autohotcue._songformer as mod

    sentinel = object()
    monkeypatch.setattr(mod, "_model", sentinel)
    monkeypatch.setattr(mod, "_model_device", "cpu")
    monkeypatch.setattr(mod, "_model_revision", _HF_REVISION)

    with pytest.raises(RuntimeError, match="ml-songformer model already loaded"):
        _get_model("mps")

    assert _get_model("cpu") is sentinel

    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_REVISION", "other-rev")
    with pytest.raises(RuntimeError, match="ml-songformer model already loaded"):
        _get_model("cpu")


@pytest.fixture
def clean_msaf_modules():
    saved = {k: sys.modules[k] for k in ("msaf", "msaf.eval") if k in sys.modules}
    for key in ("msaf.eval", "msaf"):
        sys.modules.pop(key, None)
    yield
    for key in ("msaf.eval", "msaf"):
        sys.modules.pop(key, None)
    sys.modules.update(saved)


def test_stub_msaf_idempotent_and_blocks_compute_results(clean_msaf_modules):
    _stub_msaf()
    _stub_msaf()
    assert "msaf" in sys.modules
    with pytest.raises(RuntimeError, match="msaf is stubbed out"):
        sys.modules["msaf"].eval.compute_results()


def test_apply_memory_ceiling_zero_is_noop(monkeypatch):
    import resource

    calls: list[tuple] = []

    def fake_setrlimit(which, limits):
        calls.append((which, limits))

    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)
    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_MEM_GB", "0")
    _apply_memory_ceiling()
    assert calls == []


def test_apply_memory_ceiling_huge_value_does_not_raise(monkeypatch):
    import resource

    calls: list[tuple] = []

    def fake_setrlimit(which, limits):
        calls.append((which, limits))

    def fake_getrlimit(which):
        return (resource.RLIM_INFINITY, resource.RLIM_INFINITY)

    monkeypatch.setattr(resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)
    monkeypatch.setenv("AUTOHOTCUE_SONGFORMER_MEM_GB", "9999")
    _apply_memory_ceiling()
    assert len(calls) == 1
    which, (soft, hard) = calls[0]
    assert which == resource.RLIMIT_DATA
    assert soft == hard == int(9999 * 1024**3)


def test_segment_structure_songformer_memory_error_translation(monkeypatch):
    import autohotcue._songformer as mod

    class FakeModel:
        def __call__(self, waveform):
            raise MemoryError("out of memory")

    monkeypatch.setattr(mod, "_get_model", lambda _device: FakeModel())

    rng = np.random.default_rng(0)
    y = rng.standard_normal(44100 * 2).astype(np.float32)
    beat = BeatAnalysis(
        bpm=120.0,
        beats=np.linspace(0.0, 2.0, 5),
        downbeats=np.array([0.0, 1.0, 2.0]),
        duration_s=2.0,
        source="test",
    )

    with pytest.raises(RuntimeError, match="AUTOHOTCUE_SONGFORMER_MEM_GB"):
        segment_structure_songformer("track.wav", y, 44100, beat)


def test_segment_structure_songformer_plumbing(monkeypatch):
    import autohotcue._songformer as mod

    received: dict[str, object] = {}

    class FakeModel:
        def __call__(self, waveform):
            received["waveform"] = waveform
            return [
                {"label": "INTRO", "start": 0.0, "end": 1.0},
                {"label": "silence", "start": 1.0, "end": 1.2},
                {"label": "chorus", "start": 1.2, "end": 5.0},
            ]

    monkeypatch.setattr(mod, "_get_model", lambda _device: FakeModel())

    rng = np.random.default_rng(0)
    y = rng.standard_normal(44100 * 2).astype(np.float32)
    beat = BeatAnalysis(
        bpm=120.0,
        beats=np.linspace(0.0, 2.0, 5),
        downbeats=np.array([0.0, 1.0, 2.0]),
        duration_s=2.0,
        source="test",
    )

    out = segment_structure_songformer("track.wav", y, 44100, beat)

    wf = received["waveform"]
    assert isinstance(wf, np.ndarray)
    assert wf.dtype == np.float32
    assert len(wf) == pytest.approx(2 * _MODEL_SR, rel=0.01)
    assert [s.label for s in out.segments] == ["intro", "chorus"]
    assert out.segments[-1].end == 2.0
    assert out.source == "songformer"
    assert all(0.0 <= s.energy_rank <= 1.0 for s in out.segments)
