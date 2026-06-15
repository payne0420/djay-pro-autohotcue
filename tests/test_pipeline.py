"""Decode-ahead pipeline + analyze/analyze_decoded split.

All fast and model-free: the decode and inference stages are replaced with
stubs, so these never load beat_this or touch the GPU.
"""
from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

import autohotcue.pipeline as pl
from autohotcue import analysis
from autohotcue.backends import BeatAnalysis


def _decode_threads_alive() -> list[str]:
    return [t.name for t in threading.enumerate()
            if t.name.startswith("decode-") and t.is_alive()]


def _dsp_threads_alive() -> list[str]:
    return [t.name for t in threading.enumerate()
            if t.name.startswith("dsp-") and t.is_alive()]


def _stub_offload(monkeypatch, *, cues_fn=None, infer_fn=None):
    """Stub the CPU-DSP offload path (default for ml-bass)."""
    if infer_fn is None:
        infer_fn = lambda y, device=None: "beat"
    if cues_fn is None:
        cues_fn = lambda y, path, beat, **k: ("R", path)
    monkeypatch.setattr(pl, "infer_beats", infer_fn)
    monkeypatch.setattr(pl, "cues_from_beats", cues_fn)


def _stub_inline(monkeypatch, analyze_fn=None):
    """Stub the inline analyze_decoded path (non-CPU-DSP engines / dsp_threads=0)."""
    if analyze_fn is None:
        analyze_fn = lambda y, path, **k: ("R", path)
    monkeypatch.setattr(pl, "analyze_decoded", analyze_fn)


def test_all_items_processed_exactly_once(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    _stub_offload(monkeypatch)
    items = [(i, f"/{i}", None) for i in range(25)]
    out = list(pl.iter_analyze_pipelined(items, decode_threads=4))
    assert sorted(tok for tok, _, _, _ in out) == list(range(25))
    assert all(err is None for *_, err in out)
    assert all(result == ("R", path) for _, path, result, _ in out)
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


def test_completion_order_independent(monkeypatch):
    def fake_decode(path):
        idx = int(path.strip("/"))
        time.sleep((10 - idx) * 0.005)  # later indices finish first
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    _stub_offload(monkeypatch)
    items = [(i, f"/{i}", None) for i in range(10)]
    out = list(pl.iter_analyze_pipelined(items, decode_threads=4))
    assert sorted(tok for tok, _, _, _ in out) == list(range(10))


def test_decode_error_surfaced_per_track(monkeypatch):
    def fake_decode(path):
        if path == "/bad":
            raise ValueError("boom")
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    _stub_offload(monkeypatch)
    items = [(0, "/good", None), (1, "/bad", None), (2, "/good2", None)]
    got = {tok: (path, result, err)
           for tok, path, result, err in pl.iter_analyze_pipelined(items, decode_threads=2)}
    assert isinstance(got[1][1], type(None)) and isinstance(got[1][2], ValueError)
    assert got[0][1] == ("R", "/good")
    assert got[2][1] == ("R", "/good2")


def test_analyze_error_surfaced_per_track(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)

    def fake_analyze(y, path, **k):
        if path == "/bad":
            raise RuntimeError("nope")
        return ("R", path)
    _stub_inline(monkeypatch, fake_analyze)

    items = [(0, "/good", None), (1, "/bad", None)]
    got = {tok: (result, err)
           for tok, _, result, err in pl.iter_analyze_pipelined(
               items, decode_threads=2, engine="ml-songformer",
           )}
    assert got[0] == (("R", "/good"), None)
    assert got[1][0] is None and isinstance(got[1][1], RuntimeError)


def test_backpressure_bounds_prefetch(monkeypatch):
    decoded: list[str] = []
    lock = threading.Lock()

    def fake_decode(path):
        with lock:
            decoded.append(path)
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    _stub_offload(monkeypatch)

    decode_threads, lookahead, dsp_threads = 3, 2, 1
    max_decode = decode_threads + lookahead
    max_in_flight = max_decode + 1 + (dsp_threads + 1) + dsp_threads
    items = [(i, f"/{i}", None) for i in range(50)]
    gen = pl.iter_analyze_pipelined(
        items, decode_threads=decode_threads, lookahead=lookahead,
        dsp_threads=dsp_threads,
    )
    first = next(gen)
    time.sleep(0.3)
    with lock:
        produced = len(decoded)
    assert produced <= 1 + max_in_flight + decode_threads
    rest = list(gen)
    gen.close()
    assert {first[0], *(tok for tok, *_ in rest)} == set(range(50))
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


def test_early_break_tears_down_threads(monkeypatch):
    def slow_decode(path):
        time.sleep(0.05)
        return path
    monkeypatch.setattr(pl, "decode", slow_decode)
    _stub_offload(monkeypatch)
    items = [(i, f"/{i}", None) for i in range(200)]
    gen = pl.iter_analyze_pipelined(items, decode_threads=3)
    next(gen)
    gen.close()
    deadline = time.time() + 3.0
    while time.time() < deadline and (_decode_threads_alive() or _dsp_threads_alive()):
        time.sleep(0.02)
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


# --- DSP offload path ---------------------------------------------------------

def test_dsp_offload_all_items_once(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    beats: list[str] = []

    def fake_infer(y, device=None):
        beats.append("infer")
        return f"beat-{y}"

    def fake_cues(y, path, beat, **k):
        return ("R", path, beat)
    _stub_offload(monkeypatch, infer_fn=fake_infer, cues_fn=fake_cues)

    items = [(i, f"/{i}", None) for i in range(12)]
    out = list(pl.iter_analyze_pipelined(items, decode_threads=2, dsp_threads=1))
    assert sorted(tok for tok, _, _, _ in out) == list(range(12))
    assert all(err is None for *_, err in out)
    assert len(beats) == 12
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


def test_dsp_stage_error_surfaced_per_track(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)

    def fake_cues(y, path, beat, **k):
        if path == "/bad":
            raise ValueError("dsp boom")
        return ("R", path)
    _stub_offload(monkeypatch, cues_fn=fake_cues)

    items = [(0, "/good", None), (1, "/bad", None), (2, "/good2", None)]
    got = {tok: (result, err)
           for tok, _, result, err in pl.iter_analyze_pipelined(items, decode_threads=2)}
    assert got[0] == (("R", "/good"), None)
    assert got[1][0] is None and isinstance(got[1][1], ValueError)
    assert got[2] == (("R", "/good2"), None)


def test_inference_error_surfaced_in_offload_path(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)

    def fake_infer(y, device=None):
        if y == "/bad":
            raise RuntimeError("infer nope")
        return "beat"
    _stub_offload(monkeypatch, infer_fn=fake_infer)

    items = [(0, "/good", None), (1, "/bad", None)]
    got = {tok: (result, err)
           for tok, _, result, err in pl.iter_analyze_pipelined(items, decode_threads=2)}
    assert got[0] == (("R", "/good"), None)
    assert got[1][0] is None and isinstance(got[1][1], RuntimeError)


def test_non_cpu_dsp_engine_stays_inline(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    seen: list[str] = []

    def fake_analyze(y, path, **k):
        seen.append(path)
        return ("R", path)
    _stub_inline(monkeypatch, fake_analyze)

    cues_called = False

    def fake_cues(*a, **k):
        nonlocal cues_called
        cues_called = True
        return ("R", "nope")
    monkeypatch.setattr(pl, "cues_from_beats", fake_cues)

    items = [(0, "/a", None), (1, "/b", None)]
    out = list(pl.iter_analyze_pipelined(
        items, decode_threads=2, engine="ml-songformer",
    ))
    assert len(out) == 2
    assert sorted(seen) == ["/a", "/b"]
    assert not cues_called
    assert not _dsp_threads_alive()


# --- analyze() / analyze_decoded() equivalence (guards the refactor) ----------

def _stub_pipeline_stages(monkeypatch):
    beat = BeatAnalysis(
        bpm=124.0,
        beats=np.array([0.0, 0.5, 1.0]),
        downbeats=np.array([0.0, 2.0]),
        duration_s=10.0,
        source="fake",
    )
    grid = types.SimpleNamespace(ok=False, reason="stub")
    prop = analysis.CueProposal(positions={"A": 0.0}, notes=["stub"])
    monkeypatch.setattr(analysis, "decode", lambda path, sr=analysis.SR: np.zeros(100, np.float32))
    monkeypatch.setattr(analysis, "_resolve_device", lambda device, jobs=1: "cpu")
    monkeypatch.setattr("autohotcue.backends.track_beats", lambda y, device=None: beat)
    monkeypatch.setattr("autohotcue.gridlock.fit_grid", lambda y, sr, b, d, bpm: grid)
    monkeypatch.setattr(
        "autohotcue.bassline.propose_cues_bass",
        lambda y, sr, beat, grid, djay_bpm=None: (prop, None),
    )
    return beat, prop


def test_analyze_delegates_to_analyze_decoded(monkeypatch):
    beat, prop = _stub_pipeline_stages(monkeypatch)
    t1, p1 = analysis.analyze("/x.opus", engine="ml-bass")
    t2, p2 = analysis.analyze_decoded(analysis.decode("/x.opus"), "/x.opus", engine="ml-bass")
    assert p1 is prop and p2 is prop
    assert (t1.bpm, t1.first_beat_s, t1.duration_s, t1.engine, t1.djay_bpm) == \
           (t2.bpm, t2.first_beat_s, t2.duration_s, t2.engine, t2.djay_bpm)
    assert t1.first_beat_s == float(beat.downbeats[0])


def test_cues_from_beats_matches_analyze_decoded(monkeypatch):
    beat, prop = _stub_pipeline_stages(monkeypatch)
    y = analysis.decode("/x.opus")
    t1, p1 = analysis.analyze_decoded(y, "/x.opus", engine="ml-bass", device="cpu")
    t2, p2 = analysis.cues_from_beats(
        y, "/x.opus", analysis.infer_beats(y, "cpu"), engine="ml-bass",
    )
    assert p1.positions == p2.positions
    assert p1.notes == p2.notes
    assert t1.bpm == t2.bpm
    assert round(t1.first_beat_s, 6) == round(t2.first_beat_s, 6)
    assert p1 is prop and p2 is prop


def _stub_structure_stages(monkeypatch):
    from autohotcue.backends import BeatAnalysis, Segment, StructureAnalysis
    from autohotcue.gridlock import GridFit

    beat = BeatAnalysis(
        bpm=124.0,
        beats=np.array([0.0, 0.5, 1.0]),
        downbeats=np.array([0.0, 2.0]),
        duration_s=10.0,
        source="fake",
    )
    grid = GridFit(
        bpm=124.0, render_bpm=124.0, anchor_s=1.0,
        beat_fit=0.0, bar_resid_std=0.0, splice_jump=0.0, ok=True, reason="stub",
    )
    seg = Segment(0.0, 4.0, "intro", energy_rank=0.5)
    structure = StructureAnalysis(segments=[seg], source="librosa")
    prop = analysis.CueProposal(positions={"A": 0.0, "D": 8.0}, notes=["struct"])
    monkeypatch.setattr(analysis, "_resolve_device", lambda device, jobs=1: "cpu")
    monkeypatch.setattr("autohotcue.backends.track_beats", lambda y, device=None: beat)
    monkeypatch.setattr("autohotcue.gridlock.fit_grid", lambda y, sr, b, d, bpm: grid)
    monkeypatch.setattr(
        "autohotcue.backends.segment_structure",
        lambda path, y, sr, beat, **k: structure,
    )
    monkeypatch.setattr("autohotcue.cuepolicy.propose_cues", lambda beat, structure, **k: prop)
    return beat, prop, structure


def test_cues_from_beats_structure_branch_matches_analyze_decoded(monkeypatch):
    beat, prop, structure = _stub_structure_stages(monkeypatch)
    y = np.zeros(100, np.float32)
    for engine in ("ml", "ml-librosa"):
        t1, p1 = analysis.analyze_decoded(y, "/x.opus", engine=engine, device="cpu")
        t2, p2 = analysis.cues_from_beats(
            y, "/x.opus", analysis.infer_beats(y, "cpu"), engine=engine,
        )
        assert p1.positions == p2.positions
        assert p1.notes == p2.notes
        assert t1.bpm == t2.bpm
        assert round(t1.first_beat_s, 6) == round(t2.first_beat_s, 6)
        assert t1.segments == t2.segments == list(structure.segments)
        assert p1 is prop and p2 is prop


def test_cues_from_beats_known_bpm_and_nudge_matches_analyze_decoded(monkeypatch):
    _stub_structure_stages(monkeypatch)
    y = np.zeros(100, np.float32)
    dsp_kwargs = dict(known_bpm=128.0, engine="ml", nudge_beats=1.0)
    t1, p1 = analysis.analyze_decoded(y, "/x.opus", device="cpu", **dsp_kwargs)
    t2, p2 = analysis.cues_from_beats(
        y, "/x.opus", analysis.infer_beats(y, "cpu"), **dsp_kwargs,
    )
    assert p1.positions == p2.positions
    assert p1.notes == p2.notes
    assert t1.bpm == t2.bpm
    assert round(t1.first_beat_s, 6) == round(t2.first_beat_s, 6)
    assert t1.segments == t2.segments
    assert t1.djay_bpm == t2.djay_bpm == 128.0


def test_dsp_threads_zero_forces_inline(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    seen: list[str] = []

    def fake_analyze(y, path, **k):
        seen.append(path)
        return ("R", path)
    _stub_inline(monkeypatch, fake_analyze)

    cues_called = False

    def fake_cues(*a, **k):
        nonlocal cues_called
        cues_called = True
        return ("R", "nope")
    monkeypatch.setattr(pl, "cues_from_beats", fake_cues)

    items = [(i, f"/{i}", None) for i in range(4)]
    out = list(pl.iter_analyze_pipelined(
        items, decode_threads=2, dsp_threads=0, engine="ml-bass",
    ))
    assert len(out) == 4
    assert sorted(seen) == [f"/{i}" for i in range(4)]
    assert not cues_called
    assert not _dsp_threads_alive()


@pytest.mark.parametrize(
    "engine,expected",
    [
        ("ml-bass", True),
        ("ml", True),
        ("ml-librosa", True),
        ("ml-allin1", False),
        ("ml-songformer", False),
        ("legacy", False),
    ],
)
def test_is_cpu_dsp_engine(engine, expected):
    assert analysis.is_cpu_dsp_engine(engine) is expected


def test_allin1_engine_stays_inline(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    seen: list[str] = []

    def fake_analyze(y, path, **k):
        seen.append(path)
        return ("R", path)
    _stub_inline(monkeypatch, fake_analyze)

    cues_called = False

    def fake_cues(*a, **k):
        nonlocal cues_called
        cues_called = True
        return ("R", "nope")
    monkeypatch.setattr(pl, "cues_from_beats", fake_cues)

    items = [(0, "/a", None), (1, "/b", None)]
    out = list(pl.iter_analyze_pipelined(
        items, decode_threads=2, engine="ml-allin1",
    ))
    assert len(out) == 2
    assert sorted(seen) == ["/a", "/b"]
    assert not cues_called
    assert not _dsp_threads_alive()


def test_close_waits_for_slow_inflight_work(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)

    def slow_cues(y, path, beat, **k):
        if path == "/slow":
            time.sleep(3.5)
        return ("R", path)
    _stub_offload(monkeypatch, cues_fn=slow_cues)

    items = [(0, "/fast", None), (1, "/slow", None)]
    gen = pl.iter_analyze_pipelined(items, decode_threads=1, dsp_threads=1)
    next(gen)
    time.sleep(0.2)
    gen.close()
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


# --- cmd_propose routing through the pipeline (no models) ---------------------

def test_cmd_propose_folder_routes_through_pipeline(monkeypatch, tmp_path, capsys):
    from autohotcue import cli

    for name in ("a.opus", "b.opus", "c.opus"):
        (tmp_path / name).write_bytes(b"")

    track = types.SimpleNamespace(
        engine="ml-bass", bpm=124.0, first_beat_s=0.5, duration_s=10.0, grid_fit=None,
    )
    prop = analysis.CueProposal(positions={"A": 0.0}, notes=[])
    seen: list[str] = []

    def fake_decode(path):
        seen.append(path)
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    _stub_offload(monkeypatch, cues_fn=lambda y, path, beat, **k: (track, prop))

    args = types.SimpleNamespace(
        path=str(tmp_path), bpm=None, library=None, engine="ml-bass",
        nudge_beats=0.0, jobs=1, decode_threads=2, dsp_threads=1, no_pipeline=False,
    )
    cli.cmd_propose(args)

    assert sorted(seen) == [str(tmp_path / n) for n in ("a.opus", "b.opus", "c.opus")]
    out = capsys.readouterr().out
    assert "3 analyzed, 0 failed (3 files)" in out
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()


def test_cmd_apply_pipeline_precheck_failure_does_not_abort_batch(monkeypatch, tmp_path, capsys):
    """A generic precheck failure on one track must be reported per-track and the
    rest of the folder still processed (parity with the sequential -j1 path)."""
    from autohotcue import cli, djaydb

    for name in ("a.opus", "b.opus", "c.opus"):
        (tmp_path / name).write_bytes(b"")

    monkeypatch.setattr(cli, "_djay_running", lambda: False)

    class _FakeDB:
        def backup(self, d): return "backup"
        def checkpoint(self): pass
        def close(self): pass
    monkeypatch.setattr(djaydb, "DjayDB", lambda *a, **k: _FakeDB())

    def fake_precheck(db, path, args):
        if path.endswith("b.opus"):
            raise ValueError("corrupt TSAF")
        return (path, None, 120.0)
    monkeypatch.setattr(cli, "_precheck_one", fake_precheck)

    track = types.SimpleNamespace(
        engine="ml-bass", bpm=124.0, first_beat_s=0.5, duration_s=10.0, grid_fit=None,
    )
    prop = analysis.CueProposal(positions={"A": 0.0}, notes=[])
    monkeypatch.setattr(pl, "decode", lambda path: path)
    _stub_offload(monkeypatch, cues_fn=lambda y, path, beat, **k: (track, prop))
    monkeypatch.setattr(cli, "_write_one", lambda *a, **k: 8)

    args = types.SimpleNamespace(
        path=str(tmp_path), bpm=None, library="dummy", engine="ml-bass",
        nudge_beats=0.0, jobs=1, decode_threads=2, dsp_threads=1, no_pipeline=False,
        force=False, no_grid_lock=False, backup_dir=str(tmp_path / "bk"),
    )
    cli.cmd_apply(args)

    out = capsys.readouterr().out
    assert "FAILED: corrupt TSAF" in out
    assert "2 written, 0 skipped, 1 failed (3 files)" in out
    assert not _decode_threads_alive()
    assert not _dsp_threads_alive()
