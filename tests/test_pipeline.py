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


def test_all_items_processed_exactly_once(monkeypatch):
    monkeypatch.setattr(pl, "decode", lambda path: path)
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: ("R", path))
    items = [(i, f"/{i}", None) for i in range(25)]
    out = list(pl.iter_analyze_pipelined(items, decode_threads=4))
    assert sorted(tok for tok, _, _, _ in out) == list(range(25))
    assert all(err is None for *_, err in out)
    assert all(result == ("R", path) for _, path, result, _ in out)
    assert not _decode_threads_alive()


def test_completion_order_independent(monkeypatch):
    def fake_decode(path):
        idx = int(path.strip("/"))
        time.sleep((10 - idx) * 0.005)  # later indices finish first
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: ("R", path))
    items = [(i, f"/{i}", None) for i in range(10)]
    out = list(pl.iter_analyze_pipelined(items, decode_threads=4))
    assert sorted(tok for tok, _, _, _ in out) == list(range(10))


def test_decode_error_surfaced_per_track(monkeypatch):
    def fake_decode(path):
        if path == "/bad":
            raise ValueError("boom")
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: ("R", path))
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
    monkeypatch.setattr(pl, "analyze_decoded", fake_analyze)
    items = [(0, "/good", None), (1, "/bad", None)]
    got = {tok: (result, err)
           for tok, _, result, err in pl.iter_analyze_pipelined(items, decode_threads=2)}
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
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: ("R", path))

    decode_threads, lookahead = 3, 2
    maxsize = decode_threads + lookahead
    items = [(i, f"/{i}", None) for i in range(50)]
    gen = pl.iter_analyze_pipelined(
        items, decode_threads=decode_threads, lookahead=lookahead,
    )
    first = next(gen)  # start the machinery, then pause the consumer
    time.sleep(0.3)    # let decoders saturate the bounded queue
    with lock:
        produced = len(decoded)
    # consumed 1; the queue holds <= maxsize; each thread may hold one more
    # in hand blocked on put. Far below the 50 items -> prefetch is bounded.
    assert produced <= 1 + maxsize + decode_threads
    rest = list(gen)
    gen.close()
    assert {first[0], *(tok for tok, *_ in rest)} == set(range(50))
    assert not _decode_threads_alive()


def test_early_break_tears_down_threads(monkeypatch):
    def slow_decode(path):
        time.sleep(0.05)
        return path
    monkeypatch.setattr(pl, "decode", slow_decode)
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: ("R", path))
    items = [(i, f"/{i}", None) for i in range(200)]
    gen = pl.iter_analyze_pipelined(items, decode_threads=3)
    next(gen)          # start, consume one
    gen.close()        # GeneratorExit -> stop + join in the finally
    deadline = time.time() + 3.0
    while time.time() < deadline and _decode_threads_alive():
        time.sleep(0.02)
    assert not _decode_threads_alive()


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


# --- cmd_propose routing through the pipeline (no models) ---------------------

def test_cmd_propose_folder_routes_through_pipeline(monkeypatch, tmp_path, capsys):
    from autohotcue import cli

    for name in ("a.opus", "b.opus", "c.opus"):
        (tmp_path / name).write_bytes(b"")  # extension is all _expand_paths checks

    track = types.SimpleNamespace(
        engine="ml-bass", bpm=124.0, first_beat_s=0.5, duration_s=10.0, grid_fit=None,
    )
    prop = analysis.CueProposal(positions={"A": 0.0}, notes=[])
    seen: list[str] = []

    def fake_decode(path):
        seen.append(path)
        return path
    monkeypatch.setattr(pl, "decode", fake_decode)
    monkeypatch.setattr(pl, "analyze_decoded", lambda y, path, **k: (track, prop))

    args = types.SimpleNamespace(
        path=str(tmp_path), bpm=None, library=None, engine="ml-bass",
        nudge_beats=0.0, jobs=1, decode_threads=2, no_pipeline=False,
    )
    cli.cmd_propose(args)

    assert sorted(seen) == [str(tmp_path / n) for n in ("a.opus", "b.opus", "c.opus")]
    out = capsys.readouterr().out
    assert "3 analyzed, 0 failed (3 files)" in out
    assert not _decode_threads_alive()
