"""End-to-end integration tests against real audio tracks and library copies."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from autohotcue import analysis, djaydb

REAL_LIBRARY = Path(
    "/Users/payne/Music/djay/djay Media Library.djayMediaLibrary/MediaLibrary.db"
)

E2E_TRACKS = [
    "/Users/payne/Music/Setlist/Afro House 2026  Top 100   UNCOMMON/"
    "Adam Port, Stryv, Keinemusik, Orso, Malachiii - Move (Extended Mix).opus",
    "/Users/payne/Music/Setlist/Afro House 2026  Top 100   UNCOMMON/"
    "Adassiya, Greg Herma, Alex Garett - Headlights (Extended Mix).opus",
    "/Users/payne/Music/Setlist/Deep x Melodic  Anjunadeep/"
    "Maty Owl - Green & Blue (Extended Mix).opus",
    "/Users/payne/Music/Setlist/Deep x Melodic  Anjunadeep/O'Flynn - Kelsier.opus",
]

APPLY_TRACK = E2E_TRACKS[0]

library_exists = REAL_LIBRARY.is_file()


def _on_downbeat(t: float, downbeats: np.ndarray, tol_s: float = 0.015) -> bool:
    if len(downbeats) == 0:
        return False
    return float(np.min(np.abs(downbeats - t))) <= tol_s


def _beats_after(t: float, beats: np.ndarray) -> int:
    return int(np.sum(beats > t))


def _check_ordering(pos: dict[str, float]) -> None:
    """A<=B<=C<D<E<F<=G=H for present cues."""
    order = ["A", "B", "C", "D", "E", "F", "G", "H"]
    present = [k for k in order if k in pos]
    prev = None
    for letter in present:
        cur = pos[letter]
        if prev is not None:
            assert cur >= prev, f"{letter}={cur} before previous {prev}"
        if letter == "C" and "D" in pos:
            assert cur < pos["D"], f"C={cur} not before D={pos['D']}"
        if letter == "D" and "E" in pos:
            assert cur < pos["E"], f"D={cur} not before E={pos['E']}"
        if letter == "E" and "F" in pos:
            assert cur < pos["F"], f"E={cur} not before F={pos['F']}"
        prev = cur
    if "G" in pos and "H" in pos:
        assert pos["G"] == pos["H"]


@pytest.mark.parametrize("track_path", E2E_TRACKS)
def test_ml_analyze_real_track(track_path: str):
    if not Path(track_path).is_file():
        pytest.skip(f"missing track: {track_path}")

    t0 = time.perf_counter()
    track, prop = analysis.analyze(track_path, engine="ml", jobs=1)
    elapsed = time.perf_counter() - t0
    print(f"\n{Path(track_path).name}: analyze() {elapsed:.1f}s")

    assert track.engine == "ml"
    assert track.downbeats is not None and len(track.downbeats) > 0
    assert track.beats is not None and len(track.beats) > 0

    for letter, t in prop.positions.items():
        assert _on_downbeat(t, track.downbeats), (
            f"{letter}={t:.3f}s not on a tracked downbeat"
        )

    _check_ordering(prop.positions)

    if "A" in prop.positions:
        assert prop.positions["A"] <= track.duration_s * 0.25

    if "G" in prop.positions:
        assert _beats_after(prop.positions["G"], track.beats) >= 8


@pytest.mark.skipif(not library_exists, reason="real djay library not present")
@pytest.mark.skipif(not Path(APPLY_TRACK).is_file(), reason="apply e2e track missing")
def test_apply_verify_roundtrip(tmp_path):
    from autohotcue.cli import _djay_running

    if _djay_running():
        pytest.skip("djay Pro is running — quit it before apply e2e")
    """Copy library to tmp_path, apply ml cues, verify readback matches proposal."""
    lib_dir = tmp_path / "libcopy"
    lib_dir.mkdir()
    db_path = lib_dir / "MediaLibrary.db"
    shutil.copy2(REAL_LIBRARY, db_path)
    for suffix in ("-wal", "-shm"):
        src = Path(str(REAL_LIBRARY) + suffix)
        if src.is_file():
            shutil.copy2(src, lib_dir / f"MediaLibrary.db{suffix}")

    from autohotcue.backends import init_worker
    from autohotcue.cli import _precheck_one, _write_one

    init_worker(1)

    class Args:
        bpm = None
        force = True

    db = djaydb.DjayDB(str(db_path))
    try:
        key, existing, bpm = _precheck_one(db, APPLY_TRACK, Args())
        assert key is not None

        track, prop = analysis.analyze(APPLY_TRACK, known_bpm=bpm, engine="ml", jobs=1)
        backed = {"done": False}

        def ensure_backup():
            if not backed["done"]:
                db.backup(tmp_path / "backups")
                backed["done"] = True

        n = _write_one(db, key, existing, prop, ensure_backup)
        assert n > 0
        db.checkpoint()

        doc = db.get("mediaItemUserData", key)
        assert doc is not None
        stored = {}
        from autohotcue import tsaf

        for cp in doc.root.get("cuePoints").items:
            num = cp.get("number")
            idx = num.value if isinstance(num, tsaf.Int) else (0 if num.tag == 0x2D else 1)
            letter = chr(65 + idx)
            stored[letter] = float(cp.get("time").value)

        for letter, t in prop.positions.items():
            assert letter in stored, f"missing cue {letter} after apply"
            assert stored[letter] == pytest.approx(t, abs=1e-3), (
                f"{letter}: stored {stored[letter]} != proposed {t}"
            )
    finally:
        db.close()
