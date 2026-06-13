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

TEST_PLAYLIST = Path("/Users/payne/Music/Setlist/zzzzzTEST")

E2E_TRACKS = [
    str(TEST_PLAYLIST / "Andrea Oliva, Moeaike, Shimza - I Love You So - Shimza Remix (Extended Mix).opus"),
    str(TEST_PLAYLIST / "Jan Blomqvist - Maybe Not (Extended Mix).opus"),
    str(TEST_PLAYLIST / "MIRAMAR - Esplanade (Extended Mix).opus"),
    str(
        TEST_PLAYLIST
        / "Brando, Hugo Cantarra, HUGEL - Look Into My Eyes - HUGEL & Hugo Cantarra Remix (Extended Mix).opus"
    ),
]

FINGERPRINT_TRACK = str(
    TEST_PLAYLIST
    / "Adam Port, Stryv, Keinemusik, Orso, Malachiii - Move (Extended Mix).opus"
)

APPLY_TRACK = E2E_TRACKS[0]

library_exists = REAL_LIBRARY.is_file()


def _copy_library_to(dest_dir: Path) -> Path:
    """Copy the live library into a temp dir for read-only or write e2e tests."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    db_path = dest_dir / "MediaLibrary.db"
    shutil.copy2(REAL_LIBRARY, db_path)
    for suffix in ("-wal", "-shm"):
        src = Path(str(REAL_LIBRARY) + suffix)
        if src.is_file():
            shutil.copy2(src, dest_dir / f"MediaLibrary.db{suffix}")
    return db_path


def _on_downbeat(t: float, downbeats: np.ndarray, tol_s: float = 0.015) -> bool:
    if len(downbeats) == 0:
        return False
    return float(np.min(np.abs(downbeats - t))) <= tol_s


def _beats_after(t: float, beats: np.ndarray) -> int:
    return int(np.sum(beats > t))


def _check_ordering(pos: dict[str, float]) -> None:
    """A<=B<=C<D<E<F<=G<=H for present cues."""
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
        if letter == "F" and "G" in pos:
            assert cur < pos["G"], f"F={cur} not before G={pos['G']}"
        prev = cur
    if "G" in pos and "H" in pos:
        assert pos["G"] <= pos["H"]


def _allin1_prereqs() -> bool:
    from pathlib import Path as P

    if P("all-in-one-mlx/mlx-weights/harmonix-fold0_mlx.npz").is_file():
        return True
    try:
        from autohotcue._allin1 import resolve_weights_dir

        resolve_weights_dir()
        return True
    except Exception:
        return False


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


@pytest.mark.parametrize("track_path", E2E_TRACKS)
def test_ml_bass_analyze_real_track(track_path: str):
    if not Path(track_path).is_file():
        pytest.skip(f"missing track: {track_path}")

    t0 = time.perf_counter()
    track, prop = analysis.analyze(track_path, engine="ml-bass", jobs=1)
    elapsed = time.perf_counter() - t0
    print(f"\n{Path(track_path).name}: ml-bass analyze() {elapsed:.1f}s")

    assert track.engine == "ml-bass"
    assert track.bass is not None
    assert "A" in prop.positions
    _check_ordering(prop.positions)

    fit = track.grid_fit
    if fit is not None and fit.ok:
        bar_period = 60.0 / fit.bpm
        anchor = fit.anchor_s
        for letter, t in prop.positions.items():
            k = (t - anchor) / bar_period
            assert abs(k - round(k)) < 1e-6, (
                f"{letter}={t:.3f}s not on grid-lock lattice"
            )


@pytest.mark.parametrize("track_path", E2E_TRACKS)
@pytest.mark.skipif(not _allin1_prereqs(), reason="ml-allin1 prerequisites missing")
def test_ml_allin1_analyze_real_track(track_path: str):
    if not Path(track_path).is_file():
        pytest.skip(f"missing track: {track_path}")

    t0 = time.perf_counter()
    track, prop = analysis.analyze(track_path, engine="ml-allin1", jobs=1)
    elapsed = time.perf_counter() - t0
    print(f"\n{Path(track_path).name}: ml-allin1 analyze() {elapsed:.1f}s")

    assert track.engine == "ml-allin1"
    assert track.downbeats is not None and len(track.downbeats) > 0
    assert track.beats is not None and len(track.beats) > 0
    assert track.segments is not None and len(track.segments) > 0

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
    """Copy library to tmp_path, apply ml cues, verify readback matches proposal."""
    from autohotcue.cli import _djay_running

    if _djay_running():
        pytest.skip("djay Pro is running — quit it before apply e2e")

    db_path = _copy_library_to(tmp_path / "libcopy")

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

        from autohotcue.gridlock import snap_cues

        n = _write_one(
            db, key, existing, track, prop, ensure_backup,
            grid_lock=True, force=True,
        )
        assert n > 0
        db.checkpoint()

        expected = (
            snap_cues(prop.positions, track.grid_fit)
            if track.grid_fit is not None and track.grid_fit.ok
            else prop.positions
        )

        doc = db.get("mediaItemUserData", key)
        assert doc is not None
        stored = {}
        from autohotcue import tsaf

        for cp in doc.root.get("cuePoints").items:
            num = cp.get("number")
            idx = num.value if isinstance(num, tsaf.Int) else (0 if num.tag == 0x2D else 1)
            letter = chr(65 + idx)
            stored[letter] = float(cp.get("time").value)

        for letter, t in expected.items():
            assert letter in stored, f"missing cue {letter} after apply"
            assert stored[letter] == pytest.approx(t, abs=1e-3), (
                f"{letter}: stored {stored[letter]} != proposed {t}"
            )
    finally:
        db.close()


@pytest.mark.skipif(not library_exists, reason="real djay library not present")
@pytest.mark.skipif(
    not Path(FINGERPRINT_TRACK).is_file(),
    reason="fingerprint guard e2e track missing",
)
def test_apply_skips_fingerprint_user_data(tmp_path):
    """Real library record with ADCAudioAlignmentFingerprint must skip apply."""
    from autohotcue.cli import _Skip, _djay_running, _precheck_one

    if _djay_running():
        pytest.skip("djay Pro is running — quit it before apply e2e")

    db_path = _copy_library_to(tmp_path / "libcopy")

    class Args:
        bpm = None
        force = True

    db = djaydb.DjayDB(str(db_path))
    try:
        with pytest.raises(_Skip, match="audio alignment fingerprint"):
            _precheck_one(db, FINGERPRINT_TRACK, Args())
    finally:
        db.close()
