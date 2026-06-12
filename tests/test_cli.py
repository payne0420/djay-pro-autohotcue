"""Tests for CLI path expansion (directory batch mode)."""
from __future__ import annotations

import sqlite3

import pytest

from autohotcue import djaydb, tsaf
from autohotcue.cli import _Skip, _bpm_for, _djay_bpm_for_path, _expand_paths, _precheck_one


def _library_db(tmp_path, track_path: str, *, bpm: float | None = 122.0):
    key = "track1"
    db_path = tmp_path / "MediaLibrary.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE "database2" (rowid INTEGER PRIMARY KEY, collection CHAR, '
        "key CHAR, data BLOB, metadata BLOB)"
    )
    conn.execute('CREATE UNIQUE INDEX "true_primary_key" ON "database2" ("collection", "key")')
    loc = tsaf.Obj("ADCMediaItemLocation")
    loc.fields = [
        ("uuid", key),
        ("sourceURIs", tsaf.Arr(tsaf.TAG_ARRAY_B, [tsaf.Url(f"file://{track_path}")])),
    ]
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("localMediaItemLocations", key, tsaf.serialize(tsaf.Document((3, 3), loc))),
    )
    analyzed = tsaf.Obj("ADCMediaItemAnalyzedData")
    if bpm is not None:
        analyzed.fields = [("bpm", tsaf.F32.of(bpm))]
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("mediaItemAnalyzedData", key, tsaf.serialize(tsaf.Document((3, 3), analyzed))),
    )
    conn.commit()
    conn.close()
    return db_path


def _title_id(key: str = "track1") -> tsaf.Obj:
    tid = tsaf.Obj("ADCMediaItemTitleID")
    tid.fields = [
        ("uuid", key),
        ("title", "T"),
        ("artist", "A"),
        ("duration", tsaf.F32.of(200.0)),
    ]
    return tid


def _library_db_with_user_data(
    tmp_path,
    track_path: str,
    user_doc: tsaf.Document,
    *,
    bpm: float | None = 122.0,
    key: str = "track1",
):
    db_path = tmp_path / "MediaLibrary.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        'CREATE TABLE "database2" (rowid INTEGER PRIMARY KEY, collection CHAR, '
        "key CHAR, data BLOB, metadata BLOB)"
    )
    conn.execute('CREATE UNIQUE INDEX "true_primary_key" ON "database2" ("collection", "key")')
    loc = tsaf.Obj("ADCMediaItemLocation")
    loc.fields = [
        ("uuid", key),
        ("sourceURIs", tsaf.Arr(tsaf.TAG_ARRAY_B, [tsaf.Url(f"file://{track_path}")])),
    ]
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("localMediaItemLocations", key, tsaf.serialize(tsaf.Document((3, 3), loc))),
    )
    tid = _title_id(key)
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("mediaItemTitleIDs", key, tsaf.serialize(tsaf.Document((3, 3), tid))),
    )
    analyzed = tsaf.Obj("ADCMediaItemAnalyzedData")
    if bpm is not None:
        analyzed.fields = [("bpm", tsaf.F32.of(bpm))]
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("mediaItemAnalyzedData", key, tsaf.serialize(tsaf.Document((3, 3), analyzed))),
    )
    conn.execute(
        "INSERT INTO database2 (collection, key, data) VALUES (?, ?, ?)",
        ("mediaItemUserData", key, tsaf.serialize(user_doc)),
    )
    conn.commit()
    conn.close()
    return db_path


class _ApplyArgs:
    force = True
    bpm = None


def test_single_file_passed_through(tmp_path):
    f = tmp_path / "track.opus"
    f.write_bytes(b"")
    assert _expand_paths(str(f)) == [str(f)]


def test_single_file_extension_not_filtered(tmp_path):
    # A file given explicitly is used as-is, whatever its extension —
    # ffmpeg decides what it can decode, not the extension list.
    f = tmp_path / "track.weird"
    f.write_bytes(b"")
    assert _expand_paths(str(f)) == [str(f)]


def test_directory_recursive_audio_only_sorted(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    audio = [
        tmp_path / "b.opus",
        tmp_path / "a.MP3",  # extension match is case-insensitive
        tmp_path / "sub" / "c.ogg",
        tmp_path / "sub" / "deeper" / "d.flac",
    ]
    other = [
        tmp_path / "cover.jpg",
        tmp_path / "notes.txt",
        tmp_path / "sub" / "playlist.m3u",
    ]
    for f in audio + other:
        f.write_bytes(b"")
    assert _expand_paths(str(tmp_path)) == sorted(str(f) for f in audio)


def test_directory_skips_hidden_files_and_dirs(tmp_path):
    (tmp_path / ".cache").mkdir()
    visible = tmp_path / "track.opus"
    visible.write_bytes(b"")
    (tmp_path / "._track.opus").write_bytes(b"")  # AppleDouble sidecar
    (tmp_path / ".hidden.mp3").write_bytes(b"")
    (tmp_path / ".cache" / "buried.mp3").write_bytes(b"")
    assert _expand_paths(str(tmp_path)) == [str(visible)]


def test_directory_with_no_audio_exits(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(SystemExit):
        _expand_paths(str(tmp_path))


def test_djay_bpm_for_path_missing_analyzed_bpm_returns_none(tmp_path):
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    db = djaydb.DjayDB(_library_db(tmp_path, str(track), bpm=None))
    try:
        assert _djay_bpm_for_path(db, str(track), None) is None
        assert _djay_bpm_for_path(db, str(track), 128.0) == 128.0
    finally:
        db.close()


def test_djay_bpm_for_path_operational_error_returns_fallback(tmp_path):
    """Broken library reads must degrade on propose, not crash."""
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    db_path = _library_db(tmp_path, str(track))
    db = djaydb.DjayDB(db_path)
    db_path.unlink()
    try:
        assert _djay_bpm_for_path(db, str(track), 128.0) == 128.0
    finally:
        db.close()


def test_bpm_for_still_raises_when_required(tmp_path):
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    db = djaydb.DjayDB(_library_db(tmp_path, str(track), bpm=None))
    try:
        with pytest.raises(_Skip):
            _bpm_for(db, "track1", None)
    finally:
        db.close()


def test_precheck_skips_unsafe_analysis_objects(tmp_path):
    """Records with unknown nested Obj types must skip the entire edit."""
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    tid = _title_id()
    fp = tsaf.Obj("ADCAudioAlignmentFingerprint")
    fp.fields = [("payload", tsaf.Data(b"\x00\x01\x02"))]
    root = tsaf.Obj("ADCMediaItemUserData")
    root.fields = [
        ("uuid", "track1"),
        ("titleIDs", tsaf.Arr(tsaf.TAG_ARRAY_B, [tid])),
        ("cuePoints", tsaf.Arr(tsaf.TAG_ARRAY_A, [])),
        ("audioAlignmentFingerprint", fp),
    ]
    doc = tsaf.Document((3, 3), root)
    db = djaydb.DjayDB(_library_db_with_user_data(tmp_path, str(track), doc))
    try:
        with pytest.raises(_Skip, match="audio alignment fingerprint"):
            _precheck_one(db, str(track), _ApplyArgs())
    finally:
        db.close()


def test_precheck_allows_known_safe_classnames(tmp_path):
    """Known-safe nested Obj classnames must pass the edit guard."""
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    tid = _title_id()
    doc = djaydb.build_user_data(
        "track1",
        tid,
        [{"time": 1.0, "number": 0, "comment": "First Beat"}],
    )
    doc.root.set("beatGridEdits", djaydb.build_beat_grid_edits(1.547))
    db = djaydb.DjayDB(_library_db_with_user_data(tmp_path, str(track), doc))
    try:
        key, existing, bpm = _precheck_one(db, str(track), _ApplyArgs())
        assert key == "track1"
        assert existing is not None
        assert bpm == pytest.approx(122.0)
    finally:
        db.close()


def test_precheck_safe_classname_walks_cue_arrays(tmp_path):
    """The guard must walk Arr items (cue arrays) without false positives."""
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    tid = _title_id()
    cues = [
        {"time": 1.0, "number": 0, "comment": "First Beat"},
        {"time": 32.0, "number": 3, "comment": "Drop"},
    ]
    doc = djaydb.build_user_data("track1", tid, cues)
    db = djaydb.DjayDB(_library_db_with_user_data(tmp_path, str(track), doc))
    try:
        _precheck_one(db, str(track), _ApplyArgs())
    finally:
        db.close()


def test_precheck_skips_unknown_obj_inside_array(tmp_path):
    """Unknown Obj classnames nested inside Arr items must trigger the unsafe skip."""
    track = tmp_path / "track.opus"
    track.write_bytes(b"")
    tid = _title_id()
    fp = tsaf.Obj("ADCAudioAlignmentFingerprint")
    fp.fields = [("payload", tsaf.Data(b"\xab\xcd"))]
    root = tsaf.Obj("ADCMediaItemUserData")
    root.fields = [
        ("uuid", "track1"),
        ("titleIDs", tsaf.Arr(tsaf.TAG_ARRAY_B, [tid])),
        ("cuePoints", tsaf.Arr(tsaf.TAG_ARRAY_A, [])),
        ("nestedAnalysis", tsaf.Arr(tsaf.TAG_ARRAY_A, [fp])),
    ]
    doc = tsaf.Document((3, 3), root)
    db = djaydb.DjayDB(_library_db_with_user_data(tmp_path, str(track), doc))
    try:
        with pytest.raises(_Skip, match="audio alignment fingerprint"):
            _precheck_one(db, str(track), _ApplyArgs())
    finally:
        db.close()
