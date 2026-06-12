"""Tests for CLI path expansion (directory batch mode)."""
from __future__ import annotations

import sqlite3

import pytest

from autohotcue import djaydb, tsaf
from autohotcue.cli import _Skip, _bpm_for, _djay_bpm_for_path, _expand_paths


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
